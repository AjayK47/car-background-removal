import io
import os
import tarfile
from pathlib import Path
import modal

# ---------------------------------------------------------------------------
# Container image with segmentation_models_pytorch (U-Net)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch==2.3.1 torchvision==0.18.1 segmentation-models-pytorch pillow opencv-python-headless 'albumentations>=2.0.0' tqdm numpy timm"
    )
)

app = modal.App("unet-car-segmentation", image=image)

# Persistent volume for dataset and weights
data_volume    = modal.Volume.from_name("segformer-dataset",  create_if_missing=True)
weights_volume = modal.Volume.from_name("rf-detr-weights",    create_if_missing=True)

# ---------------------------------------------------------------------------
# Remote training function for U-Net
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={"/data": data_volume, "/weights": weights_volume},
)
def train_unet(epochs: int = 60, lr: float = 1e-4):
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import segmentation_models_pytorch as smp
    from PIL import Image
    from pathlib import Path
    from tqdm import tqdm
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 2
    IMG_SIZE    = 512
    BATCH_SIZE  = 8
    DATA_ROOT   = Path("/data/processed_dataset_clean")
    SAVE_DIR    = Path("/weights/unet_car")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================")
    print(f"   Training Model: U-Net (ResNet34 Backbone)")
    print(f"   Device: {DEVICE} | Resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Epochs: {epochs} | Batch: {BATCH_SIZE} | LR: {lr}")
    print(f"==================================================\n")

    # Augmentations
    train_aug = A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    class CarDataset(Dataset):
        def __init__(self, split, aug):
            self.img_dir  = DATA_ROOT / split / "images"
            self.mask_dir = DATA_ROOT / split / "masks"
            self.aug      = aug
            self.imgs     = sorted(self.img_dir.glob("*.png"))

        def __len__(self): return len(self.imgs)

        def __getitem__(self, idx):
            img_path  = self.imgs[idx]
            mask_path = self.mask_dir / f"{img_path.stem}_mask.png"
            img  = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            # Black (0) = car body (class 1) | White (255) = background/window (class 0)
            mask = (mask < 128).astype(np.int64)
            out  = self.aug(image=img, mask=mask)
            return out["image"].float(), torch.tensor(out["mask"], dtype=torch.long)

    train_dl = DataLoader(CarDataset("train", train_aug), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(CarDataset("val",   val_aug),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # U-Net Model Initialization with ResNet34 Encoder
    # ------------------------------------------------------------------
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    def compute_iou(pred, target, cls):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        return (inter / union).item() if union > 0 else float("nan")

    best_iou = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_dl, desc=f"[U-Net] Ep {epoch:02d}/{epochs} [train]", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validation
        model.eval()
        val_loss, iou_bg, iou_car = 0.0, [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_dl, desc=f"[U-Net] Ep {epoch:02d}/{epochs} [val]", leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                preds = logits.argmax(dim=1)

                for p, m in zip(preds, masks):
                    iou_bg.append(compute_iou(p, m, 0))
                    iou_car.append(compute_iou(p, m, 1))

        val_loss /= len(val_dl)
        miou_bg  = float(np.nanmean(iou_bg))
        miou_car = float(np.nanmean(iou_car))
        miou     = (miou_bg + miou_car) / 2
        scheduler.step()

        print(f"[U-Net] Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | IoU_bg={miou_bg:.4f} | IoU_car={miou_car:.4f} | mIoU={miou:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "iou_bg": miou_bg, "iou_car": miou_car, "miou": miou})

        if miou > best_iou:
            best_iou = miou
            torch.save(model.state_dict(), SAVE_DIR / "best.pth")
            print(f"  ✓ [U-Net] New best mIoU={best_iou:.4f} saved to /weights/unet_car/best.pth")

    with open(SAVE_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    weights_volume.commit()
    return {"best_miou": best_iou, "history": history}

# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(epochs: int = 60):
    print(f"\n>>> Starting Modal run for U-Net (ResNet34) ({epochs} epochs)...")
    res = train_unet.remote(epochs=epochs)
    print(f">>> Finished U-Net! Best mIoU: {res['best_miou']:.4f}")
