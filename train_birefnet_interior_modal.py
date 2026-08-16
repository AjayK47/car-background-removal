import io
import os
import tarfile
from pathlib import Path
import modal

# ---------------------------------------------------------------------------
# Container image with BiRefNet dependencies
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch==2.3.1 torchvision==0.18.1 transformers==4.41.2 pillow opencv-python-headless 'albumentations>=2.0.0' tqdm numpy timm einops kornia"
    )
)

app = modal.App("birefnet-interior-segmentation", image=image)

# Persistent volumes for dataset and weights
data_volume    = modal.Volume.from_name("segformer-dataset",  create_if_missing=True)
weights_volume = modal.Volume.from_name("rf-detr-weights",    create_if_missing=True)

# ---------------------------------------------------------------------------
# Upload Dataset to Volume
# ---------------------------------------------------------------------------
@app.function(volumes={"/data": data_volume})
def upload_file(chunk: bytes, rel_path: str):
    dest = Path("/data") / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(chunk)

def upload_interior_dataset_if_needed():
    dataset_path = Path("interior_dataset")
    if not dataset_path.exists():
        print(f"Error: Local path {dataset_path} not found.")
        return
    all_files = [p for p in dataset_path.rglob("*") if p.is_file()]
    print(f"Uploading {len(all_files)} files from interior_dataset to Modal volume...")
    calls = []
    for fpath in all_files:
        rel = fpath.relative_to(dataset_path.parent)
        calls.append((fpath.read_bytes(), str(rel)))
    list(upload_file.starmap(calls))
    print("Interior dataset upload complete.")

# ---------------------------------------------------------------------------
# Remote training function for BiRefNet on Interior Dataset
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={"/data": data_volume, "/weights": weights_volume},
)
def train_birefnet_interior(epochs: int = 50, lr: float = 3e-5):
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForImageSegmentation
    from PIL import Image
    from pathlib import Path
    from tqdm import tqdm
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE    = 512
    BATCH_SIZE  = 2
    DATA_ROOT   = Path("/data/interior_dataset")
    SAVE_DIR    = Path("/weights/birefnet_interior")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================")
    print(f"   Fine-Tuning Model: BiRefNet (Interior Dataset)")
    print(f"   GPU: NVIDIA A10G | Device: {DEVICE}")
    print(f"   Epochs: {epochs} | Batch: {BATCH_SIZE} | LR: {lr}")
    print(f"==================================================\n")

    # Augmentations
    train_aug = A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_aug = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    class InteriorDataset(Dataset):
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
            # 255 = foreground car interior (class 1.0) | 0 = background window (class 0.0)
            mask = (mask >= 128).astype(np.float32)
            out  = self.aug(image=img, mask=mask)
            return out["image"].float(), torch.tensor(out["mask"], dtype=torch.float32)

    # drop_last=True ensures BatchNorm never receives a single sample batch
    train_dl = DataLoader(InteriorDataset("train", train_aug), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(InteriorDataset("val",   val_aug),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    # ------------------------------------------------------------------
    # BiRefNet Model Initialization
    # ------------------------------------------------------------------
    print("Loading pre-trained BiRefNet model weights...")
    model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    def extract_logits(out):
        """Recursively unwrap nested lists/tuples until we reach the main torch Tensor."""
        while isinstance(out, (list, tuple)):
            out = out[0] if len(out) > 0 else out
        if hasattr(out, "logits"):
            out = out.logits
        while isinstance(out, (list, tuple)):
            out = out[0]
        return out

    def compute_iou(pred_binary, target_binary):
        inter = (pred_binary & target_binary).sum().float()
        union = (pred_binary | target_binary).sum().float()
        return (inter / union).item() if union > 0 else float("nan")

    best_iou = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for imgs, masks in tqdm(train_dl, desc=f"[BiRefNet-Interior] Ep {epoch:02d}/{epochs} [train]", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(imgs)
                logits = extract_logits(out)
                if logits.dim() == 4 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                logits = nn.functional.interpolate(logits.unsqueeze(1), size=masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            
        train_loss /= len(train_dl)

        # Validation
        model.eval()
        val_loss, iou_bg, iou_car = 0.0, [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_dl, desc=f"[BiRefNet-Interior] Ep {epoch:02d}/{epochs} [val]", leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(imgs)
                    logits = extract_logits(out)
                    if logits.dim() == 4 and logits.shape[1] == 1:
                        logits = logits.squeeze(1)
                    logits = nn.functional.interpolate(logits.unsqueeze(1), size=masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                    val_loss += nn.functional.binary_cross_entropy_with_logits(logits, masks).item()
                    preds = (torch.sigmoid(logits) > 0.5)
                    targets = (masks > 0.5)

                for p, t in zip(preds, targets):
                    iou_car.append(compute_iou(p, t))
                    iou_bg.append(compute_iou(~p, ~t))

        val_loss /= len(val_dl)
        miou_bg  = float(np.nanmean(iou_bg))
        miou_car = float(np.nanmean(iou_car))
        miou     = (miou_bg + miou_car) / 2
        scheduler.step()

        print(f"[BiRefNet-Interior] Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | IoU_bg={miou_bg:.4f} | IoU_interior={miou_car:.4f} | mIoU={miou:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "iou_bg": miou_bg, "iou_interior": miou_car, "miou": miou})

        if miou > best_iou:
            best_iou = miou
            model.save_pretrained(str(SAVE_DIR / "best"))
            print(f"  ✓ [BiRefNet-Interior] New best mIoU={best_iou:.4f} saved to /weights/birefnet_interior/best")

    with open(SAVE_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    weights_volume.commit()
    return {"best_miou": best_iou, "history": history}

# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(epochs: int = 50):
    upload_interior_dataset_if_needed()
    print(f"\n>>> Starting Modal run for BiRefNet on Interior Dataset ({epochs} epochs)...")
    res = train_birefnet_interior.remote(epochs=epochs)
    print(f">>> Finished BiRefNet! Best mIoU: {res['best_miou']:.4f}")
