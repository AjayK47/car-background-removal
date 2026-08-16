import io
import os
import tarfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Container image – all training deps
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch==2.3.1 torchvision==0.18.1 transformers==4.41.2 pillow opencv-python-headless 'albumentations>=2.0.0' tqdm numpy"
    )
)

app = modal.App("segformer-car-segmentation", image=image)

# Two volumes: one for the dataset, one for saved weights
data_volume    = modal.Volume.from_name("segformer-dataset",  create_if_missing=True)
weights_volume = modal.Volume.from_name("rf-detr-weights",    create_if_missing=True)

# ---------------------------------------------------------------------------
# Step 1: Upload dataset to Modal Volume (run locally, uploads chunks)
# ---------------------------------------------------------------------------
@app.function(volumes={"/data": data_volume})
def upload_dataset(chunk: bytes, rel_path: str):
    """Receive one file at a time and write it into the volume."""
    from pathlib import Path
    dest = Path("/data") / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(chunk)


@app.local_entrypoint()
def main():
    import os

    dataset_path = Path("processed_dataset_clean")
    all_files = [
        p for p in dataset_path.rglob("*") if p.is_file()
    ]
    print(f"Uploading {len(all_files)} files from '{dataset_path}' to Modal volume...")

    # Upload in parallel batches
    calls = []
    for fpath in all_files:
        rel = fpath.relative_to(dataset_path.parent)  # keeps "processed_dataset_clean/..."
        data = fpath.read_bytes()
        calls.append((data, str(rel)))

    # Starmap – fan-out uploads
    list(upload_dataset.starmap(calls))
    print(f"Dataset upload complete ({len(all_files)} files).")

    # Now kick off training
    print("Starting SegFormer training on A10G...")
    result = train_segformer.remote()
    print(f"\n=== Training complete ===  Best mIoU: {result['best_miou']:.4f}")


# ---------------------------------------------------------------------------
# Step 2: Training function – reads dataset from volume
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        "/data":    data_volume,
        "/weights": weights_volume,
    },
)
def train_segformer():
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import SegformerForSemanticSegmentation
    from PIL import Image
    from pathlib import Path
    from tqdm import tqdm
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 2          # 0 = background/window  |  1 = car body
    IMG_SIZE    = 1024
    EPOCHS      = 80
    BATCH_SIZE  = 4
    LR          = 8e-5
    # Dataset lives at /data/processed_dataset_clean/
    DATA_ROOT   = Path("/data/processed_dataset_clean")
    SAVE_DIR    = Path("/weights/segformer_car")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device  : {DEVICE}")
    print(f"Data    : {DATA_ROOT}")
    print(f"Epochs  : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")

    n_train = len(list((DATA_ROOT / "train" / "images").glob("*.png")))
    n_val   = len(list((DATA_ROOT / "val"   / "images").glob("*.png")))
    print(f"Train   : {n_train}  |  Val: {n_val}")

    # ------------------------------------------------------------------
    # Augmentations  (albumentations 2.x API)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
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
            # black(0) = car body -> class 1  |  white(255) = background -> class 0
            mask = (mask < 128).astype(np.int64)
            out  = self.aug(image=img, mask=mask)
            return out["image"].float(), torch.tensor(out["mask"], dtype=torch.long)

    train_dl = DataLoader(
        CarDataset("train", train_aug),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dl = DataLoader(
        CarDataset("val", val_aug),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Model: SegFormer-B2 pretrained on ADE20K, re-headed for 2 classes
    # ------------------------------------------------------------------
    print("\n--- Loading SegFormer-B2 (nvidia/segformer-b2-finetuned-ade-512-512) ---")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        id2label={0: "background", 1: "car"},
        label2id={"background": 0, "car": 1},
        ignore_mismatched_sizes=True,   # re-initialises the final classifier head
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # Loss / Optimiser / Scheduler
    # ------------------------------------------------------------------
    criterion     = nn.CrossEntropyLoss(ignore_index=255)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )

    def compute_iou(pred, target, cls):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        return (inter / union).item() if union > 0 else float("nan")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_iou = 0.0
    history  = []

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_dl, desc=f"Ep {epoch:02d}/{EPOCHS} [train]", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = nn.functional.interpolate(
                model(pixel_values=imgs).logits,
                size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # ---- Validate ----
        model.eval()
        val_loss, iou_bg, iou_car = 0.0, [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_dl, desc=f"Ep {epoch:02d}/{EPOCHS} [val]  ", leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = nn.functional.interpolate(
                    model(pixel_values=imgs).logits,
                    size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
                val_loss += criterion(logits, masks).item()
                preds = logits.argmax(dim=1)
                for p, m in zip(preds, masks):
                    iou_bg.append(compute_iou(p, m, 0))
                    iou_car.append(compute_iou(p, m, 1))

        val_loss  /= len(val_dl)
        miou_bg    = float(np.nanmean(iou_bg))
        miou_car   = float(np.nanmean(iou_car))
        miou       = (miou_bg + miou_car) / 2
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"IoU_bg={miou_bg:.4f} | IoU_car={miou_car:.4f} | mIoU={miou:.4f}"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "iou_bg": miou_bg, "iou_car": miou_car, "miou": miou
        })

        # Save best checkpoint every time mIoU improves
        if miou > best_iou:
            best_iou = miou
            model.save_pretrained(str(SAVE_DIR / "best"))
            print(f"  ✓ New best mIoU={best_iou:.4f} — saved to /weights/segformer_car/best")

    # Save final checkpoint + history
    model.save_pretrained(str(SAVE_DIR / "final"))
    with open(SAVE_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    weights_volume.commit()

    print(f"\nTraining complete!  Best mIoU: {best_iou:.4f}")
    print(f"Checkpoints at: /weights/segformer_car/best  &  /weights/segformer_car/final")
    return {"best_miou": best_iou, "epochs": EPOCHS, "history": history}
