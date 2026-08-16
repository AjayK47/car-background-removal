import os
import json
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import modal

# Define the container image with all training dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")  # Required for OpenCV
    .pip_install(
        "rfdetr[train]",
        "datasets",
        "opencv-python-headless",
        "pillow",
        "albumentations",
        "pytorch_lightning",
        "torch",
        "torchvision"
    )
)

app = modal.App("rf-detr-car-segmentation", image=image)

# Create a persistent Volume to save weights securely in the cloud
volume = modal.Volume.from_name("rf-detr-weights", create_if_missing=True)

@app.function(
    gpu="A10G",          # A10G GPU
    timeout=1800,        # 30-minute timeout limit
    volumes={"/weights": volume}, # Mount the persistent volume to /weights inside the container
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]
)
def train_remote():
    # Import inside the container function
    import cv2
    from datasets import load_dataset
    from rfdetr import RFDETRSeg2XLarge

    # Setup directories in the container's ephemeral /tmp storage
    tmp_dir = Path("/tmp")
    output_dir = tmp_dir / "processed_dataset"
    coco_dir = tmp_dir / "coco_dataset"
    
    train_img_dir = output_dir / "train" / "images"
    train_mask_dir = output_dir / "train" / "masks"
    val_img_dir = output_dir / "val" / "images"
    val_mask_dir = output_dir / "val" / "masks"
    
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Download & Preprocess (Binary Masks)
    print("--- Loading & Preprocessing Dataset from Hugging Face ---")
    dataset = load_dataset("aspis/car_background_removal_v2")
    train_split = dataset["train"]
    labeled_examples = [ex for ex in train_split if ex.get("status") == "LABELED"]
    
    random.seed(42)
    random.shuffle(labeled_examples)
    
    split_idx = int(len(labeled_examples) * 0.85)
    train_examples = labeled_examples[:split_idx]
    val_examples = labeled_examples[split_idx:]
    
    def process_and_save(examples, img_dir, mask_dir):
        for idx, ex in enumerate(examples):
            name = ex.get("name")
            image = ex.get("image")
            mask = ex.get("label.segmentation_bitmap")
            annotations = ex.get("label.annotations", [])
            if not image or not mask:
                continue
            
            # Resize image to a maximum dimension of 768 while maintaining aspect ratio
            image = image.convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            
            # Resize mask to match resized image size (use NEAREST to prevent interpolation artifacts)
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
            
            id_to_cat = {ann["id"]: ann["category_id"] for ann in annotations}
            mask_arr = np.array(mask)
            binary_mask = np.zeros_like(mask_arr, dtype=np.uint8)
            for ann_id, cat_id in id_to_cat.items():
                if cat_id in [0, 2]:
                    binary_mask[mask_arr == ann_id] = 255
            
            base_name = Path(name).stem
            image.save(img_dir / f"{base_name}.png", "PNG")
            Image.fromarray(binary_mask).save(mask_dir / f"{base_name}_mask.png", "PNG")

    process_and_save(train_examples, train_img_dir, train_mask_dir)
    process_and_save(val_examples, val_img_dir, val_mask_dir)
    print("Finished Binary preprocessing.")

    # 2. Convert to COCO Format
    print("\n--- Converting to COCO Format ---")
    for split, src_split in [("train", "train"), ("valid", "val")]:
        split_src_img_dir = output_dir / src_split / "images"
        split_src_mask_dir = output_dir / src_split / "masks"
        split_dest_dir = coco_dir / split
        split_dest_dir.mkdir(parents=True, exist_ok=True)
        
        coco_output = {
            "info": {"description": "Car Background Removal Dataset", "version": "1.0", "year": 2026},
            "images": [], "annotations": [],
            "categories": [{"id": 1, "name": "car", "supercategory": "vehicle"}]
        }
        
        image_paths = sorted(list(split_src_img_dir.glob("*.png")))
        annotation_id = 1
        for img_id, img_path in enumerate(image_paths, 1):
            file_name = img_path.name
            mask_path = split_src_mask_dir / f"{img_path.stem}_mask.png"
            shutil.copy2(img_path, split_dest_dir / file_name)
            with Image.open(img_path) as img:
                width, height = img.size
            
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Use RLE encoding instead of polygons to properly preserve window holes.
            # COCO polygon format UNIONS all polygons, filling holes.
            # RLE encodes the actual binary mask pixel-by-pixel, preserving holes.
            from pycocotools import mask as mask_utils
            binary_mask = (mask_img > 0).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")  # Make JSON serializable
            
            y_indices, x_indices = np.where(mask_img > 0)
            if len(x_indices) > 0:
                x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = float(np.sum(binary_mask))
            else:
                bbox, area = [0, 0, 0, 0], 0.0
                
            coco_output["images"].append({"id": img_id, "width": width, "height": height, "file_name": file_name})
            if area > 0:
                coco_output["annotations"].append({
                    "id": annotation_id, "image_id": img_id, "category_id": 1,
                    "segmentation": rle, "area": area, "bbox": bbox, "iscrowd": 1  # iscrowd=1 tells pycocotools to use RLE
                })
                annotation_id += 1
                
        with open(split_dest_dir / "_annotations.coco.json", "w") as f:
            json.dump(coco_output, f, indent=4)

    # Create dummy test split
    test_dir = coco_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    with open(test_dir / "_annotations.coco.json", "w") as f:
        json.dump({"images": [], "annotations": [], "categories": [{"id": 1, "name": "car"}]}, f)

    print("COCO Conversion Complete!")

    # 3. Train RF-DETR model
    print("\n--- Training RF-DETR Model on GPU ---")
    model = RFDETRSeg2XLarge()
    model.train(
        dataset_dir=str(coco_dir.resolve()),
        epochs=15,             # 15 epochs
        batch_size=2,          # Reduced from 8 to 2 to prevent validation OOM
        grad_accum_steps=8,    # Increased from 2 to 8 to keep effective batch size at 16
        output_dir="/weights", # Save directly to the mounted persistent Modal Volume
        device="cuda",
        progress_bar=False,    # Disable interactive bar in headless logs
        resolution=768         # Best resolution (must be divisible by patch_size * num_windows = 24)
    )
    print("Training Completed!")
    
    # Commit changes to the Volume explicitly so they are persisted immediately
    volume.commit()

    # 4. Load the best model checkpoint bytes
    checkpoint_file = Path("/weights/checkpoint_best_ema.pth")
    if not checkpoint_file.exists():
        checkpoint_file = Path("/weights/checkpoint_best_regular.pth")
        
    if not checkpoint_file.exists():
        # Fallback: check recursively for any .pth files
        parent_runs_dir = Path("/weights")
        pth_files = list(parent_runs_dir.rglob("*.pth"))
        if pth_files:
            checkpoint_file = pth_files[0]
        else:
            raise FileNotFoundError("No checkpoint file found in /weights")
            
    print(f"Reading checkpoint file '{checkpoint_file.name}' ({checkpoint_file.stat().st_size / 1024 / 1024:.2f} MB) from Volume...")
    with open(checkpoint_file, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    print("="*60)
    print(" Submitting training job to Modal container (A10G GPU)...")
    print(" NOTE: Using spawn() to trigger completely detached cloud training.")
    print(" You can safely close your terminal or turn off your internet now.")
    print("="*60 + "\n")
    
    try:
        # Spawn the remote function asynchronously (returns immediately)
        call = train_remote.spawn()
        print(f" SUCCESS: Job submitted successfully!")
        print(f" Remote Call ID: {call.object_id}")
        print("\nTo download the final model weights after training finishes (takes ~10 mins):")
        print("  modal volume get rf-detr-weights checkpoint_best_ema.pth runs/rf_detr_car/checkpoint_best_regular.pth")
        print("="*60)
    except Exception as e:
        print(f"\nFailed to submit job: {e}")

if __name__ == "__main__":
    main()
