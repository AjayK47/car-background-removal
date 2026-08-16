import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForImageSegmentation

from safetensors.torch import load_file

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading fine-tuned BiRefNet model weights from runs/birefnet_car/best on device: {device}...")
model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to(device)
weights = load_file("runs/birefnet_car/best/model.safetensors")
model.load_state_dict(weights)
model = model.float()
model.eval()
print("BiRefNet model loaded successfully!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Load and convert image to RGBA
        input_image = Image.open(file.stream).convert("RGBA")
        width, height = input_image.size
        
        # Aspect-ratio preserving resize + letterbox padding to 1024x1024
        max_dim = max(width, height)
        scale = 1024.0 / max_dim
        nw, nh = int(width * scale), int(height * scale)
        rgb_img = input_image.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)
        
        pad_img = Image.new("RGB", (1024, 1024), (128, 128, 128))
        pad_x = (1024 - nw) // 2
        pad_y = (1024 - nh) // 2
        pad_img.paste(rgb_img, (pad_x, pad_y))
        
        # Normalize and convert to tensor
        img_arr = np.array(pad_img).astype(np.float32) / 255.0
        img_arr = (img_arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor_img = torch.tensor(img_arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor_img)
            logits = out[-1] if isinstance(out, (list, tuple)) else (out.logits if hasattr(out, "logits") else out)
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]
            if logits.dim() == 4:
                logits = logits[0, 0]
            elif logits.dim() == 3:
                logits = logits[0]
            
            # Crop padding away
            logits_crop = logits[pad_y : pad_y + nh, pad_x : pad_x + nw]
            # Upsample cropped logits to original dimensions
            logits_full = torch.nn.functional.interpolate(
                logits_crop.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
            )[0, 0]
            
            probs = torch.sigmoid(logits_full).cpu().numpy()
            
        mask_arr = (probs > 0.5)
            
        # 1. Generate Mask Overlay
        overlay = Image.new("RGBA", input_image.size, (0, 0, 0, 0))
        overlay_data = np.array(overlay)
        # Bright cyan mask overlay
        overlay_data[mask_arr] = [0, 240, 255, 120]  # Cyan color with ~47% opacity
        overlay = Image.fromarray(overlay_data)
        overlay_img = Image.alpha_composite(input_image, overlay)
        
        # 2. Generate Cutout (No background)
        cutout_data = np.array(input_image)
        cutout_data[~mask_arr, 3] = 0  # Zero out alpha on background
        cutout_img = Image.fromarray(cutout_data)
        
        # Encode to base64
        def to_b64(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            
        return jsonify({
            "original": to_b64(input_image.convert("RGB")),
            "overlay": to_b64(overlay_img),
            "cutout": to_b64(cutout_img)
        })
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Start the Flask app
    app.run(host="127.0.0.1", port=5001, debug=False)
