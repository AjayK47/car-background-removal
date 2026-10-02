# Car Background Removal

Local Flask application for generating a car mask overlay and transparent-background cutout with a fine-tuned [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) model.

## What Is Included

- `app.py`: local inference server and HTTP API.
- `templates/index.html`: browser UI.
- Training scripts used to fine-tune the car segmentation model.

The custom checkpoint is intentionally **not** included in this repository. It is approximately 844 MiB and must be shared separately by the repository owner.

## Requirements

- Python 3.12
- macOS, Linux, or Windows
- A copy of the fine-tuned `model.safetensors` checkpoint

Apple Silicon Macs use Metal Performance Shaders (MPS) automatically. CUDA is used automatically when available; otherwise the app runs on CPU.

## Setup

```bash
git clone git@github.com:AjayK47/car-background-removal.git
cd car-background-removal
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download or copy the custom checkpoint to:

```text
runs/birefnet_car/best/model.safetensors
```

Alternatively, put it elsewhere and set `MODEL_PATH` before starting the app:

```bash
MODEL_PATH=/absolute/path/to/model.safetensors python app.py
```

Start the server:

```bash
python app.py
```

Open <http://127.0.0.1:5001>, upload a car image, and choose either a cyan mask overlay or a transparent-background cutout.

## API

`POST /api/process` accepts a multipart form upload named `image`.

```bash
curl -F "image=@car.jpg" http://127.0.0.1:5001/api/process
```

The response contains data-URL encoded `original`, `overlay`, and `cutout` PNGs.

## Model Distribution

Share the custom checkpoint through a private Hugging Face model repository, GitHub Release, Google Drive, or another access-controlled file store. Do not commit it directly to Git: `model.safetensors` is intentionally excluded by `.gitignore`.

Before distributing the checkpoint outside your team, review the licenses for the base BiRefNet model and the training data used for fine-tuning.
