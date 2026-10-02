# Car Background Removal

Local Flask application for generating a car mask overlay and transparent-background cutout with a fine-tuned [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) model.

## What Is Included

- `app.py`: local inference server and HTTP API.
- `templates/index.html`: browser UI.
- Training scripts used to fine-tune the car segmentation model.

The custom checkpoint is intentionally **not** included in this repository. It is approximately 844 MiB and is downloaded automatically from [Ajayk/car-background-removal-birefnet](https://huggingface.co/Ajayk/car-background-removal-birefnet) on first run.

## Requirements

- Python 3.12
- macOS, Linux, or Windows
- Internet access for the first run, to download the fine-tuned `model.safetensors` checkpoint

Apple Silicon Macs use Metal Performance Shaders (MPS) automatically. CUDA is used automatically when available; otherwise the app runs on CPU.

## Setup

```bash
git clone git@github.com:AjayK47/car-background-removal.git
cd car-background-removal
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The app downloads and caches the fine-tuned checkpoint automatically on first run. To use a checkpoint you already have locally, set `MODEL_PATH` instead:

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

The checkpoint is published at [Ajayk/car-background-removal-birefnet](https://huggingface.co/Ajayk/car-background-removal-birefnet). Do not commit model weights directly to Git: `model.safetensors` is intentionally excluded by `.gitignore`.

Before distributing the checkpoint outside your team, review the licenses for the base BiRefNet model and the training data used for fine-tuning.
