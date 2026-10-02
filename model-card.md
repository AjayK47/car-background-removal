---
base_model: ZhengPeng7/BiRefNet
tags:
- image-segmentation
- background-removal
- car
---

# Car Background Removal BiRefNet

Fine-tuned BiRefNet checkpoint for segmenting cars and producing background masks.

## Usage

Use the inference application at <https://github.com/AjayK47/car-background-removal>. It downloads `model.safetensors` automatically on first run.

## Files

- `model.safetensors`: fine-tuned checkpoint, 844 MiB.
- `config.json`: training/export configuration retained with the checkpoint.

## Intended Use

This model is intended for foreground segmentation of vehicle images. It returns a per-pixel foreground mask that can be used for background removal or a transparent PNG cutout.

## Limitations

Output quality varies with image composition, occlusion, lighting, reflections, and vehicle type. It should not be used as the sole basis for safety-critical or high-stakes decisions.

## Provenance And Licensing

This checkpoint was fine-tuned from [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet). Review the base-model license and the licenses of all fine-tuning data before redistributing or using this model commercially.

SHA-256 for `model.safetensors`:

```text
d10aea1fbee3c059b90eaab3399ca5ed4121c0c3647796d588a0c9510c383c5b
```
