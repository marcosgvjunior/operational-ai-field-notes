# operational-ai-field-notes

Technical field notes on ML systems: objectives, metrics, feedback loops, and where symbolic depth breaks.

This is a study-support repository designed to back LinkedIn posts with:

- small, typed, testable utilities (the "operational contract")
- framework demos in separate notebooks (PyTorch, TensorFlow, and optional YOLO)
- tiny images so the whole repo stays reproducible

## What this repo demonstrates

Object detection becomes operational when you define what counts as "correct":

candidates (boxes + scores) -> thresholding -> IoU -> NMS -> final detections

Those choices do not just evaluate a model.
They shape what the system becomes consistent at producing.

## Layout

- `src/` typed, self-contained utilities plus thin framework adapters
- `tests/` PyTest suite for the contract primitives (framework installs are not required for tests)
- `notebooks/` step-by-step demos
- `data/` tiny inputs and optional outputs

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

pytest
ruff check .
```

Framework notebooks install their own extras:

- PyTorch: `requirements-torch.txt`
- TensorFlow: `requirements-tf.txt`
- YOLO: `requirements-yolo.txt`

## Notes on architectures

This repo includes detection demos aligned with the Post 1 scope:
- SSD + MobileNet (PyTorch and TensorFlow)

YOLO is also a detector, so it is included as an optional notebook.
U-Net is for segmentation, not detection, so it is parked as a stub for the next post.
