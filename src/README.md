# src/

Typed, self-contained utilities plus thin adapters for framework inference.

## "Self-contained" in practical terms

- No hidden global state.
- Functions are explicit about inputs and outputs.
- Modules are small and cohesive.

This matches the intent behind:
- SRP (Single Responsibility Principle): one module, one reason to change.
- high cohesion: functions in a module relate to one responsibility.
- low coupling: contract utilities do not depend on frameworks.

Cyclomatic complexity stays low because each function is short and linear.

## Modules

- `boxes.py`
  - `Box` dataclass (XYXY)
  - `iou` overlap agreement

- `contracts.py`
  - `apply_threshold` decision gate
  - `nms` conflict resolver

- `viz.py`
  - `draw_boxes` for inspection

Framework adapters (optional dependencies, used in notebooks):

- `torchvision_det.py`
  - SSD + MobileNet detector via torchvision

- `tfhub_det.py`
  - SSD + MobileNet detector via TensorFlow Hub

- `yolo_ultralytics_det.py`
  - YOLO detector via ultralytics (optional)

Segmentation placeholder:

- `unet_segmentation_stub.py`
  - U-Net is segmentation (next post), so this file is a placeholder + notes.
