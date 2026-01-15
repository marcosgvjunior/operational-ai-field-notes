# tests/

These tests protect the operational contract primitives.

Scope:
- IoU invariants
- thresholding behavior
- NMS suppression
- visualization sanity

Framework adapters (PyTorch, TensorFlow, YOLO) are intentionally not tested here because
they depend on large external wheels and sometimes on network downloads.
