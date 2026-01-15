"""U-Net (segmentation) placeholder.

U-Net is primarily used for semantic/instance segmentation, not for detection.

WHY this file exists
--------------------
The LinkedIn series separates:
- Post 1: detection (boxes), operational contracts (threshold, IoU, NMS)
- Next post: segmentation (masks), where U-Net and related models fit naturally

This placeholder keeps the repo aligned with the series roadmap without forcing
a premature segmentation implementation into the detection MVP.
"""
