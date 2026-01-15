from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in XYXY format."""

    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        w = max(0.0, self.x2 - self.x1)
        h = max(0.0, self.y2 - self.y1)
        return w * h


def iou(a: Box, b: Box) -> float:
    """Intersection over Union (IoU), overlap agreement primitive."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = a.area() + b.area() - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
