from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """
    Axis-aligned bounding box in XYXY format.

    :param x1: The x-coordinate of the top-left corner.
    :param y1: The y-coordinate of the top-left corner.
    :param x2: The x-coordinate of the bottom-right corner.
    :param y2: The y-coordinate of the bottom-right corner.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        """Calculates the area of the bounding box."""
        w = max(0.0, self.x2 - self.x1)
        h = max(0.0, self.y2 - self.y1)
        return w * h


def iou(a: Box, b: Box) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    IoU is a measure of the overlap between two bounding boxes.

    :param a: The first bounding box.
    :type a: Box
    :param b: The second bounding box.
    :type b: Box
    :return: The IoU score, a float between 0.0 and 1.0.
    :rtype: float
    """
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
