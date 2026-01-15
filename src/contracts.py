from __future__ import annotations

from typing import List, Sequence, Tuple

from .boxes import Box, iou


def apply_threshold(
    boxes: Sequence[Box],
    scores: Sequence[float],
    threshold: float,
) -> Tuple[List[Box], List[float]]:
    """Decision gate: accept boxes with score >= threshold."""
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")

    kept_b: List[Box] = []
    kept_s: List[float] = []
    for b, s in zip(boxes, scores):
        score = float(s)
        if score >= threshold:
            kept_b.append(b)
            kept_s.append(score)
    return kept_b, kept_s


def nms(
    boxes: Sequence[Box],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
) -> List[int]:
    """Conflict resolver: keep the strongest claim, suppress overlapping duplicates."""
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []

    while order:
        i = order.pop(0)
        keep.append(i)

        remaining: List[int] = []
        for j in order:
            if iou(boxes[i], boxes[j]) <= iou_threshold:
                remaining.append(j)
        order = remaining

    return keep
