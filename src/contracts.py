from __future__ import annotations

from typing import List, Sequence, Tuple

from .boxes import Box, iou


def apply_threshold(
    boxes: Sequence[Box],
    scores: Sequence[float],
    labels: Sequence[str],
    threshold: float,
) -> Tuple[List[Box], List[float], List[str]]:
    """Decision gate: accept boxes with score >= threshold."""
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("boxes, scores, and labels must have the same length")

    kept_b: List[Box] = []
    kept_s: List[float] = []
    kept_l: List[str] = []

    for b, s, l in zip(boxes, scores, labels):
        score = float(s)
        if score >= threshold:
            kept_b.append(b)
            kept_s.append(score)
            kept_l.append(l)
    return kept_b, kept_s, kept_l


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
