from __future__ import annotations

from typing import List, Sequence, Tuple

from .boxes import Box, iou


def apply_threshold(
    boxes: Sequence[Box],
    scores: Sequence[float],
    labels: Sequence[str],
    threshold: float,
) -> Tuple[List[Box], List[float], List[str]]:
    """
    Filters detection results based on a confidence score threshold.

    This function takes sequences of bounding boxes, scores, and labels, and
    returns only those that have a score greater than or equal to the specified
    threshold.

    :param boxes: A sequence of bounding box objects.
    :type boxes: Sequence[Box]
    :param scores: A sequence of confidence scores corresponding to each box.
    :type scores: Sequence[float]
    :param labels: A sequence of labels corresponding to each box.
    :type labels: Sequence[str]
    :param threshold: The minimum score for a detection to be kept.
    :type threshold: float
    :return: A tuple containing three lists: kept boxes, kept scores, and kept labels.
    :rtype: Tuple[List[Box], List[float], List[str]]
    :raises ValueError: If the input sequences have different lengths.
    """
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
    """
    Performs Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes.

    NMS is a conflict resolution algorithm that keeps the box with the highest
    score in a cluster of overlapping boxes and suppresses the others.

    :param boxes: A sequence of bounding box objects.
    :type boxes: Sequence[Box]
    :param scores: A sequence of confidence scores corresponding to each box.
    :type scores: Sequence[float]
    :param iou_threshold: The Intersection over Union (IoU) threshold. Boxes with
                          IoU greater than this value will be suppressed.
                          Defaults to 0.5.
    :type iou_threshold: float, optional
    :return: A list of indices of the boxes to keep.
    :rtype: List[int]
    :raises ValueError: If the input sequences have different lengths.
    """
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
