from __future__ import annotations

from PIL import Image

from src.vision.boxes import Box, iou
from src.vision.contracts import apply_threshold, nms
from src.vision.viz import draw_boxes


def test_iou_identical_is_one() -> None:
    a = Box(0.0, 0.0, 10.0, 10.0)
    b = Box(0.0, 0.0, 10.0, 10.0)
    assert iou(a, b) == 1.0


def test_iou_disjoint_is_zero() -> None:
    a = Box(0.0, 0.0, 10.0, 10.0)
    b = Box(20.0, 20.0, 30.0, 30.0)
    assert iou(a, b) == 0.0


def test_threshold_filters() -> None:
    boxes = [Box(0, 0, 10, 10), Box(0, 0, 5, 5)]
    scores = [0.9, 0.2]
    labels = ["a", "b"]
    b, s, l = apply_threshold(boxes, scores, labels, threshold=0.5)
    assert len(b) == 1
    assert s == [0.9]
    assert l == ["a"]


def test_nms_suppresses_overlap() -> None:
    boxes = [
        Box(0, 0, 10, 10),
        Box(1, 1, 9, 9),
        Box(20, 20, 30, 30),
    ]
    scores = [0.9, 0.6, 0.8]
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_draw_boxes_keeps_size() -> None:
    img = Image.new("RGB", (64, 64), color="white")
    out = draw_boxes(img, [Box(5, 5, 20, 20)], scores=[0.9])
    assert out.size == img.size
