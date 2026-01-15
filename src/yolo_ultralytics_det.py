from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PIL import Image

from .boxes import Box

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


@dataclass(frozen=True)
class YoloDetResult:
    boxes: List[Box]
    scores: List[float]
    labels: List[str]


def run_yolo_ultralytics(
    image: Image.Image,
    *,
    model_name: str = "yolov8n.pt",
    max_detections: int = 50,
) -> YoloDetResult:
    """Run YOLO detection via ultralytics.

    WHY: YOLO is a detector, so it belongs in the detection scope. This notebook is optional
    because it adds a heavier dependency.

    Install
    -------
    `pip install -r requirements-yolo.txt`
    """
    if YOLO is None:  # pragma: no cover
        raise RuntimeError("Missing ultralytics. Install: pip install -r requirements-yolo.txt")

    model = YOLO(model_name)
    # ultralytics accepts PIL images directly
    results = model.predict(image, verbose=False, max_det=max_detections)

    r = results[0]
    boxes: List[Box] = []
    scores: List[float] = []
    labels: List[str] = []

    if r.boxes is None:
        return YoloDetResult(boxes=[], scores=[], labels=[])

    xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
    conf = r.boxes.conf.cpu().numpy().astype(float)
    cls = r.boxes.cls.cpu().numpy().astype(int)

    # Get the class names from the model
    names = model.names

    n = min(max_detections, xyxy.shape[0])
    for i in range(n):
        b = xyxy[i]
        class_id = int(cls[i])
        boxes.append(Box(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        scores.append(float(conf[i]))
        labels.append(names[class_id])

    return YoloDetResult(boxes=boxes, scores=scores, labels=labels)
