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
    """
    Holds the results of a YOLO detection.

    :param boxes: A list of detected bounding boxes.
    :param scores: A list of confidence scores for each detection.
    :param labels: A list of class labels for each detection.
    """
    boxes: List[Box]
    scores: List[float]
    labels: List[str]


def run_yolo_ultralytics(
    image: Image.Image,
    *,
    model_name: str = "yolov8n.pt",
    max_detections: int = 50,
) -> YoloDetResult:
    """
    Runs YOLO object detection on an image using the ultralytics library.

    This function requires the 'ultralytics' package to be installed.

    :param image: The input image to process.
    :type image: PIL.Image.Image
    :param model_name: The name of the YOLO model file to use.
                       Defaults to "yolov8n.pt".
    :type model_name: str, optional
    :param max_detections: The maximum number of detections to return.
                           Defaults to 50.
    :type max_detections: int, optional
    :return: An object containing the detected boxes, scores, and labels.
    :rtype: YoloDetResult
    :raises RuntimeError: If the 'ultralytics' library is not installed.
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
