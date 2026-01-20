from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

from .boxes import Box

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except Exception:  # pragma: no cover
    tf = None  # type: ignore
    hub = None  # type: ignore


@dataclass(frozen=True)
class TfDetResult:
    """
    Holds the results of a TensorFlow Hub object detection.

    Note: This class is used by both COCO and Open Images models, but is defined
    in both modules to keep them self-contained.

    :param boxes: A list of detected bounding boxes.
    :type boxes: List[Box]
    :param scores: A list of confidence scores for each detection.
    :type scores: List[float]
    :param labels: A list of class labels for each detection.
    :type labels: List[str]
    """
    boxes: List[Box]
    scores: List[float]
    labels: List[str]


def run_tfhub_ssd_mobilenet(image: Image.Image, *, max_detections: int = 50) -> TfDetResult:
    """
    Runs object detection using a TF Hub SSD MobileNet V2 model trained on Open Images.

    This model returns class labels directly as strings. This function requires
    `tensorflow` and `tensorflow_hub` to be installed.

    :param image: The input image to process.
    :type image: PIL.Image.Image
    :param max_detections: The maximum number of detections to return. Defaults to 50.
    :type max_detections: int, optional
    :return: An object containing the detected boxes, scores, and labels.
    :rtype: TfDetResult
    :raises RuntimeError: If TensorFlow/TF Hub are not installed or a callable detector
                          cannot be obtained.
    """
    if tf is None or hub is None:  # pragma: no cover
        raise RuntimeError("Missing TensorFlow/TF Hub. Install: pip install -r requirements-tf.txt")

    model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")
    detector = model.signatures.get("default") or model.signatures.get("serving_default")
    if detector is None or not callable(detector):
        raise RuntimeError("Could not get a callable detector function.")

    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    x = tf.image.convert_image_dtype(tf.convert_to_tensor(arr), tf.float32)[tf.newaxis, ...]

    out = detector(x)

    boxes = np.array(out["detection_boxes"].numpy())
    scores = np.array(out["detection_scores"].numpy())
    labels = np.array(out["detection_class_entities"].numpy())

    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 2:
        scores = scores[0]
    if labels.ndim == 2:
        labels = labels[0]

    if boxes.ndim == 1:
        if boxes.size % 4 != 0:
            raise RuntimeError(f"Unexpected boxes shape {boxes.shape}")
        boxes = boxes.reshape(-1, 4)
    else:
        boxes = boxes.reshape(-1, 4)

    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    h, w = arr.shape[0], arr.shape[1]
    n = min(max_detections, len(scores), len(labels), boxes.shape[0])

    out_boxes: List[Box] = []
    out_scores: List[float] = []
    out_labels: List[str] = []

    for i in range(n):
        ymin, xmin, ymax, xmax = [float(v) for v in boxes[i]]
        out_boxes.append(Box(xmin * w, ymin * h, xmax * w, ymax * h))
        out_scores.append(float(scores[i]))

        lb = labels[i]
        out_labels.append(lb.decode("ascii", errors="ignore") if isinstance(lb, (bytes, np.bytes_)) else str(lb))

    return TfDetResult(boxes=out_boxes, scores=out_scores, labels=out_labels)
