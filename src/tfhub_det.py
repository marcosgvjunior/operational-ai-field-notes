from __future__ import annotations

import json
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


with open("data/coco_labels.json", "r") as f:
    COCO_ID_TO_NAME = json.load(f)


def run_tfhub_ssd_mobilenet(image: Image.Image, *, max_detections: int = 50) -> TfDetResult:
    """
    Runs object detection using a pre-trained SSD MobileNet V2 model from TensorFlow Hub.

    This function requires `tensorflow` and `tensorflow_hub` to be installed.
    The model is trained on the COCO dataset.

    :param image: The input image to process.
    :type image: PIL.Image.Image
    :param max_detections: The maximum number of detections to return.
                           Defaults to 50.
    :type max_detections: int, optional
    :return: An object containing the detected boxes, scores, and labels.
    :rtype: TfDetResult
    :raises RuntimeError: If `tensorflow` or `tensorflow_hub` are not installed,
                          or if a callable detector function cannot be obtained from the model.
    """
    if tf is None or hub is None:  # pragma: no cover
        raise RuntimeError("Missing TensorFlow/TF Hub. Install: pip install -r requirements-tf.txt")

    model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

    # Some TF Hub modules are callable; otherwise use a callable signature.
    detector = model if callable(model) else (
        model.signatures.get("serving_default") or model.signatures.get("default")
    )
    if detector is None or not callable(detector):
        raise RuntimeError("Could not get a callable detector function.")

    # Minimal input conversion: PIL -> uint8 tensor with batch dim
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    x = tf.convert_to_tensor(arr)[tf.newaxis, ...]

    out = detector(x)

    boxes = np.array(out["detection_boxes"].numpy())
    scores = np.array(out["detection_scores"].numpy())
    classes = np.array(out["detection_classes"].numpy())

    # Remove batch dim if present
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 2:
        scores = scores[0]
    if classes.ndim == 2:
        classes = classes[0]

    # Safety guard: sometimes boxes can come flattened
    if boxes.ndim == 1:
        if boxes.size % 4 != 0:
            raise RuntimeError(f"Unexpected boxes shape {boxes.shape}")
        boxes = boxes.reshape(-1, 4)
    else:
        boxes = boxes.reshape(-1, 4)

    scores = scores.reshape(-1)
    classes = classes.reshape(-1)

    h, w = arr.shape[0], arr.shape[1]
    n = min(max_detections, len(scores), len(classes), boxes.shape[0])

    out_boxes: List[Box] = []
    out_scores: List[float] = []
    out_labels: List[str] = []

    for i in range(n):
        # TF returns normalized boxes: [ymin, xmin, ymax, xmax] in [0, 1]
        ymin, xmin, ymax, xmax = [float(v) for v in boxes[i]]
        out_boxes.append(Box(xmin * w, ymin * h, xmax * w, ymax * h))
        out_scores.append(float(scores[i]))

        cid = int(classes[i])
        out_labels.append(COCO_ID_TO_NAME.get(str(cid), f"coco_{cid}"))

    return TfDetResult(boxes=out_boxes, scores=out_scores, labels=out_labels)
