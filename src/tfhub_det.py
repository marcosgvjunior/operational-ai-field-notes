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
    boxes: List[Box]
    scores: List[float]
    class_ids: List[int]


def run_tfhub_ssd_mobilenet(
    image: Image.Image,
    *,
    max_detections: int = 50,
) -> TfDetResult:
    """Run object detection with TF Hub SSD MobileNet.

    Dependencies
    ------------
    Requires tensorflow and tensorflow_hub. Install via:
    `pip install -r requirements-tf.txt`
    """
    if tf is None or hub is None:  # pragma: no cover
        raise RuntimeError("Missing TensorFlow/TF Hub. Install: pip install -r requirements-tf.txt")

    model_urls = [
        "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
        "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1",
    ]

    detector = None
    last_err = None
    for url in model_urls:
        try:
            detector = hub.load(url)
            last_err = None
            break
        except Exception as e:  # pragma: no cover
            last_err = e

    if detector is None:  # pragma: no cover
        raise RuntimeError(f"Failed to load TF Hub detector. Last error: {last_err}")

    arr = np.array(image.convert("RGB")).astype(np.uint8)
    tensor = tf.convert_to_tensor(arr)[tf.newaxis, ...]

    outputs = detector(tensor)
    boxes_n = outputs["detection_boxes"][0].numpy()
    scores = outputs["detection_scores"][0].numpy()
    classes = outputs["detection_classes"][0].numpy().astype(int)

    h, w = arr.shape[0], arr.shape[1]
    n = min(max_detections, boxes_n.shape[0])

    boxes: List[Box] = []
    for b in boxes_n[:n]:
        ymin, xmin, ymax, xmax = [float(x) for x in b]
        boxes.append(Box(xmin * w, ymin * h, xmax * w, ymax * h))

    return TfDetResult(
        boxes=boxes,
        scores=[float(s) for s in scores[:n]],
        class_ids=[int(c) for c in classes[:n]],
    )
