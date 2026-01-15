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
    labels: List[str]


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

    # This model from Open Images v4 is used because it returns class names ('entities').
    model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

    try:
        detector = hub.load(model_url)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load TF Hub detector. Last error: {e}")

    # The model expects a float32 tensor.
    arr = np.array(image.convert("RGB")).astype(np.uint8)
    img_tensor = tf.convert_to_tensor(arr)
    converted_img = tf.image.convert_image_dtype(img_tensor, tf.float32)[tf.newaxis, ...]

    # Run detection
    outputs = detector(converted_img)

    # Extract results
    boxes_n = outputs["detection_boxes"][0].numpy()
    scores = outputs["detection_scores"][0].numpy()
    labels_bytes = outputs["detection_class_entities"][0].numpy()

    h, w = arr.shape[0], arr.shape[1]
    n = min(max_detections, boxes_n.shape[0])

    boxes: List[Box] = []
    labels: List[str] = []
    for i in range(n):
        # Convert box coordinates from normalized to pixel values
        b = boxes_n[i]
        ymin, xmin, ymax, xmax = [float(x) for x in b]
        boxes.append(Box(xmin * w, ymin * h, xmax * w, ymax * h))
        # Decode class name from bytes to string
        labels.append(labels_bytes[i].decode("ascii"))

    return TfDetResult(
        boxes=boxes,
        scores=[float(s) for s in scores[:n]],
        labels=labels,
    )
