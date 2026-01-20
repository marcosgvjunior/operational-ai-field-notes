from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

from .boxes import Box

try:
    import torch
    import torchvision
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    torchvision = None  # type: ignore


@dataclass(frozen=True)
class TorchDetResult:
    """
    Holds the results of a TorchVision object detection.

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


def run_torchvision_ssd_mobilenet(
    image: Image.Image,
    *,
    max_detections: int = 50,
) -> TorchDetResult:
    """
    Runs object detection using a pre-trained SSDlite MobileNet V3 model from TorchVision.

    This function requires `torch` and `torchvision` to be installed.

    :param image: The input image to process.
    :type image: PIL.Image.Image
    :param max_detections: The maximum number of detections to return.
                           Defaults to 50.
    :type max_detections: int, optional
    :return: An object containing the detected boxes, scores, and labels.
    :rtype: TorchDetResult
    :raises RuntimeError: If `torch` or `torchvision` are not installed.
    """
    if torch is None or torchvision is None:  # pragma: no cover
        raise RuntimeError(
            "Missing torch/torchvision. Install with: pip install -r requirements-torch.txt"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights).to(device)
    model.eval()

    # Get class names from the model's metadata
    coco_classes = weights.meta["categories"]

    preprocess = weights.transforms()
    x = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)[0]

    boxes_xyxy = out["boxes"].detach().cpu().numpy().astype(float)
    scores = out["scores"].detach().cpu().numpy().astype(float).tolist()
    labels_int = out["labels"].detach().cpu().numpy().astype(int)

    n = min(max_detections, boxes_xyxy.shape[0])
    
    boxes: List[Box] = [
        Box(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes_xyxy[:n]
    ]
    
    # Convert integer labels to string labels
    labels_str = [coco_classes[i] for i in labels_int[:n]]

    return TorchDetResult(
        boxes=boxes,
        scores=scores[:n],
        labels=labels_str,
    )
