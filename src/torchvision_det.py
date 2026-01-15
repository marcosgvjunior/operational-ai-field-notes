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
    boxes: List[Box]
    scores: List[float]
    labels: List[str]


def run_torchvision_ssd_mobilenet(
    image: Image.Image,
    *,
    max_detections: int = 50,
) -> TorchDetResult:
    """Run object detection with torchvision SSDlite MobileNet V3.

    WHY: SSD + MobileNet is a typical efficient detector, matching the Post 1 scope.

    Dependencies
    ------------
    Requires torch and torchvision. Install via:
    `pip install -r requirements-torch.txt`
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
    scores = out["scores"].detach().cpu().numpy().astype(float)
    labels_int = out["labels"].detach().cpu().numpy().astype(int)

    n = min(max_detections, boxes_xyxy.shape[0])
    
    boxes: List[Box] = [
        Box(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes_xyxy[:n]
    ]
    
    # Convert integer labels to string labels
    labels_str = [coco_classes[i] for i in labels_int[:n]]

    return TorchDetResult(
        boxes=boxes,
        scores=[float(s) for s in scores[:n]],
        labels=labels_str,
    )
