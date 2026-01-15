from __future__ import annotations

from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .boxes import Box


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Box],
    scores: Optional[Sequence[float]] = None,
    *,
    color: str = "red",
    width: int = 4,
) -> Image.Image:
    """Return a copy of the image with boxes (and optional scores) drawn."""
    if scores is not None and len(scores) != len(boxes):
        raise ValueError("scores must match the number of boxes")

    out = image.copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, b in enumerate(boxes):
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=width)
        if scores is not None:
            draw.text((b.x1 + 6.0, b.y1 + 6.0), f"{scores[idx]:.2f}", fill=color, font=font)

    return out
