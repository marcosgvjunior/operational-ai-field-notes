from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .boxes import Box


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Box],
    scores: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    *,
    color: str = "red",
    width: int = 4,
    font_size: int = 25, # Increased default font size
) -> Image.Image:
    """Return a copy of the image with boxes, scores, and labels drawn legibly."""
    if scores is not None and len(scores) != len(boxes):
        raise ValueError("scores must match the number of boxes")
    if labels is not None and len(labels) != len(boxes):
        raise ValueError("labels must match the number of boxes")

    out = image.copy()
    draw = ImageDraw.Draw(out)

    # Load a reasonably-sized font, falling back to the default if not found.
    # This is the robust approach from the user's reference code.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", font_size)
    except IOError:
        print("Font not found, using default font. Text may be small.")
        font = ImageFont.load_default()

    for idx, b in enumerate(boxes):
        # Draw the bounding box itself
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=width)

        # Prepare the text to display
        display_str = ""
        has_label = labels is not None and labels[idx]
        has_score = scores is not None

        if has_label and has_score:
            display_str = f"{labels[idx]}: {scores[idx]:.0%}"
        elif has_label:
            display_str = labels[idx]
        elif has_score:
            display_str = f"{scores[idx]:.0%}"

        # If there is text, draw it with a background for readability
        if display_str:
            # Calculate text size
            try: # Modern method
                text_bbox = draw.textbbox((0, 0), display_str, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError: # Older method
                text_width, text_height = draw.textsize(display_str, font=font)

            margin = int(np.ceil(0.05 * text_height))

            # Position the text box: above the bounding box if there's space, else just inside.
            if b.y1 > text_height + 2 * margin:
                text_y_start = b.y1 - text_height - (2 * margin)
            else:
                text_y_start = b.y1
            
            text_x_start = b.x1

            # Draw a background rectangle for the text
            rect_coords = [
                (text_x_start, text_y_start),
                (text_x_start + text_width + 2 * margin, text_y_start + text_height + 2 * margin),
            ]
            draw.rectangle(rect_coords, fill=color)

            # Draw the text on top of the rectangle
            draw.text((text_x_start + margin, text_y_start + margin), display_str, fill="white", font=font)

    return out
