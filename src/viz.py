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
    width: int = 10,
    font_size: int = 75,
) -> Image.Image:
    """
    Draws bounding boxes, scores, and labels on a copy of an image.

    This function provides a flexible way to visualize detection results. It draws
    bounding boxes and can optionally display class labels and confidence scores.
    Text is rendered with a contrasting background for improved legibility.

    Note: The function attempts to load a specific Liberation Sans font. If not
    found, it falls back to a default font, which may affect text appearance.

    :param image: The base image (PIL.Image.Image) to draw on.
    :type image: PIL.Image.Image
    :param boxes: A sequence of Box objects to draw.
    :type boxes: Sequence[Box]
    :param scores: An optional sequence of confidence scores for each box.
    :type scores: Optional[Sequence[float]]
    :param labels: An optional sequence of string labels for each box.
    :type labels: Optional[Sequence[str]]
    :param color: The color to use for the bounding box and text background.
                  Defaults to "red".
    :type color: str, optional
    :param width: The line width for the bounding box. Defaults to 10.
    :type width: int, optional
    :param font_size: The desired font size for labels and scores. Defaults to 75.
    :type font_size: int, optional
    :return: A new PIL.Image.Image with the annotations drawn.
    :rtype: PIL.Image.Image
    :raises ValueError: If the length of `scores` or `labels` does not match
                        the length of `boxes`.
    """
    if scores is not None and len(scores) != len(boxes):
        raise ValueError("scores must match the number of boxes")
    if labels is not None and len(labels) != len(boxes):
        raise ValueError("labels must match the number of boxes")

    out = image.copy()
    draw = ImageDraw.Draw(out)

    # Load a reasonably-sized font, falling back to the default if not found.
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
            try:  # Modern method
                text_bbox = draw.textbbox((0, 0), display_str, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:  # Older method
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
