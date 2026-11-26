"""Resolution-preserving chart image cropping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .chart_detector import ChartBox
from .image_utils import crop_from_bounds, padded_bounds


def padded_box(
    image: np.ndarray,
    box: ChartBox,
    padding_fraction: float = 0.12,
    *,
    left_padding_fraction: float = 0.25,
) -> tuple[int, int, int, int]:
    """Expand a detection to preserve nearby axis labels, clamped to the image.
    
    Args:
        image: Source image
        box: Chart detection bounding box
        padding_fraction: Symmetric padding fraction for top, right, bottom (default: 0.12)
        left_padding_fraction: Extra padding fraction for left side to capture y-axis
            scale multipliers and labels that often extend left of the plot area (default: 0.25)
    
    Returns:
        Padded bounding box (x1, y1, x2, y2) clamped to image dimensions
    """
    return padded_bounds(
        image.shape,
        (box.x1, box.y1, box.x2, box.y2),
        padding_fraction,
        left_padding_fraction=left_padding_fraction,
    )


def crop_chart(image: np.ndarray, box: ChartBox) -> np.ndarray:
    """Crop an already-clamped chart rectangle without resizing it."""
    crop = crop_from_bounds(image, (box.x1, box.y1, box.x2, box.y2))
    if crop.size == 0:
        raise ValueError(f"Chart box produced an empty crop: {box.as_list()}")
    return crop


def save_chart_crop(
    image: np.ndarray,
    box: ChartBox,
    source_path: Path,
    chart_id: int,
    output_directory: Path,
    padding_fraction: float = 0.12,
    *,
    crop: np.ndarray | None = None,
) -> Path:
    """Save a chart crop using the source image stem and one-based chart ID."""
    output_directory.mkdir(parents=True, exist_ok=True)
    crop_path = output_directory / f"{source_path.stem}_chart_{chart_id:03d}.png"
    if crop is None:
        left, top, right, bottom = padded_box(image, box, padding_fraction)
        crop = crop_from_bounds(image, (left, top, right, bottom))
    
    # Validate crop before checking size
    if crop is None:
        raise ValueError(f"Crop is None for chart box: {box.as_list()}")
    if not isinstance(crop, np.ndarray):
        raise TypeError(f"Crop must be a numpy ndarray, got {type(crop)}")
    if crop.size == 0:
        raise ValueError(f"Chart box produced an empty crop: {box.as_list()}")
    if not cv2.imwrite(str(crop_path), crop):
        raise OSError(f"Could not save chart crop: {crop_path}")
    return crop_path
