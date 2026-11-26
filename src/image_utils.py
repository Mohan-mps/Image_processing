"""Shared image-coordinate and crop operations."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def image_metadata_dpi(image_path: str | Path) -> float | None:
    """Return one valid image DPI value, or None when metadata is unavailable."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Reading image DPI metadata requires Pillow; install project "
            "dependencies from requirements.txt."
        ) from error

    try:
        with Image.open(image_path) as source:
            metadata = source.info.get("dpi")
    except OSError as error:
        LOGGER.warning("Could not read DPI metadata from %s: %s", image_path, error)
        return None
    if metadata is None:
        return None

    values = metadata if isinstance(metadata, (tuple, list)) else (metadata,)
    try:
        dpi_values = [float(value) for value in values]
    except (TypeError, ValueError):
        LOGGER.warning("Ignoring invalid DPI metadata in %s: %r", image_path, metadata)
        return None
    if (
        not dpi_values
        or any(not math.isfinite(value) or value <= 0 for value in dpi_values)
        or any(
            not math.isclose(value, dpi_values[0], rel_tol=1e-3)
            for value in dpi_values[1:]
        )
    ):
        LOGGER.warning(
            "Ignoring invalid or non-uniform DPI metadata in %s: %r",
            image_path,
            metadata,
        )
        return None
    return dpi_values[0]


def padded_bounds(
    image_shape: tuple[int, ...],
    box: tuple[int, int, int, int],
    padding_fraction: float,
    *,
    left_padding_fraction: float | None = None,
) -> tuple[int, int, int, int]:
    """Pad an XYXY box proportionally and clamp it to image dimensions.
    
    Args:
        image_shape: Image dimensions (height, width, ...)
        box: Bounding box as (x1, y1, x2, y2)
        padding_fraction: Symmetric padding fraction for all sides
        left_padding_fraction: Optional extra padding fraction for left side only.
            Useful for preserving y-axis labels that extend left of chart borders.
            If None, uses padding_fraction for all sides.
    
    Returns:
        Padded bounding box (x1, y1, x2, y2) clamped to image dimensions
    """
    if padding_fraction < 0:
        raise ValueError("Crop padding cannot be negative")
    if left_padding_fraction is not None and left_padding_fraction < 0:
        raise ValueError("Left padding cannot be negative")
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    padding_x = round(box_width * padding_fraction)
    padding_y = round(box_height * padding_fraction)
    # Use extra left padding if specified to capture y-axis labels
    left_pad = round(box_width * left_padding_fraction) if left_padding_fraction is not None else padding_x
    return (
        max(0, min(width, x1 - left_pad)),
        max(0, min(height, y1 - padding_y)),
        max(0, min(width, x2 + padding_x)),
        max(0, min(height, y2 + padding_y)),
    )


def crop_from_bounds(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Return a resolution-preserving image view for XYXY bounds."""
    left, top, right, bottom = bounds
    return image[top:bottom, left:right]


def map_box_to_original(
    box: tuple[int, int, int, int],
    image_to_original: Sequence[Sequence[float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ),
) -> tuple[int, int, int, int]:
    """Map an XYWH pixel-edge box through a 2x3 affine transform.

    The transform maps image-space pixel-edge coordinates to original-image
    pixel-edge coordinates. The returned XYWH box encloses all transformed
    corners, including when rotation or shear makes the result non-axis-aligned.
    """
    x, y, width, height = box
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in box
    ):
        raise ValueError("Box coordinates and dimensions must be integers")
    x, y, width, height = (int(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError("Box width and height must be positive")

    matrix = np.asarray(image_to_original, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("image_to_original must be a finite 2x3 matrix")
    
    # Validate that the matrix represents a valid affine transformation
    # Check if the transformation is degenerate (determinant near zero)
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(determinant) < 1e-10:
        raise ValueError(
            "Transform matrix is degenerate (determinant near zero). "
            "This would produce invalid coordinate mappings."
        )

    corners = np.array(
        [
            [x, y, 1.0],
            [x + width, y, 1.0],
            [x, y + height, 1.0],
            [x + width, y + height, 1.0],
        ],
        dtype=np.float64,
    )
    transformed = corners @ matrix.T
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)

    def snap_near_integer(value: float) -> float:
        nearest = round(value)
        return float(nearest) if math.isclose(value, nearest, abs_tol=1e-9) else value

    left, top = (math.floor(snap_near_integer(value)) for value in minimum)
    right, bottom = (math.ceil(snap_near_integer(value)) for value in maximum)
    if right <= left or bottom <= top:
        raise ValueError("Transform produced an empty box")
    return left, top, right - left, bottom - top
