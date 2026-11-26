"""Utilities for cropping and saving non-chart image regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_utils import crop_from_bounds, padded_bounds


@dataclass(frozen=True)
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class NonChartCropper:
    """Save individual non-chart regions while preserving their original resolution."""

    def __init__(self, output_directory: str | Path | None = None, padding: float = 0.0) -> None:
        self.output_directory = Path(output_directory) if output_directory else None
        self.padding = float(padding)

    def crop_bounds(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int] | CropBox,
        *,
        padding: float | None = None,
    ) -> tuple[int, int, int, int]:
        if isinstance(box, CropBox):
            x1, y1, x2, y2 = box.as_tuple()
        else:
            if len(box) != 4:
                raise ValueError("Bounding box must contain x1, y1, x2, y2")
            x1, y1, x2, y2 = [int(value) for value in box]
        padding = self.padding if padding is None else padding
        bounds = padded_bounds(image.shape, (x1, y1, x2, y2), padding)
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(f"Invalid crop bounds: {bounds}")
        return bounds

    def crop(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int] | CropBox,
        *,
        padding: float | None = None,
        maintain_aspect_ratio: bool = True,
    ) -> np.ndarray:
        image = self._as_image_array(image)
        bounds = self.crop_bounds(image, box, padding=padding)
        return crop_from_bounds(image, bounds)

    @staticmethod
    def _as_image_array(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image must not be None")
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
            if image.ndim == 0:
                raise TypeError("image must be a NumPy array or array-like object")
        return image

    def _save_crop(self, crop: np.ndarray, filename: str) -> Path:
        directory = self.output_directory or Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        if not cv2.imwrite(str(path), crop):
            raise OSError(
                f"Could not save crop to {path}. "
                f"Check disk space and write permissions. "
                f"Crop shape: {crop.shape}, dtype: {crop.dtype}"
            )
        return path

    def save(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int] | CropBox,
        *,
        filename: str,
        padding: float | None = None,
    ) -> Path:
        crop = self.crop(image, box, padding=padding)
        return self._save_crop(crop, filename)

    def crop_and_save(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int] | CropBox,
        *,
        filename: str,
        padding: float | None = None,
    ) -> tuple[np.ndarray, Path]:
        crop = self.crop(image, box, padding=padding)
        path = self._save_crop(crop, filename)
        return crop, path

    def crop_and_save_with_bounds(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int] | CropBox,
        *,
        filename: str,
        padding: float | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int], Path]:
        image = self._as_image_array(image)
        crop = self.crop(image, box, padding=padding)
        bounds = self.crop_bounds(image, box, padding=padding)
        path = self._save_crop(crop, filename)
        return crop, bounds, path


def save_non_chart_crop(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    output_directory: str | Path,
    filename: str,
    *,
    padding: float = 0.0,
) -> Path:
    cropper = NonChartCropper(output_directory=output_directory, padding=padding)
    return cropper.save(image, box, filename=filename, padding=padding)


def crop_region(
    image: np.ndarray,
    box: tuple[int, int, int, int] | CropBox,
    *,
    padding: float = 0.0,
) -> np.ndarray:
    return NonChartCropper(padding=padding).crop(image, box, padding=padding)
