"""Image preprocessing for scientific figure regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessedImage:
    original: np.ndarray
    processed: np.ndarray
    transformations: tuple[str, ...]
    upscaled: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "transforms": list(self.transformations),
            "upscaled": self.upscaled,
        }


class ImagePreprocessor:
    """Apply conservative preprocessing without discarding visual evidence."""

    def __init__(
        self,
        *,
        contrast_clip_limit: float = 2.0,
        blur_kernel: int = 3,
        upscale_factor: int = 2,
        denoise_strength: int = 10,
    ) -> None:
        if contrast_clip_limit <= 0:
            raise ValueError("contrast_clip_limit must be positive")
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if upscale_factor < 1:
            raise ValueError("upscale_factor must be >= 1")
        self.contrast_clip_limit = contrast_clip_limit
        self.blur_kernel = blur_kernel
        self.upscale_factor = upscale_factor
        self.denoise_strength = denoise_strength

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()

    def preprocess(
        self,
        image: np.ndarray,
        *,
        contrast: bool = True,
        denoise: bool = True,
        upscale: bool = False,
    ) -> PreprocessedImage:
        original = image.copy()
        processed = self._ensure_bgr(original)
        transforms: list[str] = []

        if contrast:
            if processed.ndim == 3 and processed.shape[-1] == 3:
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=self.contrast_clip_limit, tileGridSize=(8, 8))
                l = clahe.apply(l)
                processed = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=self.contrast_clip_limit, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
            transforms.append("contrast")

        if denoise:
            if processed.ndim == 3:
                processed = cv2.fastNlMeansDenoisingColored(processed, None, self.denoise_strength, self.denoise_strength, 7, 21)
            else:
                processed = cv2.fastNlMeansDenoising(processed, None, self.denoise_strength, 7, 21)
            transforms.append("denoise")

        upscaled = False
        if upscale and max(processed.shape[:2]) < 4096:
            processed = cv2.resize(
                processed,
                (
                    max(1, processed.shape[1] * self.upscale_factor),
                    max(1, processed.shape[0] * self.upscale_factor),
                ),
                interpolation=cv2.INTER_CUBIC,
            )
            upscaled = True
            transforms.append("upscale")

        return PreprocessedImage(
            original=original,
            processed=processed,
            transformations=tuple(transforms),
            upscaled=upscaled,
        )

    def preprocess_file(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path | None = None,
        **kwargs: Any,
    ) -> PreprocessedImage:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        result = self.preprocess(image, **kwargs)
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), result.processed):
                raise OSError(f"Could not save processed image: {path}")
        return result

    @staticmethod
    def save_visualization(original: np.ndarray, processed: np.ndarray, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas = np.hstack([original, processed]) if original.shape[:2] == processed.shape[:2] else np.concatenate([original, processed], axis=1)
        if not cv2.imwrite(str(path), canvas):
            raise OSError(f"Could not save preprocessing visualization: {path}")
        return path
