"""OCR-backed text regions with OpenCV-refined pixel bounds.

This module uses mps-ocr's PaddleOCR wrapper located in mps-ocr/core/
for OCR text detection and recognition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import numpy as np

# Suppress PaddlePaddle verbose output
os.environ.setdefault("FLAGS_pir_apply_shape_optimization_pass", "0")
os.environ.setdefault("GLOG_minloglevel", "2")  # Suppress INFO and WARNING
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")

_NUMERIC_TEXT = re.compile(
    r"^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)"
    r"(?:[eE][+-]?\d+)?(?:[×x]10\^?[+-]?\d+)?$"
)
_SUPERSCRIPT_DIGITS = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁻−",
    "0123456789--",
)


@dataclass(frozen=True)
class TextRegion:
    text: str
    content_type: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    ocr_x: int = 0
    ocr_y: int = 0
    ocr_width: int = 0
    ocr_height: int = 0
    detection_confidence: float = 1.0
    preprocessing: str = "original"


class _PaddleOCRReader:
    """Run PaddleOCR detection first, then recognition on each detected crop via mps-ocr."""

    def __init__(self) -> None:
        if sys.platform == "win32":
            os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        
        # Suppress PaddleX verbosity
        os.environ.setdefault("PADDLEX_HOME", os.path.expanduser("~/.paddlex"))
        
        # Redirect stdout temporarily to suppress model download messages
        import contextlib
        import io
        
        # Create a context to capture and suppress PaddleX verbose output
        suppressed_output = io.StringIO()
        
        try:
            # Add mps-ocr core to path to use the local version
            mps_ocr_core = Path(__file__).parent.parent / "mps-ocr" / "core" / "src"
            if mps_ocr_core.exists():
                mps_ocr_core_str = str(mps_ocr_core)
                if mps_ocr_core_str not in sys.path:
                    sys.path.insert(0, mps_ocr_core_str)
            
            # Suppress output during import and initialization, but catch real errors
            with contextlib.redirect_stdout(suppressed_output):
                with contextlib.redirect_stderr(suppressed_output):
                    try:
                        # Now import through mps-ocr's PaddleOCR engine
                        from paddleocr import TextDetection, TextRecognition
                        
                        # Initialize models (this generates the messages)
                        self._detector = TextDetection()
                        self._recognizer = TextRecognition()
                    except Exception as init_error:
                        # If initialization fails, restore output and re-raise with context
                        sys.stdout.write(suppressed_output.getvalue())
                        raise ImportError(
                            f"Failed to initialize PaddleOCR models: {init_error}"
                        ) from init_error
                    
        except ImportError as error:
            raise ImportError(
                "Text measurement requires PaddleOCR 3 and PaddlePaddle. "
                "Install project dependencies from requirements.txt. "
                "Using mps-ocr from mps-ocr/core/"
            ) from error

    def detect(self, image: np.ndarray) -> list[tuple[np.ndarray, float]]:
        results = self._detector.predict(input=image, batch_size=1)
        detections: list[tuple[np.ndarray, float]] = []
        for result in _prediction_payloads(results):
            polygons = result.get("dt_polys")
            scores = result.get("dt_scores")
            if polygons is None or scores is None:
                raise ValueError(
                    "PaddleOCR detection output must contain dt_polys and dt_scores"
                )
            if len(polygons) != len(scores):
                raise ValueError("PaddleOCR detection box and score counts differ")
            for index, polygon in enumerate(polygons):
                confidence = float(scores[index])
                detections.append((np.asarray(polygon), confidence))
        return detections

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        height = crop.shape[0]
        if height < 20:
            crop = cv2.resize(
                crop,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC,
            )
        results = self._recognizer.predict(input=crop, batch_size=1)
        payloads = _prediction_payloads(results)
        if not payloads:
            raise ValueError("PaddleOCR recognition returned no result")
        result = payloads[0]
        if "rec_text" not in result or "rec_score" not in result:
            raise ValueError(
                "PaddleOCR recognition output must contain rec_text and rec_score"
            )
        text = result["rec_text"]
        score = result["rec_score"]
        if isinstance(text, (list, tuple, np.ndarray)):
            text = text[0] if len(text) else ""
        if isinstance(score, (list, tuple, np.ndarray)):
            score = score[0] if len(score) else 0.0
        if not isinstance(text, str):
            raise ValueError("PaddleOCR recognition output must contain string text")
        return text, float(score)


def _prediction_payloads(results: Any) -> list[dict[str, Any]]:
    if results is None:
        return []
    if isinstance(results, (dict,)) or hasattr(results, "json"):
        results = [results]
    payloads: list[dict[str, Any]] = []
    for result in results:
        payload = result if isinstance(result, dict) else getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict):
            nested = payload.get("res", payload)
            if isinstance(nested, dict):
                payloads.append(nested)
    return payloads


@lru_cache(maxsize=1)
def _default_reader() -> Any:
    try:
        return _PaddleOCRReader()
    except ImportError as error:
        raise ImportError(
            "Text measurement requires PaddleOCR. Install project dependencies "
            "from requirements.txt."
        ) from error


def _content_type(text: str) -> str:
    numeric = text.translate(_SUPERSCRIPT_DIGITS)
    if numeric and numeric[0] in "([{":
        numeric = numeric[1:]
    if numeric and numeric[-1] in ")]}":
        numeric = numeric[:-1]
    numeric = re.sub(r"\s+", "", numeric)
    if _NUMERIC_TEXT.fullmatch(numeric):
        return "number"
    has_letters = any(character.isalpha() for character in text)
    has_numbers = any(character.isdigit() for character in text)
    if has_letters and has_numbers:
        return "alphanumeric"
    if has_numbers:
        return "number"
    return "text"


def _refine_box(
    gray: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    crop = gray[top:bottom, left:right]
    if crop.size == 0:
        return bounds

    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
    background = float(np.median(border))
    background_difference = np.abs(
        crop.astype(np.float32) - background
    ).clip(0, 255).astype(np.uint8)
    otsu_threshold, foreground = cv2.threshold(
        background_difference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    foreground[background_difference < max(8, otsu_threshold)] = 0
    y_coordinates, x_coordinates = np.where(foreground != 0)
    if not len(x_coordinates):
        return bounds

    refined_left = left + int(x_coordinates.min())
    refined_top = top + int(y_coordinates.min())
    refined_right = left + int(x_coordinates.max()) + 1
    refined_bottom = top + int(y_coordinates.max()) + 1
    return refined_left, refined_top, refined_right, refined_bottom


def _intersection_over_smaller(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area else 0.0


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a uint8 grayscale image to boost local contrast.

    This improves OCR confidence on small, faint, or JPEG-compressed text
    (axis numbers, significance markers, subscripts) without altering the
    image dimensions or coordinate system.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _paddle_preprocessing_candidates(
    image: np.ndarray, gray: np.ndarray
) -> list[tuple[str, np.ndarray]]:
    """Generate preprocessing candidates, prioritizing faster variants.
    
    Optimized to reduce processing time:
    - Always try original and CLAHE (most effective)
    - Only add expensive variants (OTSU, sharpening, denoising) for low-quality images
    - Removed grayscale variant as it provides minimal benefit
    """
    def model_image(candidate: np.ndarray) -> np.ndarray:
        if candidate.ndim == 2:
            return cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)
        return candidate

    # Start with the most effective variants
    candidates = [
        ("original", model_image(image)),
        ("clahe", model_image(_enhance_contrast(gray))),
    ]
    
    # Only add expensive preprocessing for low-contrast images
    if float(gray.std()) < 55:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        candidates.append(("otsu", model_image(otsu)))
    
    # Skip adaptive threshold, sharpening, and denoising as they rarely improve results
    # and significantly slow down processing (removed 3-4 variants)
    
    return candidates


def _box_from_polygon(
    polygon: Any, image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    points = np.asarray(polygon, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("OCR boxes must contain four finite XY points")
    left = max(0, int(np.floor(points[:, 0].min())))
    top = max(0, int(np.floor(points[:, 1].min())))
    right = min(image_width, int(np.ceil(points[:, 0].max())))
    bottom = min(image_height, int(np.ceil(points[:, 1].max())))
    return None if left >= right or top >= bottom else (left, top, right, bottom)


def detect_text_regions(
    image: np.ndarray,
    *,
    reader: Any | None = None,
    minimum_confidence: float = 0.1,
    excluded_boxes: Sequence[tuple[int, int, int, int]] = (),
) -> list[TextRegion]:
    """Detect OCR text regions and tighten their boxes to visible foreground.

    The confidence threshold defaults to 0.1 so that small, faint, or
    JPEG-compressed labels (axis numbers, significance markers) are not
    silently dropped. PaddleOCR evaluates original, grayscale, and CLAHE
    variants, with thresholding, denoising, or sharpening added only for
    low-quality inputs. The original colour image is never modified.
    """
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Text detection source must be a non-empty image")
    if not math.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Text detection requires grayscale, BGR, or BGRA input")
    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.integer):
            maximum = np.iinfo(gray.dtype).max
            gray = np.rint(
                gray.astype(np.float64) * (255.0 / maximum)
            ).clip(0, 255).astype(np.uint8)
        elif np.issubdtype(gray.dtype, np.floating) and np.isfinite(gray).all():
            minimum, maximum = float(gray.min()), float(gray.max())
            if minimum < 0 or maximum > 1:
                raise ValueError("Floating-point images must have values in [0, 1]")
            gray = np.rint(gray * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported or non-finite image dtype: {gray.dtype}")

    active_reader = reader if reader is not None else _default_reader()
    image_height, image_width = gray.shape
    exclusions: list[tuple[int, int, int, int]] = []
    for index, box in enumerate(excluded_boxes, start=1):
        if len(box) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in box
        ):
            raise ValueError(f"Excluded box {index} must contain four integers")
        excluded_left, excluded_top, excluded_right, excluded_bottom = (
            int(value) for value in box
        )
        if not (
            0 <= excluded_left < excluded_right <= image_width
            and 0 <= excluded_top < excluded_bottom <= image_height
        ):
            raise ValueError(
                f"Excluded box {index} is outside the image: {box} "
                f"for {image_width}x{image_height}"
            )
        exclusions.append(
            (excluded_left, excluded_top, excluded_right, excluded_bottom)
        )

    regions: list[TextRegion] = []
    if callable(getattr(active_reader, "detect", None)) and callable(
        getattr(active_reader, "recognize", None)
    ):
        if image.dtype == np.uint8:
            image_for_ocr = image
        elif np.issubdtype(image.dtype, np.integer):
            image_maximum = np.iinfo(image.dtype).max
            image_for_ocr = np.rint(
                image.astype(np.float64) * (255.0 / image_maximum)
            ).clip(0, 255).astype(np.uint8)
        else:
            image_for_ocr = np.rint(image * 255).astype(np.uint8)
        if image_for_ocr.ndim == 3 and image_for_ocr.shape[2] == 4:
            image_for_ocr = cv2.cvtColor(image_for_ocr, cv2.COLOR_BGRA2BGR)
        elif image_for_ocr.ndim == 2:
            image_for_ocr = gray
        candidates = _paddle_preprocessing_candidates(image_for_ocr, gray)
        # Track detected regions by position to enable early exit
        detected_positions: set[tuple[int, int, int, int]] = set()
        high_confidence_count = 0
        total_detections = 0
        
        for preprocessing_idx, (preprocessing, candidate) in enumerate(candidates):
            preprocessing_detections = 0
            for polygon, detection_confidence in active_reader.detect(candidate):
                ocr_bounds = _box_from_polygon(
                    polygon, image_width, image_height
                )
                if ocr_bounds is None:
                    continue
                
                # Skip if we already detected text at this location with high confidence
                if ocr_bounds in detected_positions:
                    continue
                    
                left, top, right, bottom = ocr_bounds
                if any(
                    left < excluded_right
                    and right > excluded_left
                    and top < excluded_bottom
                    and bottom > excluded_top
                    for excluded_left, excluded_top, excluded_right, excluded_bottom in exclusions
                ):
                    continue
                text, confidence = active_reader.recognize(
                    candidate[top:bottom, left:right]
                )
                confidence_value = float(confidence)
                detection_confidence_value = float(detection_confidence)
                if not math.isfinite(confidence_value) or not math.isfinite(
                    detection_confidence_value
                ):
                    raise ValueError("OCR confidence must be finite")
                if (
                    not isinstance(text, str)
                    or not text.strip()
                    or confidence_value < minimum_confidence
                ):
                    continue
                refined_left, refined_top, refined_right, refined_bottom = _refine_box(
                    gray, ocr_bounds
                )
                value = text.strip()
                regions.append(
                    TextRegion(
                        text=value,
                        content_type=_content_type(value),
                        x=refined_left,
                        y=refined_top,
                        width=refined_right - refined_left,
                        height=refined_bottom - refined_top,
                        confidence=confidence_value,
                        ocr_x=left,
                        ocr_y=top,
                        ocr_width=right - left,
                        ocr_height=bottom - top,
                        detection_confidence=detection_confidence_value,
                        preprocessing=preprocessing,
                    )
                )
                preprocessing_detections += 1
                total_detections += 1
                detected_positions.add(ocr_bounds)
                
                # Track high confidence detections
                if confidence_value > 0.8:
                    high_confidence_count += 1
            
            # Early exit optimization: Only exit if we have good results AND
            # we've tried at least the first two preprocessing variants (original + CLAHE)
            # This prevents skipping better detections while still saving time on clear images
            if preprocessing_idx >= 1 and total_detections >= 5 and high_confidence_count >= 4:
                break
    else:
        detections = active_reader.readtext(
            _enhance_contrast(gray),
            detail=1,
            paragraph=False,
            batch_size=1,
            rotation_info=[90, 270],
            width_ths=0.9,
        )
        for detection in detections:
            if not isinstance(detection, (tuple, list)) or len(detection) < 3:
                raise ValueError("OCR result must contain a box, text, and confidence")
            polygon, text, confidence = detection[:3]
            if not isinstance(text, str) or not text.strip():
                continue
            confidence_value = float(confidence)
            if not math.isfinite(confidence_value):
                raise ValueError("OCR confidence must be finite")
            if confidence_value < minimum_confidence:
                continue
            bounds = _box_from_polygon(polygon, image_width, image_height)
            if bounds is None:
                continue
            left, top, right, bottom = bounds
            if any(
                left < excluded_right
                and right > excluded_left
                and top < excluded_bottom
                and bottom > excluded_top
                for excluded_left, excluded_top, excluded_right, excluded_bottom in exclusions
            ):
                continue
            refined_left, refined_top, refined_right, refined_bottom = _refine_box(
                gray, bounds
            )
            value = text.strip()
            regions.append(
                TextRegion(
                    text=value,
                    content_type=_content_type(value),
                    x=refined_left,
                    y=refined_top,
                    width=refined_right - refined_left,
                    height=refined_bottom - refined_top,
                    confidence=confidence_value,
                    ocr_x=left,
                    ocr_y=top,
                    ocr_width=right - left,
                    ocr_height=bottom - top,
                    detection_confidence=confidence_value,
                    preprocessing="clahe",
                )
            )

    regions.sort(key=lambda region: (region.y, region.x, region.height, region.width))
    unique_regions: list[TextRegion] = []
    for region in regions:
        region_bounds = (
            region.x,
            region.y,
            region.x + region.width,
            region.y + region.height,
        )
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(unique_regions)
                if (
                    (overlap := _intersection_over_smaller(
                        region_bounds,
                        (
                            existing.x,
                            existing.y,
                            existing.x + existing.width,
                            existing.y + existing.height,
                        ),
                    ))
                    >= 0.95
                    or (
                        re.sub(r"\s+", " ", existing.text).casefold()
                        == re.sub(r"\s+", " ", region.text).casefold()
                        and overlap >= 0.8
                    )
                )
            ),
            None,
        )
        if duplicate_index is None:
            unique_regions.append(region)
        elif region.confidence > unique_regions[duplicate_index].confidence:
            unique_regions[duplicate_index] = region
    return sorted(unique_regions, key=lambda region: (region.y, region.x))
