"""Original-image OCR text measurements and low-level foreground analysis.

The standalone command and pipeline use OCR regions; ``measure_image`` remains
a non-semantic threshold-segmentation helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .image_utils import map_box_to_original
from .text_detection import detect_text_regions

# Global OCR cache with bounded size: key = (image_hash, origin, excluded_boxes_hash), value = OCR results
# Using collections.OrderedDict for LRU-like behavior
from collections import OrderedDict
_OCR_CACHE: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_OCR_CACHE_MAX_SIZE = 100  # Maximum number of cached OCR results

ThresholdMethod = Literal["otsu", "adaptive"]
SegmentationMethod = Literal["components", "contours"]
Polarity = Literal["auto", "dark", "light"]
Morphology = Literal["none", "open", "close", "open_close"]
Box = tuple[int, int, int, int]
ObjectType = Literal["connected_component", "contour", "provided_region"]
POINTS_PER_INCH = 72.0
DEFAULT_DPI = 96.0
HEIGHT_ADJUSTMENT_PX = 2


def _validate_dpi(dpi: float) -> float:
    if isinstance(dpi, bool):
        raise ValueError("dpi must be a finite positive number")
    try:
        numeric_dpi = float(dpi)
    except (TypeError, ValueError) as error:
        raise ValueError("dpi must be a finite positive number") from error
    if not np.isfinite(numeric_dpi) or numeric_dpi <= 0:
        raise ValueError("dpi must be a finite positive number")
    return numeric_dpi


def _points(pixels: int, dpi: float) -> float:
    return round(pixels * POINTS_PER_INCH / dpi, 4)


@dataclass(frozen=True)
class MeasurementConfig:
    """Controls foreground segmentation without changing source resolution."""

    threshold_method: ThresholdMethod = "otsu"
    segmentation_method: SegmentationMethod = "components"
    polarity: Polarity = "auto"
    morphology: Morphology = "none"
    morphology_kernel_size: int = 3
    adaptive_block_size: int = 31
    adaptive_constant: float = 5.0
    connectivity: Literal[4, 8] = 8
    minimum_area: int = 1
    maximum_area: int | None = None
    dpi: float = DEFAULT_DPI

    def __post_init__(self) -> None:
        if self.threshold_method not in ("otsu", "adaptive"):
            raise ValueError(f"Unsupported threshold method: {self.threshold_method}")
        if self.segmentation_method not in ("components", "contours"):
            raise ValueError(
                f"Unsupported segmentation method: {self.segmentation_method}"
            )
        if self.polarity not in ("auto", "dark", "light"):
            raise ValueError(f"Unsupported foreground polarity: {self.polarity}")
        if self.morphology not in ("none", "open", "close", "open_close"):
            raise ValueError(f"Unsupported morphology operation: {self.morphology}")
        if self.morphology_kernel_size < 1 or self.morphology_kernel_size % 2 == 0:
            raise ValueError("morphology_kernel_size must be a positive odd integer")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be an odd integer of at least 3")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if self.minimum_area < 1:
            raise ValueError("minimum_area must be at least 1")
        if self.maximum_area is not None and self.maximum_area < self.minimum_area:
            raise ValueError("maximum_area must be >= minimum_area")
        _validate_dpi(self.dpi)


@dataclass(frozen=True)
class ObjectMeasurement:
    """One measured object's enclosing box in original-image pixels."""

    object_id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    object_type: str
    foreground_area: int | None = None
    original_height_px: int | None = None
    height_adjustment_px: int = 0
    text: str | None = None
    content_type: str | None = None
    confidence: float | None = None
    ocr_bbox: dict[str, int] | None = None
    refined_bbox: dict[str, int] | None = None
    detection_confidence: float | None = None
    preprocessing: str | None = None

    def as_dict(self, dpi: float = DEFAULT_DPI) -> dict[str, Any]:
        scale = _validate_dpi(dpi)
        value: dict[str, Any] = {
            "object_id": self.object_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "width_px": self.width,
            "height": self.height,
            "adjusted_height_px": self.height,
            "original_height_px": self.original_height_px,
            "height_adjustment_px": self.height_adjustment_px,
            "area": self.area,
            "area_type": "bounding_box",
            "bounding_box": {
                "x_min": self.x,
                "y_min": self.y,
                "x_max": self.x + self.width,
                "y_max": self.y + self.height,
            },
            "original_bounding_box": {
                "x_min": self.x,
                "y_min": self.y,
                "x_max": self.x + self.width,
                "y_max": self.y + (self.original_height_px or self.height),
            },
            "coordinate_system": "original_image_pixels",
            "object_type": self.object_type,
            "x_points": _points(self.x, scale),
            "y_points": _points(self.y, scale),
            "width_points": _points(self.width, scale),
            "height_points": _points(self.height, scale),
            "dpi": scale,
            "area_points_squared": round(
                _points(self.width, scale) * _points(self.height, scale), 4
            ),
        }
        if self.foreground_area is not None:
            value["foreground_area"] = self.foreground_area
        if self.text is not None:
            value["text"] = self.text
        if self.content_type is not None:
            value["content_type"] = self.content_type
        if self.confidence is not None:
            value["text_detection_confidence"] = round(self.confidence, 4)
        if self.ocr_bbox is not None:
            value["ocr_bbox"] = self.ocr_bbox
        if self.refined_bbox is not None:
            value["refined_bbox"] = self.refined_bbox
        if self.detection_confidence is not None:
            value["detection_confidence"] = round(self.detection_confidence, 4)
        if self.preprocessing is not None:
            value["preprocessing"] = self.preprocessing
        return value


@dataclass(frozen=True)
class MeasurementResult:
    """Measurements and source-image metadata for a single image."""

    source_image: str
    original_width: int
    original_height: int
    measurements: tuple[ObjectMeasurement, ...]
    config: MeasurementConfig | None
    otsu_threshold: float | None
    image_channels: int | None = None
    image_format: str | None = None
    dpi: float = DEFAULT_DPI
    detection_method: str = "opencv_threshold_segmentation"

    def as_dict(self) -> dict[str, Any]:
        if self.config is None:
            method = {
                "threshold": (
                    "opencv_otsu_within_ocr_boxes"
                    if self.detection_method.startswith(("easyocr", "paddleocr"))
                    else None
                ),
                "segmentation": (
                    "ocr_regions"
                    if self.detection_method.startswith(("easyocr", "paddleocr"))
                    else "provided_regions"
                ),
                "polarity": None,
                "morphology": None,
                "otsu_threshold": None,
                "dpi": self.dpi,
                "points_per_inch": POINTS_PER_INCH,
                "height_adjustment_px": HEIGHT_ADJUSTMENT_PX,
            }
        else:
            method = {
                "threshold": self.config.threshold_method,
                "segmentation": self.config.segmentation_method,
                "polarity": self.config.polarity,
                "morphology": self.config.morphology,
                "otsu_threshold": self.otsu_threshold,
                "dpi": self.dpi,
                "points_per_inch": POINTS_PER_INCH,
                "height_adjustment_px": HEIGHT_ADJUSTMENT_PX,
            }
        method["detector"] = self.detection_method
        value: dict[str, Any] = {
            "source_image": self.source_image,
            "original_dimensions": {
                "width": self.original_width,
                "height": self.original_height,
                "channels": self.image_channels,
                "format": self.image_format,
            },
            "measurement_method": method,
            "measurement_units": {
                "coordinates_and_primary_dimensions": "original_image_pixels",
                "converted_dimensions": "typographic_points",
                "dpi": self.dpi,
                "points_per_inch": POINTS_PER_INCH,
                "height_adjustment_px": HEIGHT_ADJUSTMENT_PX,
            },
            "measurements": [item.as_dict(self.dpi) for item in self.measurements],
        }
        return value


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Image must be grayscale, BGR, or BGRA")

    if gray.dtype == np.uint8:
        return gray
    if gray.dtype == np.uint16:
        return np.rint(gray.astype(np.float64) * (255.0 / 65535.0)).astype(np.uint8)
    if np.issubdtype(gray.dtype, np.integer):
        maximum = np.iinfo(gray.dtype).max
        if maximum <= 0:
            raise ValueError(f"Unsupported image dtype: {gray.dtype}")
        return np.rint(gray.astype(np.float64) * (255.0 / maximum)).clip(
            0, 255
        ).astype(np.uint8)
    if np.issubdtype(gray.dtype, np.floating) and np.isfinite(gray).all():
        minimum, maximum = float(gray.min()), float(gray.max())
        if minimum < 0 or maximum > 1:
            raise ValueError("Floating-point images must have values in [0, 1]")
        return np.rint(gray * 255).astype(np.uint8)
    raise ValueError(f"Unsupported or non-finite image dtype: {gray.dtype}")


def _read_original_image(image_path: str | Path) -> tuple[Path, np.ndarray]:
    source = Path(image_path)
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not read image: {source}")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"Image is empty or has an invalid shape: {source}")
    return source, image


def _border_median(gray: np.ndarray) -> float:
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    return float(np.median(border))


def _foreground_mask(
    gray: np.ndarray, config: MeasurementConfig
) -> tuple[np.ndarray, float | None]:
    polarity = config.polarity
    if polarity == "auto":
        polarity = "dark" if _border_median(gray) >= 127.5 else "light"
    threshold_type = cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY
    otsu_threshold: float | None = None

    if config.threshold_method == "otsu":
        otsu_threshold, mask = cv2.threshold(
            gray, 0, 255, threshold_type | cv2.THRESH_OTSU
        )
    else:
        minimum_dimension = min(gray.shape)
        if minimum_dimension < 3:
            raise ValueError(
                "Adaptive thresholding requires an image at least 3x3 pixels"
            )
        block_size = min(
            config.adaptive_block_size,
            minimum_dimension if minimum_dimension % 2 else minimum_dimension - 1,
        )
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            block_size,
            config.adaptive_constant,
        )

    if config.morphology != "none":
        size = config.morphology_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        if config.morphology in ("open", "open_close"):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if config.morphology in ("close", "open_close"):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return np.ascontiguousarray(mask), otsu_threshold


def _component_boxes(
    mask: np.ndarray, config: MeasurementConfig
) -> list[tuple[tuple[int, int, int, int], int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=config.connectivity, ltype=cv2.CV_32S
    )
    boxes = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < config.minimum_area:
            continue
        if config.maximum_area is not None and area > config.maximum_area:
            continue
        boxes.append(((x, y, width, height), area))
    return boxes


def _contour_boxes(
    mask: np.ndarray, config: MeasurementConfig
) -> list[tuple[tuple[int, int, int, int], int | None]]:
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        foreground_area = cv2.countNonZero(mask[y : y + height, x : x + width])
        if foreground_area < config.minimum_area:
            continue
        if (
            config.maximum_area is not None
            and foreground_area > config.maximum_area
        ):
            continue
        boxes.append(((x, y, width, height), None))
    return boxes


def _validate_roi(roi: Box, width: int, height: int) -> Box:
    if len(roi) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in roi
    ):
        raise ValueError("ROI coordinates must be four integers")
    left, top, right, bottom = (int(value) for value in roi)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"ROI must be a non-empty in-image XYXY box, got {roi} for {width}x{height}"
        )
    return left, top, right, bottom


def measure_image(
    image_path: str | Path,
    *,
    config: MeasurementConfig | None = None,
    roi: Box | None = None,
) -> MeasurementResult:
    """Measure foreground regions from the image read directly from disk.

    ``roi`` uses half-open XYXY coordinates in the original image. It restricts
    processing without cropping, resizing, enhancing, or changing the source.
    """
    source, image = _read_original_image(image_path)
    original_height, original_width = image.shape[:2]
    settings = config or MeasurementConfig()
    bounds = (
        (0, 0, original_width, original_height)
        if roi is None
        else _validate_roi(roi, original_width, original_height)
    )
    left, top, right, bottom = bounds
    gray = _gray_uint8(image[top:bottom, left:right])
    mask, otsu_threshold = _foreground_mask(gray, settings)

    if settings.segmentation_method == "components":
        boxes = _component_boxes(mask, settings)
        object_type: Literal["connected_component", "contour"] = (
            "connected_component"
        )
    else:
        boxes = _contour_boxes(mask, settings)
        object_type = "contour"

    translation = ((1.0, 0.0, float(left)), (0.0, 1.0, float(top)))
    original_boxes = [
        (map_box_to_original(box, translation), foreground_area)
        for box, foreground_area in boxes
    ]
    original_boxes.sort(
        key=lambda item: (
            item[0][1],
            item[0][0],
            item[0][3],
            item[0][2],
        )
    )
    measurements = tuple(
        ObjectMeasurement(
            object_id=index,
            x=x,
            y=y,
            width=width,
            height=min(height + HEIGHT_ADJUSTMENT_PX, original_height - y),
            area=width * height,
            object_type=object_type,
            foreground_area=foreground_area,
            original_height_px=height,
            height_adjustment_px=min(HEIGHT_ADJUSTMENT_PX, original_height - y - height),
        )
        for index, ((x, y, width, height), foreground_area) in enumerate(
            original_boxes, start=1
        )
    )
    return MeasurementResult(
        source_image=source.name,
        original_width=original_width,
        original_height=original_height,
        measurements=measurements,
        config=settings,
        otsu_threshold=otsu_threshold,
        image_channels=1 if image.ndim == 2 else int(image.shape[2]),
        image_format=source.suffix.lstrip(".").lower() or None,
        dpi=settings.dpi,
    )


def measure_regions(
    image_path: str | Path,
    regions: Sequence[Box],
    *,
    image_to_original: Sequence[Sequence[float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ),
    dpi: float = DEFAULT_DPI,
) -> MeasurementResult:
    """Measure supplied XYWH regions, such as text detector boxes.

    Boxes may come from a crop, resized image, or rotated image when its
    image-to-original affine transform is supplied. This measures the provided
    regions; it does not detect or verify their semantic contents.
    """
    source, image = _read_original_image(image_path)
    dpi = _validate_dpi(dpi)
    original_height, original_width = image.shape[:2]
    original_boxes = []
    for index, box in enumerate(regions, start=1):
        if len(box) != 4:
            raise ValueError(f"Region {index} must be an XYWH box")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in box
        ):
            raise ValueError(f"Region {index} coordinates must be integers")
        mapped = map_box_to_original(box, image_to_original)
        x, y, width, height = mapped
        if x < 0 or y < 0 or x + width > original_width or y + height > original_height:
            raise ValueError(f"Region {index} maps outside the original image: {mapped}")
        original_boxes.append(mapped)
    original_boxes.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    measurements = tuple(
        ObjectMeasurement(
            object_id=index,
            x=x,
            y=y,
            width=width,
            height=min(height + HEIGHT_ADJUSTMENT_PX, original_height - y),
            area=width * height,
            object_type="provided_region",
            original_height_px=height,
            height_adjustment_px=min(HEIGHT_ADJUSTMENT_PX, original_height - y - height),
        )
        for index, (x, y, width, height) in enumerate(original_boxes, start=1)
    )
    return MeasurementResult(
        source_image=source.name,
        original_width=original_width,
        original_height=original_height,
        measurements=measurements,
        config=None,
        otsu_threshold=None,
        image_channels=1 if image.ndim == 2 else int(image.shape[2]),
        image_format=source.suffix.lstrip(".").lower() or None,
        dpi=dpi,
        detection_method="provided_regions",
    )


def _create_cache_key(
    image: np.ndarray,
    origin: tuple[int, int],
    excluded_boxes: Sequence[Box],
) -> str:
    """Create a cache key from image content and parameters.
    
    Optimized to use image shape and a checksum instead of sampling,
    which provides better cache hit rates while remaining fast.
    """
    h, w = image.shape[:2]
    
    # Use image shape, dtype, and a quick checksum of corner regions
    # This is faster than full hash but more reliable than sampling
    corners = [
        image[0:min(10, h), 0:min(10, w)],
        image[max(0, h-10):h, max(0, w-10):w]
    ]
    corner_bytes = b''.join(c.tobytes() for c in corners)
    image_sig = hashlib.sha256(corner_bytes).hexdigest()[:12]
    
    # Create deterministic hash of excluded boxes using SHA256
    boxes_str = ",".join(f"{b[0]},{b[1]},{b[2]},{b[3]}" for b in sorted(excluded_boxes))
    boxes_hash = hashlib.sha256(boxes_str.encode()).hexdigest()[:8] if boxes_str else "none"
    
    return f"{h}x{w}_{image_sig}_{origin[0]}_{origin[1]}_{boxes_hash}"


def clear_ocr_cache() -> None:
    """Clear the global OCR cache. Useful for testing or memory management."""
    global _OCR_CACHE
    _OCR_CACHE.clear()


def _add_to_cache(key: str, value: list[dict[str, Any]]) -> None:
    """Add to cache with LRU eviction when size limit is reached."""
    global _OCR_CACHE
    # Remove oldest entry if cache is full
    if len(_OCR_CACHE) >= _OCR_CACHE_MAX_SIZE:
        _OCR_CACHE.popitem(last=False)  # Remove oldest (FIFO/LRU)
    _OCR_CACHE[key] = value


def measure_text_regions(
    image: np.ndarray,
    *,
    dpi: float = DEFAULT_DPI,
    reader: Any | None = None,
    origin: tuple[int, int] = (0, 0),
    original_dimensions: tuple[int, int] | None = None,
    minimum_confidence: float = 0.1,
    excluded_boxes: Sequence[Box] = (),
) -> list[dict[str, Any]]:
    """Detect text outside excluded XYXY boxes and measure it in source pixels."""
    # Check cache first
    cache_key = _create_cache_key(image, origin, excluded_boxes)
    if cache_key in _OCR_CACHE:
        # Move to end to mark as recently used
        _OCR_CACHE.move_to_end(cache_key)
        return _OCR_CACHE[cache_key].copy()
    
    dpi = _validate_dpi(dpi)
    if len(origin) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in origin
    ):
        raise ValueError("Text-region origin must contain two integer coordinates")
    offset_x, offset_y = (int(value) for value in origin)
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Text measurement source must be a non-empty image")
    image_height, image_width = image.shape[:2]
    original_width, original_height = original_dimensions or (
        image_width + offset_x,
        image_height + offset_y,
    )
    validated_exclusions: list[Box] = []
    for index, box in enumerate(excluded_boxes, start=1):
        if len(box) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in box
        ):
            raise ValueError(f"Excluded box {index} must contain four integers")
        left, top, right, bottom = (int(value) for value in box)
        if not (
            0 <= left < right <= image_width
            and 0 <= top < bottom <= image_height
        ):
            raise ValueError(
                f"Excluded box {index} is outside the image: {box} "
                f"for {image_width}x{image_height}"
            )
        validated_exclusions.append((left, top, right, bottom))

    ocr_image = image
    if validated_exclusions:
        ocr_image = image.copy()
        if np.issubdtype(ocr_image.dtype, np.integer):
            fill_value = np.iinfo(ocr_image.dtype).max
        elif np.issubdtype(ocr_image.dtype, np.floating):
            fill_value = 1.0
        else:
            raise ValueError(f"Unsupported image dtype: {ocr_image.dtype}")
        for left, top, right, bottom in validated_exclusions:
            ocr_image[top:bottom, left:right] = fill_value

    regions = detect_text_regions(
        ocr_image,
        reader=reader,
        minimum_confidence=minimum_confidence,
        excluded_boxes=validated_exclusions,
    )
    measurements = []
    for region in regions:
        x = region.x + offset_x
        y = region.y + offset_y
        region_right = x + region.width
        region_bottom = y + region.height
        if any(
            x < right and region_right > left and y < bottom and region_bottom > top
            for left, top, right, bottom in validated_exclusions
        ):
            continue
        measurement = measurement_from_xyxy(
            len(measurements) + 1,
            "text_region",
            (x, y, region_right, region_bottom),
            original_width,
            original_height,
            dpi,
            text=region.text,
            content_type=region.content_type,
            confidence=region.confidence,
        )
        measurement["ocr_bbox"] = {
            "x": region.ocr_x + offset_x,
            "y": region.ocr_y + offset_y,
            "width": region.ocr_width,
            "height": region.ocr_height,
        }
        measurement["refined_bbox"] = {
            "x": x,
            "y": y,
            "width": region.width,
            "height": region.height,
        }
        measurement["detection_confidence"] = round(
            region.detection_confidence, 4
        )
        measurement["recognition_confidence"] = round(region.confidence, 4)
        measurement["preprocessing"] = region.preprocessing
        adjusted_bottom = measurement["bounding_box"]["y_max"]
        if any(
            x < right and region_right > left and y < bottom and adjusted_bottom > top
            for left, top, right, bottom in validated_exclusions
        ):
            continue
        measurements.append(measurement)
    
    # Cache the results before returning using LRU mechanism
    _add_to_cache(cache_key, measurements.copy())
    
    return measurements


def measure_text_image(
    image_path: str | Path,
    *,
    dpi: float = DEFAULT_DPI,
    roi: Box | None = None,
    reader: Any | None = None,
    minimum_confidence: float = 0.1,
) -> MeasurementResult:
    """Measure only OCR-detected text and numeric regions in an original image."""
    source, image = _read_original_image(image_path)
    dpi = _validate_dpi(dpi)
    original_height, original_width = image.shape[:2]
    bounds = (
        (0, 0, original_width, original_height)
        if roi is None
        else _validate_roi(roi, original_width, original_height)
    )
    left, top, right, bottom = bounds
    records = measure_text_regions(
        image[top:bottom, left:right],
        dpi=dpi,
        reader=reader,
        origin=(left, top),
        original_dimensions=(original_width, original_height),
        minimum_confidence=minimum_confidence,
    )
    measurements = tuple(
        ObjectMeasurement(
            object_id=record["object_id"],
            x=record["x"],
            y=record["y"],
            width=record["width"],
            height=record["height"],
            area=record["area"],
            object_type=record["object_type"],
            original_height_px=record["original_height_px"],
            height_adjustment_px=record["height_adjustment_px"],
            text=record["text"],
            content_type=record["content_type"],
            confidence=record["text_detection_confidence"],
            ocr_bbox=record["ocr_bbox"],
            refined_bbox=record["refined_bbox"],
            detection_confidence=record["detection_confidence"],
            preprocessing=record["preprocessing"],
        )
        for record in records
    )
    return MeasurementResult(
        source_image=source.name,
        original_width=original_width,
        original_height=original_height,
        measurements=measurements,
        config=None,
        otsu_threshold=None,
        image_channels=1 if image.ndim == 2 else int(image.shape[2]),
        image_format=source.suffix.lstrip(".").lower() or None,
        dpi=dpi,
        detection_method="paddleocr_two_stage_with_opencv_box_refinement",
    )


def measurement_from_xyxy(
    object_id: int,
    object_type: str,
    bounds: tuple[int, int, int, int],
    original_width: int,
    original_height: int,
    dpi: float = DEFAULT_DPI,
    *,
    text: str | None = None,
    content_type: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Create a semantic object measurement from an original-image XYXY box."""
    if not object_type.strip():
        raise ValueError("object_type cannot be empty")
    dpi = _validate_dpi(dpi)
    if len(bounds) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in bounds
    ):
        raise ValueError("Bounding-box coordinates must be four integers")
    x_min, y_min, x_max, y_max = (int(value) for value in bounds)
    if not (
        0 <= x_min < x_max <= original_width
        and 0 <= y_min < y_max <= original_height
    ):
        raise ValueError(
            f"Bounding box {bounds} is outside the original image "
            f"({original_width}x{original_height})"
        )
    width, original_height_px = x_max - x_min, y_max - y_min
    actual_adjustment = min(HEIGHT_ADJUSTMENT_PX, original_height - y_max)
    adjusted_height_px = original_height_px + actual_adjustment
    adjusted_y_max = y_max + actual_adjustment
    value: dict[str, Any] = {
        "object_id": object_id,
        "object_type": object_type,
        "x": x_min,
        "y": y_min,
        "width": width,
        "width_px": width,
        "height": adjusted_height_px,
        "adjusted_height_px": adjusted_height_px,
        "original_height_px": original_height_px,
        "height_adjustment_px": actual_adjustment,
        "area": width * adjusted_height_px,
        "area_type": "bounding_box",
        "bounding_box": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": adjusted_y_max,
        },
        "original_bounding_box": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "coordinate_system": "original_image_pixels",
        "x_points": _points(x_min, dpi),
        "y_points": _points(y_min, dpi),
        "width_points": _points(width, dpi),
        "height_points": _points(adjusted_height_px, dpi),
        "dpi": dpi,
        "area_points_squared": round(
            _points(width, dpi) * _points(adjusted_height_px, dpi),
            4,
        ),
    }
    if text is not None:
        value["text"] = text
    if content_type is not None:
        value["content_type"] = content_type
    if confidence is not None:
        value["text_detection_confidence"] = round(confidence, 4)
    return value


def save_measurement_visualization(
    image_path: str | Path,
    result: MeasurementResult,
    destination: str | Path,
) -> Path:
    """Save a side-by-side original and annotated copy; never edit the source."""
    source, original = _read_original_image(image_path)
    if (
        source.name != result.source_image
        or original.shape[1] != result.original_width
        or original.shape[0] != result.original_height
    ):
        raise ValueError("Visualization source does not match the measurement result")
    if original.ndim == 2:
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    elif original.shape[2] == 4:
        original_bgr = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
    elif original.shape[2] == 3:
        original_bgr = original.copy()
    else:
        raise ValueError("Visualization requires a grayscale, BGR, or BGRA image")

    annotated = original_bgr.copy()
    _font = cv2.FONT_HERSHEY_SIMPLEX
    _font_scale = 0.32
    _font_thickness = 1
    for item in result.measurements:
        start = (item.x, item.y)
        end = (item.x + item.width - 1, item.y + item.height - 1)
        color = {
            "number": (255, 0, 255),
            "alphanumeric": (0, 165, 255),
            "text": (0, 200, 0),
        }.get(item.content_type, (0, 0, 255))
        cv2.rectangle(annotated, start, end, color, 1)

        label = f"{item.object_id}: {item.width}x{item.height}px"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _font, _font_scale, _font_thickness
        )
        img_h, img_w = annotated.shape[:2]

        label_y_above = item.y - 2
        if label_y_above - text_h >= 0:
            label_origin = (item.x, label_y_above)
            bg_top = label_y_above - text_h - baseline
            bg_bottom = label_y_above + baseline
        else:
            label_origin = (item.x, item.y + text_h + 1)
            bg_top = item.y + 1
            bg_bottom = item.y + text_h + baseline + 2

        label_x = min(label_origin[0], max(0, img_w - text_w - 1))
        label_origin = (label_x, label_origin[1])
        bg_right = min(label_x + text_w, img_w - 1)
        bg_top = max(0, bg_top)
        bg_bottom = min(bg_bottom, img_h - 1)

        if bg_bottom > bg_top and bg_right > label_x:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (label_x, bg_top), (bg_right, bg_bottom), (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        cv2.putText(
            annotated,
            label,
            label_origin,
            _font,
            _font_scale,
            color,
            _font_thickness,
            cv2.LINE_AA,
        )

    canvas = np.concatenate((original_bgr, annotated), axis=1)
    output = Path(destination)
    if output.resolve() == source.resolve():
        raise ValueError("Visualization output must not overwrite the source image")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Could not save measurement visualization: {output}")
    return output


def save_semantic_measurement_visualization(
    image: np.ndarray,
    measurements: Sequence[dict[str, Any]],
    destination: str | Path,
    *,
    source_path: str | Path | None = None,
    include_object_ids: bool = False,
) -> Path:
    """Overlay original-coordinate semantic boxes without resizing the image."""
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Visualization source must be a non-empty image")
    if image.ndim == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        annotated = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] == 3:
        annotated = image.copy()
    else:
        raise ValueError("Visualization requires a grayscale, BGR, or BGRA image")

    colors = {
        "text": (0, 200, 0),
        "alphanumeric": (0, 200, 0),
        "number": (255, 0, 255),
    }
    _font = cv2.FONT_HERSHEY_SIMPLEX
    _font_thickness = 1
    image_height, image_width = annotated.shape[:2]
    for measurement in measurements:
        x, y = measurement["x"], measurement["y"]
        width, height = measurement["width"], measurement["height"]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y, width, height))
            or width < 1
            or height < 1
            or x < 0
            or y < 0
            or x + width > image_width
            or y + height > image_height
        ):
            raise ValueError(f"Invalid visualization measurement: {measurement}")
        content_type = measurement.get("content_type")
        color = colors.get(content_type, (80, 210, 80))
        cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), color, 1)

        object_id = (
            f"{measurement.get('object_id')}: " if include_object_ids else ""
        )
        label_height = measurement.get("label_height", height)
        label = (
            f"{object_id}{measurement.get('text', '')} "
            f"{width}x{label_height}px"
        ).strip()
        font_scale = 0.32
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _font, font_scale, _font_thickness
        )
        if text_w > image_width:
            font_scale *= image_width / text_w
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _font, font_scale, _font_thickness
        )

        # Place the label above the box when there is room, otherwise inside.
        label_y_above = y - 2
        if label_y_above - text_h >= 0:
            label_origin = (x, label_y_above)
            bg_top = label_y_above - text_h - baseline
            bg_bottom = label_y_above + baseline
        else:
            label_origin = (x, y + text_h + 1)
            bg_top = y + 1
            bg_bottom = y + text_h + baseline + 2

        # Clamp label origin so it never runs off the right edge.
        label_x = min(label_origin[0], max(0, image_width - text_w - 1))
        label_origin = (label_x, label_origin[1])
        bg_right = min(label_x + text_w, image_width - 1)
        bg_top = max(0, bg_top)
        bg_bottom = min(bg_bottom, image_height - 1)

        # Semi-opaque black background so the label is readable over any color.
        if bg_bottom > bg_top and bg_right > label_x:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (label_x, bg_top), (bg_right, bg_bottom), (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        cv2.putText(
            annotated,
            label,
            label_origin,
            _font,
            font_scale,
            color,
            _font_thickness,
            cv2.LINE_AA,
        )

    output = Path(destination)
    if source_path is not None and output.resolve() == Path(source_path).resolve():
        raise ValueError("Visualization output must not overwrite the source image")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise OSError(f"Could not save measurement visualization: {output}")
    return output


def save_text_measurement_visualizations(
    image: np.ndarray,
    measurements: Sequence[dict[str, Any]],
    ocr_destination: str | Path,
    refined_destination: str | Path,
    *,
    source_path: str | Path | None = None,
    include_object_ids: bool = False,
) -> tuple[Path, Path]:
    """Save OCR-proposal and refined-boundary views from the same measurements."""
    ocr_measurements = [
        {
            **measurement,
            "x": measurement["ocr_bbox"]["x"],
            "y": measurement["ocr_bbox"]["y"],
            "width": measurement["ocr_bbox"]["width"],
            "height": measurement["ocr_bbox"]["height"],
        }
        for measurement in measurements
    ]
    refined_measurements = [
        {
            **measurement,
            "height": measurement["original_height_px"],
            "label_height": measurement["adjusted_height_px"],
        }
        for measurement in measurements
    ]
    ocr_path = save_semantic_measurement_visualization(
        image,
        ocr_measurements,
        ocr_destination,
        source_path=source_path,
        include_object_ids=include_object_ids,
    )
    refined_path = save_semantic_measurement_visualization(
        image,
        refined_measurements,
        refined_destination,
        source_path=source_path,
        include_object_ids=include_object_ids,
    )
    return ocr_path, refined_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure OCR-detected text and numbers in original image pixels."
    )
    parser.add_argument("image", type=Path, help="Original source image")
    parser.add_argument("--output", type=Path, help="Write JSON to this path")
    parser.add_argument(
        "--visualization", type=Path, help="Save a side-by-side annotation image"
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=DEFAULT_DPI,
        help=(
            "Resolution used for pixel-to-point conversion "
            "(72 points per inch; default assumes 96 DPI)"
        ),
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Restrict processing to a half-open ROI in original-image coordinates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None and args.output.resolve() == args.image.resolve():
        raise ValueError("JSON output must not overwrite the source image")
    if (
        args.visualization is not None
        and args.visualization.resolve() == args.image.resolve()
    ):
        raise ValueError("Visualization output must not overwrite the source image")
    roi = (
        (args.roi[0], args.roi[1], args.roi[2], args.roi[3])
        if args.roi is not None
        else None
    )
    result = measure_text_image(args.image, dpi=args.dpi, roi=roi)
    if args.visualization is not None:
        save_measurement_visualization(args.image, result, args.visualization)
    output = json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
