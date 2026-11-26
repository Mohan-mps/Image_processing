"""Shared helpers for extracting Ultralytics detection results."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

import numpy as np


class DetectionBox(Protocol):
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


DetectionT = TypeVar("DetectionT", bound=DetectionBox)


def class_name_items(names: Any) -> list[tuple[int, str]]:
    if isinstance(names, dict):
        return [(int(class_id), str(name)) for class_id, name in names.items()]
    return [(index, str(name)) for index, name in enumerate(names)]


def normalize_class_name(name: str) -> str:
    return " ".join(name.casefold().replace("_", " ").replace("-", " ").split())


def iter_yolo_predictions(
    model: Any,
    image: np.ndarray,
    confidence_threshold: float,
    device: str | None,
) -> Iterator[tuple[np.ndarray, float, int]]:
    """Yield each raw XYXY prediction and its confidence and class ID."""
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        device=device,
        verbose=False,
    )
    for result in results:
        if result.boxes is None:
            continue
        coordinates = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for row, confidence, class_index in zip(coordinates, confidences, classes):
            yield row, float(confidence), int(class_index)


def clip_box_to_image(
    coordinates: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Round XYXY coordinates, clamp them to the image, and reject empty boxes."""
    x1, y1, x2, y2 = (int(round(value)) for value in coordinates.tolist())
    left = max(0, min(width, x1))
    top = max(0, min(height, y1))
    right = max(0, min(width, x2))
    bottom = max(0, min(height, y2))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def suppress_duplicate_detections(
    detections: list[DetectionT],
    *,
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.9,
) -> list[DetectionT]:
    """Suppress same-class duplicate boxes while retaining enclosing detections."""
    selected: list[DetectionT] = []
    for candidate in sorted(
        detections, key=lambda box: (-box.confidence, box.y1, box.x1)
    ):
        duplicate_index: int | None = None
        replace_existing = False
        for index, existing in enumerate(selected):
            if candidate.class_id != existing.class_id:
                continue
            intersection = _intersection_area(candidate, existing)
            if not intersection:
                continue
            candidate_area = _box_area(candidate)
            existing_area = _box_area(existing)
            containment = intersection / min(candidate_area, existing_area)
            union = candidate_area + existing_area - intersection
            iou = intersection / union
            if containment >= containment_threshold:
                duplicate_index = index
                replace_existing = candidate_area > existing_area
                break
            if iou >= iou_threshold:
                duplicate_index = index
                break
        if duplicate_index is None:
            selected.append(candidate)
        elif replace_existing:
            selected[duplicate_index] = candidate
    selected.sort(key=lambda box: (box.y1, box.x1, box.class_id))
    return selected


def _box_area(box: DetectionBox) -> int:
    """Calculate bounding box area with overflow protection."""
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    # Use numpy int64 to prevent overflow with large boxes
    area = int(np.int64(width) * np.int64(height))
    return area


def _intersection_area(first: DetectionBox, second: DetectionBox) -> int:
    """Calculate intersection area with overflow protection."""
    width = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    # Use numpy int64 to prevent overflow
    area = int(np.int64(width) * np.int64(height))
    return area
