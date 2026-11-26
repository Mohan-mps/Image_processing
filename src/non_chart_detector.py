"""Ultralytics detector for non-chart classes from YOLO weights."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .yolo_utils import (
    class_name_items,
    clip_box_to_image,
    iter_yolo_predictions,
    normalize_class_name,
    suppress_duplicate_detections,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectDetection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bounding_box": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
        }


class NonChartDetector:
    """Filter non-chart detections from the configured YOLO checkpoint."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
        device: str | None = None,
        *,
        model: Any | None = None,
        duplicate_iou_threshold: float = 0.85,
        duplicate_containment_threshold: float = 0.9,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Non-chart YOLO weights not found: {model_path}")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Non-chart confidence threshold must be in [0, 1]")
        if not 0 <= duplicate_iou_threshold <= 1:
            raise ValueError("Duplicate IoU threshold must be in [0, 1]")
        if not 0 <= duplicate_containment_threshold <= 1:
            raise ValueError("Duplicate containment threshold must be in [0, 1]")
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is required. Install project dependencies with "
                    "'python -m pip install -r requirements.txt'."
                ) from exc
            model = YOLO(str(model_path))
        self._model = model
        self._class_names = self._model.names
        named_non_chart_classes = {
            class_id
            for class_id, class_name in class_name_items(self._class_names)
            if normalize_class_name(class_name)
            in {"non chart", "non graph", "nonchart", "nongraph"}
        }
        named_chart_classes = {
            class_id
            for class_id, class_name in class_name_items(self._class_names)
            if normalize_class_name(class_name) in {"chart", "graph"}
        }
        if named_non_chart_classes:
            self._class_ids: set[int] | None = named_non_chart_classes
        elif named_chart_classes:
            self._class_ids = set()
            LOGGER.warning(
                "YOLO model %s has chart/graph classes but no non-chart/non-graph class",
                model_path,
            )
        else:
            self._class_ids = None
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._duplicate_iou_threshold = duplicate_iou_threshold
        self._duplicate_containment_threshold = duplicate_containment_threshold
        LOGGER.info("Loaded non-chart weights from %s", model_path)

    @property
    def model(self) -> Any:
        """Underlying Ultralytics model shared with another detector."""
        return self._model

    def detect(self, image: np.ndarray) -> list[ObjectDetection]:
        height, width = image.shape[:2]
        detections: list[ObjectDetection] = []
        for row, confidence, class_id in iter_yolo_predictions(
            self._model,
            image,
            self._confidence_threshold,
            self._device,
        ):
            if self._class_ids is not None and class_id not in self._class_ids:
                continue
            bounds = clip_box_to_image(row, width, height)
            if bounds is None:
                LOGGER.warning("Skipping empty non-chart detection: %s", row)
                continue
            detections.append(
                ObjectDetection(
                    class_id=class_id,
                    class_name=str(self._class_names[class_id]),
                    confidence=confidence,
                    x1=bounds[0],
                    y1=bounds[1],
                    x2=bounds[2],
                    y2=bounds[3],
                )
            )
        return self._suppress_duplicates(detections)

    def _suppress_duplicates(
        self, detections: list[ObjectDetection]
    ) -> list[ObjectDetection]:
        return suppress_duplicate_detections(
            detections,
            iou_threshold=self._duplicate_iou_threshold,
            containment_threshold=self._duplicate_containment_threshold,
        )
