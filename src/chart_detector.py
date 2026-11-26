"""YOLO-based detection of complete chart regions."""

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
class ChartBox:
    """A chart rectangle in original-image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    class_name: str = "chart"

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


class ChartDetector:
    """Load the supplied Ultralytics checkpoint and detect chart regions."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
        device: str | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
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
        named_chart_classes = {
            class_id
            for class_id, class_name in class_name_items(self._class_names)
            if normalize_class_name(class_name) in {"chart", "graph"}
        }
        named_non_chart_classes = {
            class_id
            for class_id, class_name in class_name_items(self._class_names)
            if normalize_class_name(class_name)
            in {"non chart", "non graph", "nonchart", "nongraph"}
        }
        if named_chart_classes:
            self._class_ids: set[int] | None = named_chart_classes
        elif named_non_chart_classes:
            self._class_ids = set()
            LOGGER.warning(
                "YOLO model %s has non-chart/non-graph classes but no chart/graph class",
                model_path,
            )
        else:
            self._class_ids = None
            LOGGER.warning(
                "No chart/graph class name found in %s; accepting all model classes",
                model_path,
            )
        self._confidence_threshold = confidence_threshold
        self._device = device
        LOGGER.info("Loaded chart detector weights from %s", model_path)

    @property
    def model(self) -> Any:
        """Underlying Ultralytics model, shareable with another detector."""
        return self._model

    def detect(self, image: np.ndarray) -> list[ChartBox]:
        """Return chart boxes sorted top-to-bottom, then left-to-right."""
        height, width = image.shape[:2]
        detections: list[ChartBox] = []
        for coordinates, confidence, class_id in iter_yolo_predictions(
            self._model,
            image,
            self._confidence_threshold,
            self._device,
        ):
            if self._class_ids is not None and class_id not in self._class_ids:
                continue
            bounds = clip_box_to_image(coordinates, width, height)
            if bounds is None:
                LOGGER.warning("Skipping empty chart detection: %s", coordinates)
                continue
            detections.append(
                ChartBox(
                    *bounds,
                    confidence,
                    class_id,
                    str(self._class_names[class_id]),
                )
            )
        detections = suppress_duplicate_detections(detections)
        LOGGER.info("Detected %d chart region(s)", len(detections))
        return detections
