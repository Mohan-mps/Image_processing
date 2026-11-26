"""Derive an image-level chart status and region-level scientific-image labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


ImageLabel = Literal["chart", "non_chart", "undetermined"]

DEFAULT_REGION_TYPES = (
    "microscopy",
    "medical imaging",
    "histology",
    "gel electrophoresis",
    "scientific photography",
    "SEM / material science",
    "satellite imagery",
    "experimental images",
    "scientific diagrams",
    "other visual content",
    "chart",
    "text-only region",
    "uncertain",
)


@dataclass(frozen=True)
class ImageClassification:
    """Image-level status; this is not a learned YOLO class."""

    label: ImageLabel
    method: str
    review_required: bool

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


class ImageClassifier:
    """Summarize graph-verification outcomes and classify scientific image regions."""

    NON_CHART_TYPES = DEFAULT_REGION_TYPES

    def classify(
        self,
        detected_regions: int,
        accepted_graphs: int,
        rejected_non_graphs: int,
    ) -> ImageClassification:
        counts = (detected_regions, accepted_graphs, rejected_non_graphs)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("Classification counts must be non-negative integers")
        if accepted_graphs + rejected_non_graphs != detected_regions:
            raise ValueError(
                "Accepted and rejected region counts must sum to detected regions"
            )

        if accepted_graphs:
            label: ImageLabel = "chart"
        elif rejected_non_graphs:
            label = "non_chart"
        else:
            label = "undetermined"

        return ImageClassification(
            label=label,
            method="yolo_detection_and_graph_verification",
            review_required=label != "chart",
        )

    def classify_region(
        self,
        image: np.ndarray | None = None,
        *,
        region_type: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if image is None and region_type is None:
            raise ValueError("Provide either an image or a region_type")

        if region_type is not None:
            normalized = str(region_type).strip()
            if not normalized:
                raise ValueError("region_type must be a non-empty string")
            chosen_type = normalized
        else:
            chosen_type = self._infer_region_type(image)

        confidence_value = float(confidence if confidence is not None else self._estimate_confidence(chosen_type, image))
        confidence_value = max(0.0, min(1.0, confidence_value))
        non_chart_bool = chosen_type not in {"chart", "text-only region", "uncertain"}
        result = {
            "region_type": chosen_type,
            "image_type": chosen_type,
            "is_non_chart_image": non_chart_bool,
            "confidence": confidence_value,
            "reason": reason or self._default_reason(chosen_type),
        }
        return result

    @staticmethod
    def _infer_region_type(image: np.ndarray | None) -> str:
        if image is None:
            return "uncertain"
        arr = np.asarray(image)
        if arr.size == 0:
            return "uncertain"
        if arr.ndim == 3:
            gray = np.mean(arr, axis=2)
        else:
            gray = arr
        texture = float(np.std(gray))
        if texture < 12:
            return "text-only region"
        if gray.shape[1] > 2 * gray.shape[0]:
            return "scientific diagrams"
        if texture > 60:
            return "microscopy"
        return "scientific photography"

    @staticmethod
    def _estimate_confidence(region_type: str, image: np.ndarray | None) -> float:
        if region_type in {"uncertain", "text-only region", "chart"}:
            return 0.55
        if image is None:
            return 0.7
        arr = np.asarray(image)
        if arr.size == 0:
            return 0.5
        if arr.ndim == 3:
            texture = float(np.std(arr))
        else:
            texture = float(np.std(arr))
        return max(0.6, min(0.98, 0.6 + min(texture / 200.0, 0.35)))

    @staticmethod
    def _default_reason(region_type: str) -> str:
        reasons = {
            "microscopy": "Visible cellular or tissue-like structures suggest a microscopy image.",
            "medical imaging": "Consistent anatomical or scan-like structure indicates medical imaging.",
            "histology": "Tissue morphology and staining patterns suggest histology.",
            "gel electrophoresis": "Distinct bands and lanes indicate gel electrophoresis.",
            "scientific photography": "Natural scientific imagery with visible sample texture is present.",
            "SEM / material science": "Surface microstructure suggests SEM or material science imagery.",
            "satellite imagery": "Earth-observation patterns indicate satellite imagery.",
            "experimental images": "Visual experimental evidence is present in the region.",
            "scientific diagrams": "Diagram-like structure and labels indicate a scientific illustration.",
            "other visual content": "The region contains meaningful visual content but no clear match to the standard categories.",
            "chart": "The region appears to be a chart or plotted graph, not a scientific image.",
            "text-only region": "The region is dominated by text or empty space and is not a non-chart image.",
            "uncertain": "The content is ambiguous and requires expert review.",
        }
        return reasons.get(region_type, "The region is classified based on visible evidence.")
