"""Validated configuration for the non-chart YOLO, OpenCV, and VLM pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NonChartPipelineConfig:
    model_path: Path | None
    confidence_threshold: float
    duplicate_iou_threshold: float
    duplicate_containment_threshold: float
    crop_padding_fraction: float
    enhance_contrast: bool
    denoise: bool
    upscale: bool
    contrast_clip_limit: float
    denoise_strength: int
    upscale_factor: int
    classify_full_image: bool
    classify_regions: bool
    categories_path: Path
    prompts_directory: Path
    scene_prompt_file: str
    region_prompt_file: str
    crop_directory: Path
    visualization_directory: Path


def _required_object(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Non-chart config field {key!r} must be a JSON object")
    return value


def _boolean(values: dict[str, Any], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Non-chart config field {key!r} must be a boolean")
    return value


def _number(
    values: dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-chart config field {key!r} must be numeric")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"Non-chart config field {key!r} must be {bounds}")
    return result


def _positive_integer(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Non-chart config field {key!r} must be a positive integer")
    return value


def _resolve_path(root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Non-chart config field {key!r} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_non_chart_config(path: str | Path) -> NonChartPipelineConfig:
    """Load and validate the non-chart pipeline configuration JSON."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Non-chart config must be a JSON object: {config_path}")

    detector = _required_object(payload, "detector")
    opencv = _required_object(payload, "opencv")
    vlm = _required_object(payload, "vlm")
    outputs = _required_object(payload, "outputs")

    model_value = payload.get("model_path")
    if model_value is not None and (not isinstance(model_value, str) or not model_value.strip()):
        raise ValueError("Non-chart config field 'model_path' must be a path or null")
    model_path = (
        _resolve_path(config_path.parent, model_value, "model_path")
        if model_value is not None
        else None
    )
    categories_path = _resolve_path(
        config_path.parent, vlm.get("categories_path"), "categories_path"
    )
    prompts_directory = _resolve_path(
        config_path.parent, vlm.get("prompts_directory"), "prompts_directory"
    )
    scene_prompt_file = vlm.get("scene_prompt_file")
    region_prompt_file = vlm.get("region_prompt_file")
    if not isinstance(scene_prompt_file, str) or not scene_prompt_file.strip():
        raise ValueError("Non-chart config field 'scene_prompt_file' must be a non-empty string")
    if not isinstance(region_prompt_file, str) or not region_prompt_file.strip():
        raise ValueError("Non-chart config field 'region_prompt_file' must be a non-empty string")

    return NonChartPipelineConfig(
        model_path=model_path,
        confidence_threshold=_number(
            detector, "confidence_threshold", minimum=0.0, maximum=1.0
        ),
        duplicate_iou_threshold=_number(
            detector, "duplicate_iou_threshold", minimum=0.0, maximum=1.0
        ),
        duplicate_containment_threshold=_number(
            detector, "duplicate_containment_threshold", minimum=0.0, maximum=1.0
        ),
        crop_padding_fraction=_number(
            opencv, "crop_padding_fraction", minimum=0.0, maximum=1.0
        ),
        enhance_contrast=_boolean(opencv, "enhance_contrast"),
        denoise=_boolean(opencv, "denoise"),
        upscale=_boolean(opencv, "upscale"),
        contrast_clip_limit=_number(opencv, "contrast_clip_limit", minimum=0.01),
        denoise_strength=_positive_integer(opencv, "denoise_strength"),
        upscale_factor=_positive_integer(opencv, "upscale_factor"),
        classify_full_image=_boolean(vlm, "classify_full_image"),
        classify_regions=_boolean(vlm, "classify_regions"),
        categories_path=categories_path,
        prompts_directory=prompts_directory,
        scene_prompt_file=scene_prompt_file,
        region_prompt_file=region_prompt_file,
        crop_directory=_resolve_path(
            config_path.parent, outputs.get("crop_directory"), "crop_directory"
        ),
        visualization_directory=_resolve_path(
            config_path.parent,
            outputs.get("visualization_directory"),
            "visualization_directory",
        ),
    )
