"""VLM analysis and visualization for non-chart images.

Note: Scene-level classification (full-image categorization) has been removed.
Only region-level analysis is performed on detected non-chart regions.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from .non_chart_detector import NonChartDetector, ObjectDetection
from .image_preprocessor import ImagePreprocessor
from .non_chart_config import NonChartPipelineConfig, load_non_chart_config
from .non_chart_cropper import NonChartCropper
from .object_measurement import (
    DEFAULT_DPI,
    HEIGHT_ADJUSTMENT_PX,
    POINTS_PER_INCH,
    measure_text_regions,
    save_text_measurement_visualizations,
)
from .prompt_manager import PromptRequest
from .vlm_extractor import VLMExtractor

LOGGER = logging.getLogger(__name__)


def _string_or_none(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Non-chart VLM field {field!r} must be a string or null")
    return value.strip() or None


def _string_list(payload: dict[str, Any], field: str) -> list[str]:
    values = payload.get(field, [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Non-chart VLM field {field!r} must be an array of strings")
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _relationships(payload: dict[str, Any]) -> list[dict[str, str]]:
    values = payload.get("relationships", [])
    if not isinstance(values, list):
        raise ValueError("Non-chart VLM field 'relationships' must be an array")
    relationships: list[dict[str, str]] = []
    for relation in values:
        if not isinstance(relation, dict):
            raise ValueError("Each relationship must be a JSON object")
        cleaned: dict[str, str] = {}
        for key in ("subject", "relation", "object"):
            value = relation.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Relationship field {key!r} must be a non-empty string")
            cleaned[key] = value.strip()
        if cleaned not in relationships:
            relationships.append(cleaned)
    return relationships


def _normalize_category_name(value: str) -> str:
    return " ".join(
        value.strip().lower().replace("-", " ").replace("_", " ").replace("/", " ").split()
    )


def validate_region_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Non-chart region output must be a JSON object")
    return {
        "object_name": _string_or_none(payload.get("object_name"), "object_name"),
        "description": _string_or_none(payload.get("description"), "description"),
        "extracted_text": _string_list(payload, "extracted_text"),
        "relationships": _relationships(payload),
    }


def _prompt(path: Path, **values: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Non-chart prompt not found: {path}")
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def analyze_non_chart_image(
    image_path: Path,
    image: np.ndarray,
    vlm_extractor: VLMExtractor,
    categories_path: Path | None,
    prompt_directory: Path | None,
    detector: NonChartDetector | None,
    crop_directory: Path,
    visualization_directory: Path,
    timings: dict[str, float] | None = None,
    include_scene: bool = True,
    config: NonChartPipelineConfig | None = None,
    ocr_reader: Any | None = None,
    minimum_confidence: float = 0.1,
    chart_boxes: Sequence[tuple[int, int, int, int]] = (),
    dpi: float = DEFAULT_DPI,
    dpi_source: str = "fallback_configuration",
    skip_vlm: bool = False,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    pipeline_config = config or load_non_chart_config(
        project_root / "config" / "non_chart_pipeline.json"
    )
    categories_file = categories_path or pipeline_config.categories_path
    prompts_dir = prompt_directory or pipeline_config.prompts_directory
    region_prompt_path = prompts_dir / pipeline_config.region_prompt_file
    if pipeline_config.classify_regions and not skip_vlm:
        if not region_prompt_path.is_file():
            raise FileNotFoundError(f"Non-chart prompt not found: {region_prompt_path}")
        region_prompt_template = region_prompt_path.read_text(encoding="utf-8")

    detection_started = time.perf_counter()
    detections = detector.detect(image) if detector is not None else []
    if timings is not None:
        timings["yolo_detection_seconds"] = time.perf_counter() - detection_started
    detector_status = (
        "configured" if detector is not None else "not_configured_full_image_only"
    )
    object_results: list[dict[str, Any]] = []
    original_height, original_width = image.shape[:2]
    cropper = NonChartCropper(
        output_directory=crop_directory,
        padding=pipeline_config.crop_padding_fraction,
    )
    preprocessor = ImagePreprocessor(
        contrast_clip_limit=pipeline_config.contrast_clip_limit,
        denoise_strength=pipeline_config.denoise_strength,
        upscale_factor=pipeline_config.upscale_factor,
    )
    for region_id, detection in enumerate(detections, start=1):
        crop_started = time.perf_counter()
        detection_box = (detection.x1, detection.y1, detection.x2, detection.y2)
        crop, crop_bounds, crop_path = cropper.crop_and_save_with_bounds(
            image,
            detection_box,
            filename=f"{image_path.stem}_region_{region_id:03d}.png",
        )
        preprocessing = preprocessor.preprocess(
            crop,
            contrast=pipeline_config.enhance_contrast,
            denoise=pipeline_config.denoise,
            upscale=pipeline_config.upscale,
        )
        if timings is not None:
            timings["opencv_crop_preprocess_seconds"] = timings.get(
                "opencv_crop_preprocess_seconds", 0.0
            ) + time.perf_counter() - crop_started
        if pipeline_config.classify_regions:
            if region_prompt_template is None:
                raise RuntimeError("Non-chart region prompt was not initialized")
            region_prompt = region_prompt_template.replace(
                "{class_name}", detection.class_name
            )
            vlm_started = time.perf_counter()
            region = vlm_extractor.extract_json(
                PromptRequest(region_prompt, (preprocessing.processed,)),
                validate_region_result,
                "non-chart region",
            )
            if timings is not None:
                timings["vlm_region_inference_seconds"] = timings.get(
                    "vlm_region_inference_seconds", 0.0
                ) + time.perf_counter() - vlm_started
        else:
            # Don't add empty region analysis fields
            region = {}
        
        detection_dict = detection.as_dict()
        result = {
            "region_id": region_id,
            "class_name": detection_dict["class_name"],
            "confidence": detection_dict["confidence"],
            "bounding_box": detection_dict["bounding_box"],
            "crop_bounding_box": {
                "x1": crop_bounds[0],
                "y1": crop_bounds[1],
                "x2": crop_bounds[2],
                "y2": crop_bounds[3],
            },
            "crop_path": str(crop_path.resolve()),
        }
        # Only add VLM fields if they exist
        if region:
            if region.get("object_name"):
                result["object_name"] = region["object_name"]
            if region.get("description"):
                result["description"] = region["description"]
            if region.get("extracted_text"):
                result["extracted_text"] = region["extracted_text"]
            if region.get("relationships"):
                result["relationships"] = region["relationships"]
        
        # Only add preprocessing if transforms were applied
        preprocessing_dict = preprocessing.as_dict()
        if preprocessing_dict["transforms"] or preprocessing_dict["upscaled"]:
            result["preprocessing"] = preprocessing_dict
        
        object_results.append(result)

    # Scene analysis removed - always return "unknown" category
    scene = {
        "category": "unknown",
        "summary": None,
        "extracted_text": [],
        "relationships": [],
    }

    ocr_started = time.perf_counter()
    measurements = measure_text_regions(
        image,
        dpi=dpi,
        reader=ocr_reader,
        minimum_confidence=minimum_confidence,
        excluded_boxes=chart_boxes,
    )
    if timings is not None:
        timings["ocr_text_detection_seconds"] = time.perf_counter() - ocr_started
    
    # Removed: non_chart_visualizations/ output
    # Visualization files are no longer created for non-chart pipeline
    
    pipeline_stages = ["yolo"] if detector is not None else []
    if detections:
        pipeline_stages.append("opencv")
    if detections and pipeline_config.classify_regions:
        pipeline_stages.append("vlm")
    
    result = {
        "detection_model_status": detector_status,
        "pipeline_stages": pipeline_stages,
        "detections": object_results,
        "measurement_method": "paddleocr_two_stage_with_opencv_box_refinement",
        "measurement_scale": {
            "dpi": dpi,
            "dpi_source": dpi_source,
            "points_per_inch": POINTS_PER_INCH,
            "conversion": "points = pixels * 72 / dpi",
        },
        "measurements": measurements,
    }
    return result
