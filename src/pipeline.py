"""Convenience API for processing one image through ChartXtract routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .chart_detector import ChartDetector
from .main import process_image as _process_image
from .non_chart_config import load_non_chart_config
from .non_chart_detector import NonChartDetector
from .object_measurement import DEFAULT_DPI
from .prompt_manager import PromptManager
from .vlm_extractor import VLMExtractor


def process_image(
    image_path: str | Path,
    *,
    chart_model_path: str | Path | None = None,
    chart_confidence: float = 0.25,
    non_chart_model_path: str | Path | None = None,
    non_chart_confidence: float | None = None,
    non_chart_config_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    categories_path: str | Path | None = None,
    non_chart_crop_directory: str | Path | None = None,
    non_chart_visualization_directory: str | Path | None = None,
    non_chart_image_directory: str | Path | None = None,
    ollama_url: str = "https://aisandbox.mpslimited.com/ollama",
    ollama_model: str = "qwen3.5:9b",
    ollama_timeout: float = 180.0,
    prompt_mode: str = "task-specific",
    device: str | None = None,
    dpi: float = DEFAULT_DPI,
) -> dict[str, Any]:
    """Process one image and return its unified chart/non-chart JSON result.

    The non-chart checkpoint defaults to the configured path; pass
    non_chart_model_path to override it.
    """
    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")
    project_root = Path(__file__).resolve().parents[1]
    non_chart_config = load_non_chart_config(
        Path(non_chart_config_path)
        if non_chart_config_path is not None
        else project_root / "config" / "non_chart_pipeline.json"
    )
    chart_weights = Path(chart_model_path) if chart_model_path else (
        project_root / "models" / "best_chart_detect (5).pt"
    )
    output = Path(output_directory) if output_directory else project_root / "output"
    prompt_manager = PromptManager()
    prompt_manager.validate_mode(prompt_mode)
    chart_detector = ChartDetector(
        chart_weights,
        confidence_threshold=chart_confidence,
        device=device,
    )
    non_chart_weights = (
        Path(non_chart_model_path)
        if non_chart_model_path is not None
        else non_chart_config.model_path or chart_weights
    )
    shared_model = (
        chart_detector.model
        if non_chart_weights.resolve() == chart_weights.resolve()
        else None
    )
    non_chart_detector = NonChartDetector(
        non_chart_weights,
        confidence_threshold=(
            non_chart_confidence
            if non_chart_confidence is not None
            else non_chart_config.confidence_threshold
        ),
        device=device,
        model=shared_model,
        duplicate_iou_threshold=non_chart_config.duplicate_iou_threshold,
        duplicate_containment_threshold=(
            non_chart_config.duplicate_containment_threshold
        ),
    )
    vlm_extractor = VLMExtractor(
        model=ollama_model,
        base_url=ollama_url,
        timeout=ollama_timeout,
        retries=4,
        retry_backoff=2.0,
    )
    
    # Use context manager to ensure proper cleanup
    try:
        result_path = _process_image(
            source,
            chart_detector,
            project_root / "cropped_charts",
            output,
            visualization_directory=output / "visualizations",
            prompt_mode=prompt_mode,
            axis_crop_directory=project_root / "axis_crops",
            vlm_extractor=vlm_extractor,
            prompt_manager=prompt_manager,
            non_chart_detector=non_chart_detector,
            non_chart_categories_path=(
                Path(categories_path)
                if categories_path is not None
                else non_chart_config.categories_path
            ),
            non_chart_crop_directory=(
                Path(non_chart_crop_directory)
                if non_chart_crop_directory is not None
                else non_chart_config.crop_directory
            ),
            non_chart_visualization_directory=(
                Path(non_chart_visualization_directory)
                if non_chart_visualization_directory is not None
                else non_chart_config.visualization_directory
            ),
            non_chart_image_directory=(
                Path(non_chart_image_directory)
                if non_chart_image_directory is not None
                else project_root / "non_chart_images"
            ),
            non_chart_prompts_directory=non_chart_config.prompts_directory,
            non_chart_config=non_chart_config,
            dpi=dpi,
        )
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        # Ensure VLM extractor resources are cleaned up
        vlm_extractor.close()
