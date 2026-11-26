"""Command-line entry point for ChartXtract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import time
from typing import Any

import cv2

from .axis_preprocessor import AxisPreprocessor
from .chart_cropper import padded_box, save_chart_crop
from .chart_detector import ChartDetector
from .image_classifier import ImageClassifier
from .image_utils import crop_from_bounds, image_metadata_dpi
from .non_chart_config import NonChartPipelineConfig, load_non_chart_config
from .non_chart_detector import NonChartDetector
from .non_chart_pipeline import analyze_non_chart_image
from .object_measurement import (
    DEFAULT_DPI,
    HEIGHT_ADJUSTMENT_PX,
    POINTS_PER_INCH,
    measure_text_regions,
    save_text_measurement_visualizations,
)
from .plot_area_detector import PlotAreaDetector
from .validator import evaluate_tick_values
from .prompt_manager import PROMPT_MODES, PromptManager
from .vlm_extractor import VLMExtractor

LOGGER = logging.getLogger("chartxtract")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _images_from_input(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input image or directory not found: {input_path}")


def process_image(
    image_path: Path,
    detector: ChartDetector,
    crop_directory: Path,
    output_directory: Path,
    manual_plot_region: tuple[float, float, float, float] | None = None,
    visualization_directory: Path | None = None,
    ground_truth: dict[str, Any] | None = None,
    crop_padding_fraction: float = 0.12,
    prompt_mode: str = "task-specific",
    axis_crop_directory: Path | None = None,
    vlm_extractor: VLMExtractor | None = None,
    prompt_manager: PromptManager | None = None,
    timings: dict[str, Any] | None = None,
    non_chart_image_directory: Path | None = None,
    non_chart_detector: NonChartDetector | None = None,
    non_chart_categories_path: Path | None = None,
    non_chart_crop_directory: Path | None = None,
    non_chart_visualization_directory: Path | None = None,
    non_chart_prompts_directory: Path | None = None,
    non_chart_config: NonChartPipelineConfig | None = None,
    ocr_reader: Any | None = None,
    minimum_confidence: float = 0.1,
    dpi: float = DEFAULT_DPI,
    skip_vlm: bool = False,
) -> Path:
    processing_started = time.perf_counter()
    stage_seconds = {
        "image_read_seconds": 0.0,
        "yolo_detection_seconds": 0.0,
        "crop_seconds": 0.0,
        "non_chart_analysis_seconds": 0.0,
        "non_chart_yolo_detection_seconds": 0.0,
        "non_chart_opencv_seconds": 0.0,
        "non_chart_region_vlm_seconds": 0.0,
        "plot_localization_seconds": 0.0,
        "axis_preprocessing_seconds": 0.0,
        "prompt_build_seconds": 0.0,
        "vlm_inference_seconds": 0.0,
        "visualization_seconds": 0.0,
        "json_write_seconds": 0.0,
    }
    stage_started = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    stage_seconds["image_read_seconds"] = time.perf_counter() - stage_started
    if image is None:
        raise OSError(f"Could not read input image: {image_path}")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"Image is empty or has an invalid shape: {image_path}")
    image_channels = 1 if image.ndim == 2 else int(image.shape[2])
    image_format = image_path.suffix.lstrip(".").lower() or None
    metadata_dpi = image_metadata_dpi(image_path)
    dpi_source = "fallback_configuration"
    if dpi == DEFAULT_DPI and metadata_dpi is not None:
        dpi = metadata_dpi
        dpi_source = "image_metadata"
    if image.dtype.name != "uint8":
        image = cv2.convertScaleAbs(
            image, alpha=1 / 256 if image.dtype.itemsize > 1 else 1
        )
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] != 3:
        raise ValueError(f"Unsupported image channel count: {image.shape[2]}")
    if not skip_vlm and (vlm_extractor is None or prompt_manager is None):
        raise ValueError("Qwen VLM extractor and prompt manager are required")
    height, width = image.shape[:2]
    chart_results: list[dict[str, Any]] = []
    measurement_records: list[dict[str, Any]] = []
    axis_tick_viz_records: list[dict[str, Any]] = []
    rejected_non_graphs = 0
    non_chart_detected_regions = 0
    stage_started = time.perf_counter()
    detections = detector.detect(image)
    stage_seconds["yolo_detection_seconds"] = time.perf_counter() - stage_started

    # Initialize OCR reader once in main thread to avoid repeated model loading messages
    # This reader will be shared across all parallel chart processing threads
    # PERFORMANCE: Sharing the reader prevents multiple PaddleOCR model loads
    if ocr_reader is None:
        from .text_detection import _default_reader
        ocr_reader = _default_reader()

    # Process charts in parallel using ThreadPoolExecutor
    # ThreadPoolExecutor is used instead of ProcessPoolExecutor because:
    # 1. We share the VLM extractor session (HTTP connection pooling)
    # 2. OCR caching works across threads with global dict
    # 3. Avoids pickling large objects (image arrays, models)
    # 4. Shared OCR reader prevents repeated model loading
    # PERFORMANCE: Limit workers based on CPU count and detections to prevent memory exhaustion
    import os
    cpu_count = os.cpu_count() or 1
    max_workers = min(4, cpu_count, len(detections)) if len(detections) > 1 else 1
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for chart_id, box in enumerate(detections, start=1):
            future = executor.submit(
                _process_single_chart,
                chart_id,
                box,
                image,
                image_path,
                width,
                height,
                crop_directory,
                axis_crop_directory,
                output_directory,
                crop_padding_fraction,
                manual_plot_region,
                prompt_mode,
                vlm_extractor,
                prompt_manager,
                ground_truth,
                ocr_reader,
                minimum_confidence,
                dpi,
                skip_vlm,
            )
            futures[future] = chart_id
        
        # Collect results as they complete
        for future in as_completed(futures):
            chart_id = futures[future]
            try:
                chart_result, _, chart_viz_records, chart_timings = future.result()
                
                # Accumulate timing data
                for key, value in chart_timings.items():
                    stage_seconds[key] = stage_seconds.get(key, 0.0) + value
                
                if chart_result is not None:
                    chart_results.append(chart_result)
                else:
                    rejected_non_graphs += 1
                
                axis_tick_viz_records.extend(chart_viz_records)
                
            except Exception as exc:
                LOGGER.error("Chart %d processing failed: %s", chart_id, exc, exc_info=True)
                raise
    
    # Sort chart_results by chart_id to maintain order
    chart_results.sort(key=lambda c: c["chart_id"])

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{image_path.stem}.json"
    image_classification = ImageClassifier().classify(
        len(detections), len(chart_results), rejected_non_graphs
    )
    image_classification_payload = image_classification.as_dict()
    payload: dict[str, Any] = {
        "source_image": image_path.name,
        "image_width": width,
        "image_height": height,
        "original_dimensions": {
            "width": width,
            "height": height,
            "channels": image_channels,
            "format": image_format,
        },
        "measurement_method": "opencv_original_pixel_coordinates",
        "image_classification": image_classification_payload,
        "charts": chart_results,
    }
    is_chart = image_classification.label == "chart"
    # Add image_type for backward compatibility with tests
    payload["image_type"] = "chart" if is_chart else "non_chart"

    non_chart_result: dict[str, Any] | None = None
    if not is_chart or non_chart_detector is not None:
        project_root = Path(__file__).resolve().parents[1]
        stage_started = time.perf_counter()
        non_chart_timings: dict[str, float] = {}
        non_chart_result = analyze_non_chart_image(
            image_path=image_path,
            image=image,
            vlm_extractor=vlm_extractor,
            categories_path=non_chart_categories_path
            or project_root / "config" / "image_categories.json",
            prompt_directory=non_chart_prompts_directory
            or project_root / "prompts",
            detector=non_chart_detector,
            crop_directory=non_chart_crop_directory
            or project_root / "non_chart_crops",
            visualization_directory=non_chart_visualization_directory
            or project_root / "output" / "non_chart_visualizations",
            timings=non_chart_timings,
            include_scene=not is_chart,
            config=non_chart_config,
            ocr_reader=ocr_reader,
            minimum_confidence=minimum_confidence,
            chart_boxes=[tuple(box.as_list()) for box in detections],
            dpi=dpi,
            dpi_source=dpi_source,
            skip_vlm=skip_vlm,
        )
        stage_seconds["non_chart_analysis_seconds"] = (
            time.perf_counter() - stage_started
        )
        stage_seconds["non_chart_yolo_detection_seconds"] = non_chart_timings.get(
            "yolo_detection_seconds", 0.0
        )
        stage_seconds["non_chart_opencv_seconds"] = non_chart_timings.get(
            "opencv_crop_preprocess_seconds", 0.0
        )
        stage_seconds["non_chart_region_vlm_seconds"] = non_chart_timings.get(
            "vlm_region_inference_seconds", 0.0
        )
        non_chart_detected_regions = len(non_chart_result["detections"])
        measurement_records.extend(non_chart_result["measurements"])
        if is_chart:
            payload["non_chart_detections"] = non_chart_result["detections"]
            payload["non_chart_pipeline_stages"] = non_chart_result["pipeline_stages"]
        else:
            image_classification_payload.update(
                {
                    "label": "non_chart",
                    "method": "chart_detection_then_ollama_vlm",
                    "review_required": non_chart_result.get("category", "unknown") == "unknown",
                    "chart_detector_label": image_classification.label,
                }
            )
            # Removed: copying non-chart images to non_chart_images directory
            payload.update(non_chart_result)
            # Add category field for backward compatibility
            payload["category"] = non_chart_result.get("category", "unknown")
            payload["image_classification"] = image_classification_payload
    if non_chart_result is None:
        measurement_records = measure_text_regions(
            image,
            dpi=dpi,
            reader=ocr_reader,
            minimum_confidence=minimum_confidence,
            excluded_boxes=[tuple(box.as_list()) for box in detections],
        )
    for object_id, measurement in enumerate(measurement_records, start=1):
        measurement["object_id"] = object_id
    payload["original_dimensions"] = {
        "width": width,
        "height": height,
        "channels": image_channels,
        "format": image_format,
    }
    measurement_scale: dict[str, Any] = {
        "dpi": dpi,
        "points_per_inch": POINTS_PER_INCH,
        "conversion": "points = pixels * 72 / dpi",
    }
    if not is_chart:
        measurement_scale["dpi_source"] = dpi_source
    payload["measurement_scale"] = measurement_scale
    payload["measurements"] = measurement_records
    visualization_path = (
        visualization_directory or output_directory / "visualizations"
    ) / f"{image_path.stem}_measurements.png"
    ocr_visualization_path = (
        visualization_directory or output_directory / "visualizations"
    ) / f"{image_path.stem}_ocr_detection.png"
    stage_started = time.perf_counter()
    # Append axis tick OCR boxes only for visualization — they already exist
    # in the JSON under x_tick_ocr / y_tick_ocr and must not pollute the
    # non_chart_text_extraction block which expects full measurement fields.
    save_text_measurement_visualizations(
        image,
        list(measurement_records) + axis_tick_viz_records,
        ocr_visualization_path,
        visualization_path,
        source_path=image_path,
        include_object_ids=True,
    )
    stage_seconds["visualization_seconds"] += time.perf_counter() - stage_started
    payload["ocr_detection_visualization"] = str(ocr_visualization_path.resolve())
    payload["non_chart_text_extraction"] = {
        "source_image": image_path.name,
        "measurement_units": {
            "coordinates": "pixels",
            "dimensions": "pixels",
            "converted_dimensions": "points",
        },
        "dpi": dpi,
        "dpi_source": dpi_source,
        "measurement_method": "paddleocr_two_stage_with_opencv_box_refinement",
        "extracted_text": [
            {
                "object_id": measurement["object_id"],
                "text": measurement["text"],
                "content_type": measurement["content_type"],
                "detection_confidence": measurement["detection_confidence"],
                "recognition_confidence": measurement["recognition_confidence"],
                "preprocessing": measurement["preprocessing"],
                "ocr_bbox": measurement["ocr_bbox"],
                "refined_bbox": measurement["refined_bbox"],
                "x_px": measurement["x"],
                "y_px": measurement["y"],
                "width_px": measurement["width_px"],
                "original_height_px": measurement["original_height_px"],
                "adjusted_height_px": measurement["adjusted_height_px"],
                "width_points": measurement["width_points"],
                "height_points": measurement["height_points"],
                "dpi": dpi,
                "bounding_box": {
                    "x_min": measurement["x"],
                    "y_min": measurement["y"],
                    "x_max": measurement["x"] + measurement["width_px"],
                    "y_max": measurement["y"]
                    + measurement["adjusted_height_px"],
                },
            }
            for measurement in measurement_records
        ],
        "ocr_visualization": str(ocr_visualization_path.resolve()),
        "refined_visualization": str(visualization_path.resolve()),
    }
    # Removed: separate non_chart_measurements/*.json file writing
    # The non_chart_text_extraction data remains in the main output JSON
    payload["measurement_limitations"] = [
        "OCR may omit faint, rotated, low-resolution, or text-like graphical marks; "
        "reported measurements are OCR-identified text regions only."
    ]
    payload["measurement_visualization"] = str(visualization_path.resolve())
    stage_started = time.perf_counter()
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stage_seconds["json_write_seconds"] = time.perf_counter() - stage_started
    if timings is not None:
        image_size_bytes = image_path.stat().st_size
        timings.update(
            {
                "file_name": image_path.name,
                "input_path": str(image_path.resolve()),
                "file_size_bytes": image_size_bytes,
                "file_size_mb": round(image_size_bytes / 1_000_000, 6),
                "image_width": width,
                "image_height": height,
                "detected_regions": len(detections),
                "non_chart_detected_regions": non_chart_detected_regions,
                "accepted_graphs": len(chart_results),
                "rejected_non_graphs": rejected_non_graphs,
                **{
                    name: round(seconds, 6)
                    for name, seconds in stage_seconds.items()
                },
                "image_total_seconds": round(
                    time.perf_counter() - processing_started, 6
                ),
            }
        )
    LOGGER.info("Wrote %s with %d chart(s)", output_path, len(chart_results))
    return output_path


def _process_single_chart(
    chart_id: int,
    box: Any,
    image: Any,  # np.ndarray
    image_path: Path,
    width: int,
    height: int,
    crop_directory: Path,
    axis_crop_directory: Path | None,
    output_directory: Path,
    crop_padding_fraction: float,
    manual_plot_region: tuple[float, float, float, float] | None,
    prompt_mode: str,
    vlm_extractor: VLMExtractor,
    prompt_manager: PromptManager,
    ground_truth: dict[str, Any] | None,
    ocr_reader: Any | None,
    minimum_confidence: float,
    dpi: float,
    skip_vlm: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Process a single chart and return its results and timing data."""
    
    stage_timings = {
        "crop_seconds": 0.0,
        "plot_localization_seconds": 0.0,
        "chart_ocr_seconds": 0.0,
        "axis_preprocessing_seconds": 0.0,
        "prompt_build_seconds": 0.0,
        "vlm_inference_seconds": 0.0,
    }
    
    plot_area_detector = PlotAreaDetector()
    axis_preprocessor = AxisPreprocessor()
    
    stage_started = time.perf_counter()
    crop_bounds = padded_box(image, box, crop_padding_fraction)
    crop = crop_from_bounds(image, crop_bounds)
    crop_path = save_chart_crop(
        image,
        box,
        image_path,
        chart_id,
        crop_directory,
        crop_padding_fraction,
        crop=crop,
    )
    stage_timings["crop_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    plot = plot_area_detector.detect(crop, manual_plot_region)
    stage_timings["plot_localization_seconds"] = time.perf_counter() - stage_started

    # Run OCR on the full chart crop
    stage_started = time.perf_counter()
    chart_ocr_regions = measure_text_regions(
        crop,
        dpi=dpi,
        reader=ocr_reader,
        minimum_confidence=minimum_confidence,
    )
    stage_timings["chart_ocr_seconds"] = time.perf_counter() - stage_started
    
    chart_crop_ocr_results = chart_ocr_regions

    stage_started = time.perf_counter()
    axis_crops = axis_preprocessor.prepare(crop, plot)
    axis_crop_dir = axis_crop_directory or output_directory / "axis_crops"
    axis_crop_dir.mkdir(parents=True, exist_ok=True)
    for name, axis_image in (
        ("x", axis_crops.x_axis),
        ("y", axis_crops.y_axis),
        ("y_rotated", axis_crops.y_axis_rotated),
    ):
        axis_crop_path = axis_crop_dir / f"{image_path.stem}_chart_{chart_id:03d}_{name}.png"
        if not cv2.imwrite(str(axis_crop_path), axis_image):
            raise OSError(f"Could not save preprocessed axis crop: {axis_crop_path}")
    stage_timings["axis_preprocessing_seconds"] = time.perf_counter() - stage_started

    # OCR the axis crops
    def _ocr_axis_crop(
        axis_image: Any,  # np.ndarray
        bounds: tuple[int, int, int, int],
    ) -> list[dict[str, Any]]:
        scale = axis_preprocessor.scale
        regions = measure_text_regions(
            axis_image,
            dpi=dpi,
            reader=ocr_reader,
            minimum_confidence=minimum_confidence,
        )
        origin_x, origin_y = bounds[0], bounds[1]
        result = []
        for region in regions:
            x_in_crop = round(region["x"] / scale) + origin_x
            y_in_crop = round(region["y"] / scale) + origin_y
            w_in_crop = max(1, round(region["width_px"] / scale))
            h_in_crop = max(1, round(region["original_height_px"] / scale))
            result.append(
                {
                    "text": region["text"],
                    "content_type": region["content_type"],
                    "confidence": region["recognition_confidence"],
                    "detection_confidence": region["detection_confidence"],
                    "preprocessing": region["preprocessing"],
                    "bbox_in_crop": {
                        "x": x_in_crop,
                        "y": y_in_crop,
                        "width": w_in_crop,
                        "height": h_in_crop,
                    },
                }
            )
        return result

    x_tick_ocr = _ocr_axis_crop(axis_crops.x_axis, axis_crops.x_bounds)
    y_tick_ocr = _ocr_axis_crop(axis_crops.y_axis, axis_crops.y_bounds)

    # Convert coordinates to original image space
    crop_offset_x, crop_offset_y = int(crop_bounds[0]), int(crop_bounds[1])
    
    axis_tick_viz_records = []
    # Convert chart crop OCR results
    for region in chart_crop_ocr_results:
        x_img = region["x"] + crop_offset_x
        y_img = region["y"] + crop_offset_y
        w_img = region["width_px"]
        h_img = region["original_height_px"]
        if (
            x_img >= 0 and y_img >= 0
            and x_img + w_img <= width
            and y_img + h_img <= height
        ):
            axis_tick_viz_records.append(
                {
                    "x": x_img,
                    "y": y_img,
                    "width": w_img,
                    "height": h_img,
                    "text": region["text"],
                    "content_type": region["content_type"],
                    "ocr_bbox": {"x": x_img, "y": y_img, "width": w_img, "height": h_img},
                    "original_height_px": h_img,
                    "adjusted_height_px": h_img,
                    "label_height": h_img,
                }
            )
    
    # Add axis crop OCR results
    for tick in x_tick_ocr + y_tick_ocr:
        b = tick["bbox_in_crop"]
        x_img = int(b["x"]) + crop_offset_x
        y_img = int(b["y"]) + crop_offset_y
        w_img = int(b["width"])
        h_img = max(1, int(b["height"]))
        if (
            x_img >= 0 and y_img >= 0
            and x_img + w_img <= width
            and y_img + h_img <= height
        ):
            axis_tick_viz_records.append(
                {
                    "x": x_img,
                    "y": y_img,
                    "width": w_img,
                    "height": h_img,
                    "text": tick["text"],
                    "content_type": tick["content_type"],
                    "ocr_bbox": {"x": x_img, "y": y_img, "width": w_img, "height": h_img},
                    "original_height_px": h_img,
                    "adjusted_height_px": h_img,
                    "label_height": h_img,
                }
            )

    if skip_vlm:
        x_ticks = list(dict.fromkeys(str(tick["text"]).strip() for tick in x_tick_ocr if str(tick.get("text", "")).strip()))
        y_ticks = list(dict.fromkeys(str(tick["text"]).strip() for tick in y_tick_ocr if str(tick.get("text", "")).strip()))
        uncertain_values: list[str] = []
        is_graph = True
    else:
        stage_started = time.perf_counter()
        request = prompt_manager.build(
            prompt_mode,
            crop,
            axis_crops.x_axis,
            axis_crops.y_axis,
            axis_crops.y_axis_rotated,
        )
        stage_timings["prompt_build_seconds"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        vlm_result = vlm_extractor.extract(request)
        stage_timings["vlm_inference_seconds"] = time.perf_counter() - stage_started
        is_graph = vlm_result["is_graph"]
        x_ticks = vlm_result["x_axis_tick"]
        y_ticks = vlm_result["y_axis_tick"]
        uncertain_values = vlm_result["uncertain_values"]

    if not is_graph:
        LOGGER.warning(
            "Skipped YOLO detection %d in %s: VLM classified it as not a graph",
            chart_id,
            image_path.name,
        )
        return (None, [], axis_tick_viz_records, stage_timings)

    review_candidates = [
        {"text": value, "axis": "unknown", "reason": "vlm_uncertain"}
        for value in uncertain_values
    ]
    review_flags: list[str] = []
    if not plot.horizontal_axis_detected:
        review_flags.append("horizontal_plot_axis_not_detected")
    if not plot.vertical_axis_detected:
        review_flags.append("vertical_plot_axis_not_detected")
    if plot.confidence < 0.5:
        review_flags.append("plot_area_low_confidence")
    if uncertain_values:
        review_flags.append("vlm_uncertain_values")
    if not x_ticks:
        review_flags.append("no_x_ticks_detected")
    if not y_ticks:
        review_flags.append("no_y_ticks_detected")
    
    chart_result: dict[str, Any] = {
        "chart_id": chart_id,
        "bounding_box": box.as_list(),
        "class_name": box.class_name,
        "detection_confidence": round(box.confidence, 4),
        "crop_bounding_box": list(crop_bounds),
        "plot_area": plot.as_dict(),
        "plot_area_confidence": round(plot.confidence, 4),
        "plot_area_method": plot.method,
        "x_axis_tick": x_ticks,
        "y_axis_tick": y_ticks,
    }
    if review_candidates:
        chart_result["review_candidates"] = review_candidates
    if review_flags:
        chart_result["review_required"] = True
        chart_result["review_flags"] = review_flags
    if uncertain_values:
        chart_result["uncertain_values"] = uncertain_values
    chart_result["x_tick_ocr"] = x_tick_ocr
    chart_result["y_tick_ocr"] = y_tick_ocr
    chart_result["recognition_backend"] = (
        "paddleocr" if skip_vlm else vlm_extractor.model
    )
    chart_result["prompt_mode"] = prompt_mode
    
    chart_ground_truth = (
        next(
            (
                expected
                for expected in ground_truth.get("charts", [])
                if expected.get("chart_id") == chart_id
            ),
            None,
        )
        if ground_truth
        else None
    )
    if chart_ground_truth is not None:
        chart_result["axis_accuracy"] = {
            axis: evaluate_tick_values(
                _ground_truth_values(chart_ground_truth, axis),
                _ground_truth_values(chart_result, axis),
            )
            for axis in ("x", "y")
        }
        if any(
            not metrics["exact_sequence_match"]
            for metrics in chart_result["axis_accuracy"].values()
        ):
            review_flags.append("ground_truth_mismatch")
            chart_result["review_required"] = True
            chart_result["review_flags"] = review_flags
    
    return (chart_result, [], axis_tick_viz_records, stage_timings)


def _ground_truth_values(chart: dict[str, Any], axis: str) -> list[str]:
    ticks = chart.get(f"{axis}_axis_tick", [])
    return [
        str(tick["value"] if isinstance(tick, dict) else tick)
        for tick in ticks
    ]


def _read_ground_truth(path: Path | None, image_name: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", [payload])
    for image_ground_truth in images:
        if image_ground_truth.get("image_name") == image_name:
            return image_ground_truth
    raise ValueError(f"No ground truth for {image_name!r} in {path}")


def _accuracy_summary(image_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for axis in ("x", "y"):
        counts = {
            "expected_count": 0,
            "predicted_count": 0,
            "matched_count": 0,
        }
        for image_result in image_results:
            for chart in image_result["charts"]:
                metrics = chart.get("axis_accuracy", {}).get(axis)
                if metrics:
                    for key in counts:
                        counts[key] += metrics[key]
        denominator = max(counts["expected_count"], counts["predicted_count"])
        counts["accuracy"] = (
            counts["matched_count"] / denominator if denominator else 1.0
        )
        summary[f"{axis}_axis"] = counts
    return summary


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract chart axis tick labels from scientific figures."
    )
    parser.add_argument("--input", type=Path, default=project_root / "input")
    parser.add_argument("--output", type=Path, default=project_root / "output")
    # Removed: --timings-csv argument (processing_times.csv output removed)
    parser.add_argument(
        "--crops", type=Path, default=project_root / "cropped_charts"
    )
    # Removed: --non-chart-images argument (directory output removed)
    parser.add_argument(
        "--non-chart-config",
        type=Path,
        default=project_root / "config" / "non_chart_pipeline.json",
        help="JSON configuration for non-chart YOLO, OpenCV, VLM, and outputs.",
    )
    parser.add_argument(
        "--non-chart-model",
        type=Path,
        help=(
            "YOLO weights for non-chart detections. Defaults to the model_path "
            "in --non-chart-config."
        ),
    )
    parser.add_argument(
        "--non-chart-confidence",
        type=float,
        default=None,
        help="Override the non-chart detector confidence from --non-chart-config.",
    )
    parser.add_argument(
        "--non-chart-categories",
        type=Path,
        default=None,
        help="Override the category file from --non-chart-config.",
    )
    parser.add_argument(
        "--non-chart-prompts",
        type=Path,
        default=None,
        help="Override the prompt directory from --non-chart-config.",
    )
    parser.add_argument(
        "--non-chart-crops",
        type=Path,
        default=None,
        help="Override the crop directory from --non-chart-config.",
    )
    # Removed: --non-chart-visualizations argument (non_chart_visualizations directory removed)
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "models" / "best_chart_detect (5).pt",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--dpi",
        type=float,
        default=DEFAULT_DPI,
        help="Resolution used for non-chart pixel-to-point conversion (default: 96).",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.12,
        help="Crop context as a fraction of detected chart width and height.",
    )
    parser.add_argument(
        "--plot-region",
        type=float,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Manual plot bounds normalized to the crop dimensions (0 to 1).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="JSON file with manually verified tick values for accuracy reporting.",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Accepted for compatibility; Qwen is now the only recognizer.",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip all VLM calls; use YOLO/OpenCV/PaddleOCR only.",
    )
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument(
        "--ollama-url",
        default="https://aisandbox.mpslimited.com/ollama",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default="task-specific",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "dataset"
        / "few_shot_examples"
        / "examples.json",
    )
    parser.add_argument(
        "--axis-crops",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "axis_crops",
    )
    parser.add_argument("--ollama-temperature", type=float, default=0.0)
    parser.add_argument("--ollama-num-predict", type=int, default=512)
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=180.0,
        help="Ollama response read timeout in seconds",
    )
    parser.add_argument("--ollama-retries", type=int, default=4)
    parser.add_argument(
        "--ollama-retry-backoff",
        type=float,
        default=2.0,
        help="Initial backoff in seconds for exponential retry delay (default: 2.0)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    # Suppress verbose PaddlePaddle and PaddleX logging
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
    
    # Suppress PaddlePaddle INFO logs
    paddle_logger = logging.getLogger("ppocr")
    paddle_logger.setLevel(logging.WARNING)
    paddle_logger = logging.getLogger("paddle")
    paddle_logger.setLevel(logging.WARNING)
    
    image_paths = _images_from_input(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.input}")

    setup_started = time.perf_counter()
    prompt_manager = PromptManager(examples_manifest=args.few_shot_examples)
    prompt_manager.validate_mode(args.prompt_mode)
    non_chart_config = load_non_chart_config(args.non_chart_config)
    model_load_started = time.perf_counter()
    detector = ChartDetector(args.model, args.confidence, args.device)
    model_load_seconds = time.perf_counter() - model_load_started
    non_chart_model_path = (
        args.non_chart_model or non_chart_config.model_path or args.model
    )
    non_chart_confidence = (
        args.non_chart_confidence
        if args.non_chart_confidence is not None
        else non_chart_config.confidence_threshold
    )
    model_load_started = time.perf_counter()
    shared_model = (
        detector.model
        if non_chart_model_path.resolve() == args.model.resolve()
        else None
    )
    non_chart_detector = NonChartDetector(
        non_chart_model_path,
        non_chart_confidence,
        args.device,
        model=shared_model,
        duplicate_iou_threshold=non_chart_config.duplicate_iou_threshold,
        duplicate_containment_threshold=(
            non_chart_config.duplicate_containment_threshold
        ),
    )
    non_chart_model_load_seconds = time.perf_counter() - model_load_started
    vlm_extractor = None
    if not args.skip_vlm:
        vlm_extractor = VLMExtractor(
            args.ollama_model,
            args.ollama_url,
            args.ollama_temperature,
            num_predict=args.ollama_num_predict,
            timeout=args.ollama_timeout,
            retries=args.ollama_retries,
            retry_backoff=args.ollama_retry_backoff,
        )
    
    try:
        run_setup_seconds = time.perf_counter() - setup_started
        image_results: list[dict[str, Any]] = []
        # Removed: processing_times.csv tracking and writing
        for image_path in image_paths:
            ground_truth = _read_ground_truth(args.ground_truth, image_path.name)
            output_path = process_image(
                image_path,
                detector,
                args.crops,
                args.output,
                tuple(args.plot_region) if args.plot_region else None,
                args.output / "visualizations",
                ground_truth,
                args.crop_padding,
                args.prompt_mode,
                args.axis_crops,
                vlm_extractor,
                prompt_manager,
                None,  # timings parameter removed
                None,  # non_chart_image_directory removed
                non_chart_detector,
                args.non_chart_categories,
                args.non_chart_crops or non_chart_config.crop_directory,
                None,  # non_chart_visualization_directory removed
                args.non_chart_prompts or non_chart_config.prompts_directory,
                non_chart_config,
                dpi=args.dpi,
                skip_vlm=args.skip_vlm,
            )
            # Removed: timing_rows append and CSV writing
            image_results.append(json.loads(output_path.read_text(encoding="utf-8")))
        # Removed: logging for processing_times.csv
        if args.ground_truth is not None:
            report_path = args.output / "accuracy_report.json"
            report_path.write_text(
                json.dumps(_accuracy_summary(image_results), indent=2) + "\n",
                encoding="utf-8",
            )
            LOGGER.info("Wrote ground-truth accuracy report to %s", report_path)
    finally:
        # Ensure VLM extractor session is properly closed
        if vlm_extractor is not None:
            vlm_extractor.close()


if __name__ == "__main__":
    main()
