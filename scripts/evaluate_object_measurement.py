"""Score object-measurement boxes against manually measured GIMP selections."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from src.object_measurement import (
    DEFAULT_DPI,
    HEIGHT_ADJUSTMENT_PX,
    POINTS_PER_INCH,
    MeasurementConfig,
    measure_text_image,
)


def _validate_box(box: dict[str, Any], context: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(box[key] for key in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context} must contain integer x, y, width, and height") from error
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{context} coordinates and dimensions must be integers")
    x, y, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError(f"{context} width and height must be positive")
    return x, y, width, height


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union


def _match_boxes(
    expected: list[tuple[int, int, int, int]],
    predicted: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = [
        (score, expected_index, predicted_index)
        for expected_index, expected_box in enumerate(expected)
        for predicted_index, predicted_box in enumerate(predicted)
        if (score := _iou(expected_box, predicted_box)) >= iou_threshold
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    matches = []
    for score, expected_index, predicted_index in candidates:
        if expected_index in used_expected or predicted_index in used_predicted:
            continue
        used_expected.add(expected_index)
        used_predicted.add(predicted_index)
        matches.append((expected_index, predicted_index, score))
    return matches


def evaluate_dataset(
    ground_truth_path: str | Path,
    image_directory: str | Path,
    *,
    config: MeasurementConfig | None = None,
    iou_threshold: float = 0.5,
    ocr_reader: Any | None = None,
) -> dict[str, Any]:
    """Measure dataset images and summarize one-to-one GIMP box comparisons."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    truth_path = Path(ground_truth_path)
    image_root = Path(image_directory)
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Ground truth must contain a non-empty 'images' list")

    image_summaries: list[dict[str, Any]] = []
    coordinate_errors: dict[str, list[int]] = {
        "x": [],
        "y": [],
        "width": [],
        "height": [],
    }
    point_coordinate_errors: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "width": [],
        "height": [],
    }
    ious: list[float] = []
    exact_dimensions = 0
    matched_count = 0
    expected_count = 0
    predicted_count = 0
    elapsed_seconds = 0.0
    settings = config or MeasurementConfig()

    for image_entry in images:
        image_name = image_entry.get("image_name")
        objects = image_entry.get("objects")
        if not isinstance(image_name, str) or not image_name:
            raise ValueError("Each image entry must have a non-empty image_name")
        if not isinstance(objects, list):
            raise ValueError(f"{image_name}: objects must be a list")
        expected = [
            _validate_box(box, f"{image_name} object {index}")
            for index, box in enumerate(objects, start=1)
        ]
        image_path = image_root / image_name
        started = time.perf_counter()
        result = measure_text_image(
            image_path,
            dpi=settings.dpi,
            reader=ocr_reader,
        )
        elapsed = time.perf_counter() - started
        elapsed_seconds += elapsed
        predicted = []
        for item in result.measurements:
            if item.original_height_px is None:
                raise ValueError(
                    f"{image_name}: OCR measurement lacks its original text height"
                )
            predicted.append(
                (item.x, item.y, item.width, item.original_height_px)
            )
        matches = _match_boxes(expected, predicted, iou_threshold)
        for expected_index, predicted_index, overlap in matches:
            truth_box = expected[expected_index]
            prediction = predicted[predicted_index]
            for name, truth_value, predicted_value in zip(
                ("x", "y", "width", "height"), truth_box, prediction
            ):
                error = abs(truth_value - predicted_value)
                coordinate_errors[name].append(error)
                point_coordinate_errors[name].append(
                    error * POINTS_PER_INCH / settings.dpi
                )
            exact_dimensions += int(truth_box[2:] == prediction[2:])
            ious.append(overlap)

        matched_count += len(matches)
        expected_count += len(expected)
        predicted_count += len(predicted)
        image_summaries.append(
            {
                "image_name": image_name,
                "expected_objects": len(expected),
                "predicted_objects": len(predicted),
                "matched_objects": len(matches),
                "elapsed_seconds": elapsed,
            }
        )

    def mean_or_none(values: list[int] | list[float]) -> float | None:
        return fmean(values) if values else None

    return {
        "ground_truth": str(truth_path),
        "image_directory": str(image_root),
        "measurement_method": {
            "detector": "easyocr_with_opencv_box_refinement",
            "dpi": settings.dpi,
            "height_adjustment_px": HEIGHT_ADJUSTMENT_PX,
            "point_conversion": "pixels * 72 / dpi",
        },
        "iou_threshold": iou_threshold,
        "images": image_summaries,
        "summary": {
            "image_count": len(images),
            "expected_objects": expected_count,
            "predicted_objects": predicted_count,
            "matched_objects": matched_count,
            "precision": matched_count / predicted_count if predicted_count else 0.0,
            "recall": matched_count / expected_count if expected_count else 0.0,
            "mean_iou": mean_or_none(ious),
            "x_mae_pixels": mean_or_none(coordinate_errors["x"]),
            "y_mae_pixels": mean_or_none(coordinate_errors["y"]),
            "width_mae_pixels": mean_or_none(coordinate_errors["width"]),
            "height_mae_pixels": mean_or_none(coordinate_errors["height"]),
            "x_mae_points": mean_or_none(point_coordinate_errors["x"]),
            "y_mae_points": mean_or_none(point_coordinate_errors["y"]),
            "width_mae_points": mean_or_none(point_coordinate_errors["width"]),
            "height_mae_points": mean_or_none(point_coordinate_errors["height"]),
            "exact_width_height_match_percentage": (
                100.0 * exact_dimensions / matched_count if matched_count else None
            ),
            "total_elapsed_seconds": elapsed_seconds,
            "mean_elapsed_seconds_per_image": elapsed_seconds / len(images),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate OpenCV boxes against manually measured GIMP boxes."
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--dpi",
        type=float,
        default=DEFAULT_DPI,
        help="Resolution used for point metrics (default assumes 96 DPI)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_dataset(
        args.ground_truth,
        args.images,
        config=MeasurementConfig(
            dpi=args.dpi,
        ),
        iou_threshold=args.iou_threshold,
    )
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
