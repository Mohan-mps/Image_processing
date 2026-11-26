"""Compare zero-shot, task, few-shot, and enhanced-crop VLM prompting."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .axis_preprocessor import AxisPreprocessor
from .chart_cropper import padded_box
from .chart_detector import ChartDetector
from .image_utils import crop_from_bounds
from .plot_area_detector import PlotAreaDetector
from .prompt_manager import PROMPT_MODES, PromptManager
from .validator import normalize_tick_value
from .vlm_extractor import VLMExtractor

LOGGER = logging.getLogger("chartxtract.evaluation")
_ALL_MODES = ("zero-shot", "task-specific", "few-shot", "few-shot-enhanced-crops")
_NUMBER_LIKE = re.compile(r"^[+\-\d.,eE^×xX·⁰¹²³⁴⁵⁶⁷⁸⁹−]+$")


def _edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, start=1):
        current = [row]
        for column, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _normalized_labels(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = normalize_tick_value(value)
        result.append(normalized if normalized is not None else value.strip())
    return result


def _axis_metrics(samples: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    expected_flat: list[str] = []
    predicted_flat: list[str] = []
    exact = 0
    uncertain = 0
    for sample in samples:
        expected = sample[f"{axis}_axis_tick"]
        predicted = sample["prediction"][f"{axis}_axis_tick"]
        expected_flat.extend(expected)
        predicted_flat.extend(predicted)
        exact += _normalized_labels(expected) == _normalized_labels(predicted)
        uncertain += len(sample["prediction"]["uncertain_values"])
    matched = sum((Counter(expected_flat) & Counter(predicted_flat)).values())
    accuracy_denominator = max(len(expected_flat), len(predicted_flat))
    character_total = sum(len(value) for value in expected_flat)
    character_errors = sum(
        _edit_distance(expected, predicted)
        for sample in samples
        for expected, predicted in zip(
            sample[f"{axis}_axis_tick"],
            sample["prediction"][f"{axis}_axis_tick"],
        )
    ) + sum(
        len(value)
        for sample in samples
        for value in sample[f"{axis}_axis_tick"][len(sample["prediction"][f"{axis}_axis_tick"]) :]
    ) + sum(
        len(value)
        for sample in samples
        for value in sample["prediction"][f"{axis}_axis_tick"][len(sample[f"{axis}_axis_tick"]) :]
    )
    def type_accuracy(numeric: bool) -> dict[str, int | float]:
        expected_values: list[str] = []
        predicted_values: list[str] = []
        for sample in samples:
            expected_values.extend(
                value
                for value in sample[f"{axis}_axis_tick"]
                if bool(_NUMBER_LIKE.fullmatch(value.strip())) == numeric
            )
            predicted_values.extend(
                value
                for value in sample["prediction"][f"{axis}_axis_tick"]
                if bool(_NUMBER_LIKE.fullmatch(value.strip())) == numeric
            )
        matched_count = sum(
            (
                Counter(_normalized_labels(expected_values))
                & Counter(_normalized_labels(predicted_values))
            ).values()
        )
        denominator = max(len(expected_values), len(predicted_values))
        return {
            "expected_count": len(expected_values),
            "predicted_count": len(predicted_values),
            "matched_count": matched_count,
            "accuracy": matched_count / denominator if denominator else 0.0,
        }

    return {
        "chart_axis_exact_match_accuracy": exact / len(samples) if samples else 0.0,
        "chart_axis_exact_match_count": exact,
        "chart_axis_count": len(samples),
        "tick_match_accuracy": matched / accuracy_denominator if accuracy_denominator else 0.0,
        "expected_tick_count": len(expected_flat),
        "predicted_tick_count": len(predicted_flat),
        "character_error_rate": character_errors / character_total if character_total else 0.0,
        "character_error_count": character_errors,
        "expected_character_count": character_total,
        "numerical_tick_accuracy": type_accuracy(True),
        "categorical_tick_accuracy": type_accuracy(False),
        "uncertain_prediction_rate": uncertain / max(1, len(predicted_flat) + uncertain),
        "uncertain_value_count": uncertain,
    }


def score_mode(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate axis-specific and aggregate metrics for one prompt configuration."""
    x_metrics = _axis_metrics(samples, "x")
    y_metrics = _axis_metrics(samples, "y")
    all_expected = sum(
        len(sample["x_axis_tick"]) + len(sample["y_axis_tick"]) for sample in samples
    )
    all_predicted = sum(
        len(sample["prediction"]["x_axis_tick"])
        + len(sample["prediction"]["y_axis_tick"])
        for sample in samples
    )
    uncertain_count = sum(
        len(sample["prediction"]["uncertain_values"]) for sample in samples
    )
    numeric_expected = (
        x_metrics["numerical_tick_accuracy"]["expected_count"]
        + y_metrics["numerical_tick_accuracy"]["expected_count"]
    )
    numeric_predicted = (
        x_metrics["numerical_tick_accuracy"]["predicted_count"]
        + y_metrics["numerical_tick_accuracy"]["predicted_count"]
    )
    numeric_matched = (
        x_metrics["numerical_tick_accuracy"]["matched_count"]
        + y_metrics["numerical_tick_accuracy"]["matched_count"]
    )
    categorical_expected = (
        x_metrics["categorical_tick_accuracy"]["expected_count"]
        + y_metrics["categorical_tick_accuracy"]["expected_count"]
    )
    categorical_predicted = (
        x_metrics["categorical_tick_accuracy"]["predicted_count"]
        + y_metrics["categorical_tick_accuracy"]["predicted_count"]
    )
    categorical_matched = (
        x_metrics["categorical_tick_accuracy"]["matched_count"]
        + y_metrics["categorical_tick_accuracy"]["matched_count"]
    )
    character_errors = (
        x_metrics["character_error_count"] + y_metrics["character_error_count"]
    )
    expected_characters = (
        x_metrics["expected_character_count"]
        + y_metrics["expected_character_count"]
    )
    return {
        "x_axis": x_metrics,
        "y_axis": y_metrics,
        "numerical_tick_accuracy": {
            "expected_count": numeric_expected,
            "predicted_count": numeric_predicted,
            "matched_count": numeric_matched,
            "accuracy": numeric_matched / max(numeric_expected, numeric_predicted)
            if max(numeric_expected, numeric_predicted)
            else 0.0,
        },
        "categorical_tick_accuracy": {
            "expected_count": categorical_expected,
            "predicted_count": categorical_predicted,
            "matched_count": categorical_matched,
            "accuracy": categorical_matched / max(categorical_expected, categorical_predicted)
            if max(categorical_expected, categorical_predicted)
            else 0.0,
        },
        "character_error_rate": character_errors / expected_characters
        if expected_characters
        else 0.0,
        "overall_uncertain_prediction_rate": uncertain_count
        / max(1, all_predicted + uncertain_count),
        "chart_count": len(samples),
        "tick_count": all_expected,
    }


def _read_dataset(path: Path, images_directory: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", [payload]) if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise ValueError("Evaluation JSON must contain an image object or an images array")
    records: list[dict[str, Any]] = []
    for image_entry in images:
        if not isinstance(image_entry, dict) or not isinstance(image_entry.get("charts"), list):
            raise ValueError("Each evaluation image must include a charts array")
        image_name = image_entry.get("image_name")
        if not isinstance(image_name, str) or not image_name:
            raise ValueError("Each evaluation image requires image_name")
        image_path = Path(image_entry.get("image_path", image_name))
        if not image_path.is_absolute():
            image_path = images_directory / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Evaluation image not found: {image_path}")
        for chart in image_entry["charts"]:
            if not isinstance(chart, dict) or not isinstance(chart.get("chart_id"), int):
                raise ValueError(f"Invalid chart entry for {image_name}")
            record: dict[str, Any] = {
                "image_name": image_name,
                "image_path": image_path,
                "chart_id": chart["chart_id"],
            }
            for axis in ("x", "y"):
                values = chart.get(f"{axis}_axis_tick")
                if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise ValueError(
                        f"{image_name} chart {chart['chart_id']} requires "
                        f"{axis}_axis_tick as a list of strings"
                    )
                record[f"{axis}_axis_tick"] = values
            records.append(record)
    if not records:
        raise ValueError(f"No chart labels found in evaluation dataset: {path}")
    return records


def evaluate_prompt_modes(
    dataset_path: Path,
    images_directory: Path,
    model_path: Path,
    output_path: Path,
    examples_manifest: Path | None = None,
    ollama_url: str = "https://aisandbox.mpslimited.com/ollama",
    model: str = "qwen3.5:9b",
    device: str | None = None,
    crop_padding: float = 0.12,
    modes: tuple[str, ...] = _ALL_MODES,
    detector: Any | None = None,
    extractor: Any | None = None,
    prompt_manager: PromptManager | None = None,
    temperature: float = 0.0,
    num_predict: int = 512,
    timeout: float = 180.0,
) -> dict[str, Any]:
    records = _read_dataset(dataset_path, images_directory)
    unique_image_count = len({record["image_name"] for record in records})
    if unique_image_count < 50:
        LOGGER.warning(
            "Evaluation dataset has %d unique images and %d chart labels; use at "
            "least 50 manually verified scientific chart images for an initial experiment.",
            unique_image_count,
            len(records),
        )
    prompt_manager = prompt_manager or PromptManager(examples_manifest=examples_manifest)
    for mode in modes:
        prompt_manager.validate_mode(mode)
    image_paths = list(dict.fromkeys(record["image_path"] for record in records))
    detector = detector or ChartDetector(model_path, device=device)
    extractor = extractor or VLMExtractor(
        model=model,
        base_url=ollama_url,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
        retries=4,  # Increased retries for evaluation
        retry_backoff=2.0,
    )
    predictions: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in modes
    }
    chart_cache: dict[Path, tuple[np.ndarray, list[Any]]] = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read evaluation image: {image_path}")
        chart_cache[image_path] = (image, detector.detect(image))

    plot_area_detector = PlotAreaDetector()
    axis_preprocessor = AxisPreprocessor()
    for record in records:
        image, detections = chart_cache[record["image_path"]]
        chart_index = record["chart_id"] - 1
        if chart_index < 0 or chart_index >= len(detections):
            raise ValueError(
                f"Chart ID {record['chart_id']} is not detected in {record['image_name']}"
            )
        box = detections[chart_index]
        bounds = padded_box(image, box, crop_padding)
        chart_crop = crop_from_bounds(image, bounds)
        plot = plot_area_detector.detect(chart_crop)
        axis_crops = axis_preprocessor.prepare(chart_crop, plot)
        for mode in modes:
            request = prompt_manager.build(
                mode,
                chart_crop,
                axis_crops.x_axis,
                axis_crops.y_axis,
                axis_crops.y_axis_rotated,
            )
            prediction = extractor.extract(request)
            predictions[mode].append(
                {
                    "image_name": record["image_name"],
                    "chart_id": record["chart_id"],
                    "x_axis_tick": record["x_axis_tick"],
                    "y_axis_tick": record["y_axis_tick"],
                    "prediction": prediction,
                }
            )
        LOGGER.info("Evaluated %s chart %d", record["image_name"], record["chart_id"])

    configurations = {
        mode: {
            "metrics": score_mode(samples),
            "results": [
                {
                    "image_name": sample["image_name"],
                    "chart_id": sample["chart_id"],
                    "expected": {
                        "x_axis_tick": sample["x_axis_tick"],
                        "y_axis_tick": sample["y_axis_tick"],
                    },
                    "predicted": sample["prediction"],
                }
                for sample in samples
            ],
        }
        for mode, samples in predictions.items()
    }
    report = {
        "model": model,
        "dataset": str(dataset_path.resolve()),
        "manually_labeled_image_count": unique_image_count,
        "manually_labeled_chart_count": len(records),
        "minimum_recommended_image_count": 50,
        "meets_recommended_initial_dataset_size": unique_image_count >= 50,
        "same_test_charts_for_all_modes": True,
        "configurations": configurations,
        "selection_note": (
            "Compare measured metrics on this same held-out dataset. No configuration "
            "is selected as better automatically."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote prompt-mode comparison to %s", output_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=project_root / "input")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / "models" / "best_chart_detect (5).pt",
    )
    parser.add_argument("--output", type=Path, default=project_root / "output" / "vlm_prompt_evaluation.json")
    parser.add_argument("--examples", type=Path, default=project_root / "dataset" / "few_shot_examples" / "examples.json")
    parser.add_argument(
        "--ollama-url",
        default="https://aisandbox.mpslimited.com/ollama",
    )
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--ollama-temperature", type=float, default=0.0)
    parser.add_argument("--ollama-num-predict", type=int, default=512)
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=180.0,
        help="Ollama response read timeout in seconds",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--modes", nargs="+", choices=PROMPT_MODES, default=list(_ALL_MODES))
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    evaluate_prompt_modes(
        args.dataset,
        args.images,
        args.model_path,
        args.output,
        args.examples,
        args.ollama_url,
        args.ollama_model,
        args.device,
        args.crop_padding,
        tuple(args.modes),
        temperature=args.ollama_temperature,
        num_predict=args.ollama_num_predict,
        timeout=args.ollama_timeout,
    )


if __name__ == "__main__":
    main()
