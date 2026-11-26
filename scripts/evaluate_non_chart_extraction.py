"""Score saved non-chart JSON results against manually labeled ground truth."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _normalized_text(values: list[str]) -> Counter[str]:
    return Counter(re.sub(r"\s+", " ", value).strip().casefold() for value in values)


def _text_counts(expected: list[str], predicted: list[str]) -> tuple[int, int, int]:
    expected_counts = _normalized_text(expected)
    predicted_counts = _normalized_text(predicted)
    matched = sum((expected_counts & predicted_counts).values())
    return matched, sum(predicted_counts.values()), sum(expected_counts.values())


def _text_count(values: list[str]) -> int:
    return sum(_normalized_text(values).values())


def _box(value: Any, context: str) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        coordinates = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    elif isinstance(value, list) and len(value) == 4:
        coordinates = value
    else:
        raise ValueError(f"{context}: bounding_box must be x1,y1,x2,y2")
    if any(
        isinstance(coordinate, bool)
        or not isinstance(coordinate, (int, float))
        or not math.isfinite(coordinate)
        for coordinate in coordinates
    ):
        raise ValueError(f"{context}: bounding-box coordinates must be finite numbers")
    x1, y1, x2, y2 = (float(coordinate) for coordinate in coordinates)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{context}: bounding box must have positive area")
    return x1, y1, x2, y2


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _validate_detections(
    detections: Any,
    context: str,
    *,
    require_text: bool,
    name_error: str,
) -> list[dict[str, Any]]:
    if not isinstance(detections, list):
        raise ValueError(f"{context}: detections must be an array")
    validated: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        item_context = f"{context}:detections[{index}]"
        if not isinstance(detection, dict):
            raise ValueError(f"{item_context} must be an object")
        name = detection.get("class_name") or detection.get("object_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{item_context}: {name_error}")
        extracted = detection.get("extracted_text")
        if not require_text and "extracted_text" not in detection:
            extracted = []
        if not isinstance(extracted, list) or not all(
            isinstance(value, str) for value in extracted
        ):
            raise ValueError(f"{item_context}: extracted_text must be a string array")
        validated.append(
            {
                "name": name.strip(),
                "box": _box(detection.get("bounding_box"), item_context),
                "text": extracted,
            }
        )
    return validated


def _validate_prediction(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("image_type") != "non_chart":
        raise ValueError(f"{path}: expected an object with image_type='non_chart'")
    if not isinstance(payload.get("category"), str):
        raise ValueError(f"{path}: category must be a string")
    if not isinstance(payload.get("extracted_text"), list) or not all(
        isinstance(value, str) for value in payload["extracted_text"]
    ):
        raise ValueError(f"{path}: extracted_text must be a string array")
    return {
        "category": payload["category"],
        "text": payload["extracted_text"],
        "detections": _validate_detections(
            payload.get("detections"),
            str(path),
            require_text=True,
            name_error="class_name/object_name must be a non-empty string",
        ),
    }


def _validate_ground_truth_entry(entry: Any, context: str) -> dict[str, Any]:
    if not isinstance(entry, dict) or not isinstance(entry.get("image_name"), str):
        raise ValueError(f"{context}: image entry requires image_name")
    category = entry.get("category")
    if not isinstance(category, str):
        raise ValueError(f"{context}: category must be a string")
    text = entry.get("extracted_text", [])
    if not isinstance(text, list) or not all(isinstance(value, str) for value in text):
        raise ValueError(f"{context}: extracted_text must be a string array")
    return {
        "image_name": entry["image_name"],
        "category": category,
        "text": text,
        "detections": _validate_detections(
            entry.get("detections"),
            context,
            require_text=False,
            name_error="class_name/object_name is required",
        ),
    }


def _match_regions(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = sorted(
        (
            (_iou(expected_index["box"], predicted_index["box"]), gt_index, pred_index)
            for gt_index, expected_index in enumerate(expected)
            for pred_index, predicted_index in enumerate(predicted)
        ),
        reverse=True,
    )
    matches: list[tuple[int, int, float]] = []
    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    for overlap, gt_index, pred_index in candidates:
        if (
            overlap < iou_threshold
            or gt_index in used_expected
            or pred_index in used_predicted
        ):
            continue
        matches.append((gt_index, pred_index, overlap))
        used_expected.add(gt_index)
        used_predicted.add(pred_index)
    return matches, used_expected, used_predicted


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    ground_truth_path: Path,
    predictions_directory: Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be in [0, 1]")
    payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    entries = payload.get("images", [payload]) if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("Ground truth must contain one image object or a non-empty images array")
    ground_truth = [
        _validate_ground_truth_entry(entry, f"images[{index}]")
        for index, entry in enumerate(entries)
    ]
    image_names = [entry["image_name"] for entry in ground_truth]
    if len(set(image_names)) != len(image_names):
        raise ValueError("Ground-truth image_name values must be unique")
    image_stems = [Path(name).stem for name in image_names]
    if len(set(image_stems)) != len(image_stems):
        raise ValueError(
            "Ground-truth image_name values must have unique filename stems"
        )

    json_valid_count = 0
    category_correct = 0
    expected_count = predicted_count = region_matches_count = 0
    object_correct = 0
    object_tp = object_fp = object_fn = 0
    text_matched = text_predicted = text_expected = 0
    total_iou = 0.0
    image_results: list[dict[str, Any]] = []

    for truth in ground_truth:
        prediction_path = predictions_directory / f"{Path(truth['image_name']).stem}.json"
        predicted: dict[str, Any] | None = None
        error: str | None = None
        try:
            predicted = _validate_prediction(
                json.loads(prediction_path.read_text(encoding="utf-8")),
                prediction_path,
            )
            json_valid_count += 1
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            error = str(exc)
        if predicted is None:
            image_results.append(
                {
                    "image_name": truth["image_name"],
                    "json_valid": False,
                    "error": error,
                    "region_association": {
                        "expected_regions": len(truth["detections"]),
                        "predicted_regions": 0,
                        "matched_regions": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "accuracy": 0.0,
                        "mean_iou": 0.0,
                    },
                }
            )
            expected_count += len(truth["detections"])
            object_fn += len(truth["detections"])
            for detection in truth["detections"]:
                text_expected += _text_count(detection["text"])
            text_expected += _text_count(truth["text"])
            continue

        category_match = truth["category"].casefold() == predicted["category"].casefold()
        category_correct += int(category_match)
        matches, used_gt, used_pred = _match_regions(
            truth["detections"], predicted["detections"], iou_threshold
        )
        gt_count = len(truth["detections"])
        pred_count = len(predicted["detections"])
        expected_count += gt_count
        predicted_count += pred_count
        region_matches_count += len(matches)
        total_iou += sum(overlap for _, _, overlap in matches)
        class_matches = 0
        for gt_index, pred_index, _ in matches:
            class_matches += int(
                truth["detections"][gt_index]["name"].casefold()
                == predicted["detections"][pred_index]["name"].casefold()
            )
            matched_text, predicted_text, expected_text = _text_counts(
                truth["detections"][gt_index]["text"],
                predicted["detections"][pred_index]["text"],
            )
            text_matched += matched_text
            text_predicted += predicted_text
            text_expected += expected_text
        for gt_index, truth_detection in enumerate(truth["detections"]):
            if gt_index not in used_gt:
                text_expected += _text_count(truth_detection["text"])
        for pred_index, predicted_detection in enumerate(predicted["detections"]):
            if pred_index not in used_pred:
                text_predicted += _text_count(predicted_detection["text"])
        scene_matched, scene_predicted, scene_expected = _text_counts(
            truth["text"], predicted["text"]
        )
        text_matched += scene_matched
        text_predicted += scene_predicted
        text_expected += scene_expected
        object_correct += class_matches
        object_tp += class_matches
        object_fp += pred_count - class_matches
        object_fn += gt_count - class_matches
        image_results.append(
            {
                "image_name": truth["image_name"],
                "json_valid": True,
                "category_match": category_match,
                "region_association": {
                    "expected_regions": gt_count,
                    "predicted_regions": pred_count,
                    "matched_regions": len(matches),
                    "precision": _ratio(len(matches), pred_count),
                    "recall": _ratio(len(matches), gt_count),
                    "accuracy": _ratio(len(matches), max(gt_count, pred_count)),
                    "mean_iou": _ratio(
                        sum(overlap for _, _, overlap in matches), len(matches)
                    ),
                },
                "object_identification": {
                    "correct_class_matches": class_matches,
                    "accuracy_on_matched_regions": _ratio(class_matches, len(matches)),
                },
            }
        )

    object_precision = _ratio(object_tp, object_tp + object_fp)
    object_recall = _ratio(object_tp, object_tp + object_fn)
    text_precision = _ratio(text_matched, text_predicted)
    text_recall = _ratio(text_matched, text_expected)
    return {
        "task": "non_chart_extraction",
        "ground_truth": str(ground_truth_path.resolve()),
        "predictions_directory": str(predictions_directory.resolve()),
        "iou_threshold": iou_threshold,
        "image_count": len(ground_truth),
        "json_validity": {
            "valid_files": json_valid_count,
            "expected_files": len(ground_truth),
            "accuracy": _ratio(json_valid_count, len(ground_truth)),
        },
        "category_classification": {
            "correct": category_correct,
            "count": len(ground_truth),
            "accuracy": _ratio(category_correct, len(ground_truth)),
        },
        "object_identification": {
            "precision": object_precision,
            "recall": object_recall,
            "f1": _ratio(2 * object_precision * object_recall, object_precision + object_recall),
            "correct_class_matches": object_correct,
            "expected_regions": expected_count,
            "predicted_regions": predicted_count,
            "accuracy_on_iou_matched_regions": _ratio(object_correct, region_matches_count),
        },
        "region_association": {
            "matched_regions": region_matches_count,
            "expected_regions": expected_count,
            "predicted_regions": predicted_count,
            "precision": _ratio(region_matches_count, predicted_count),
            "recall": _ratio(region_matches_count, expected_count),
            "accuracy": _ratio(region_matches_count, max(expected_count, predicted_count)),
            "mean_iou_of_matches": _ratio(total_iou, region_matches_count),
        },
        "text_extraction": {
            "matched_strings": text_matched,
            "predicted_strings": text_predicted,
            "expected_strings": text_expected,
            "precision": text_precision,
            "recall": text_recall,
            "f1": _ratio(2 * text_precision * text_recall, text_precision + text_recall),
        },
        "images": image_results,
        "limitations": (
            "Region matches use greedy one-to-one IoU matching. Text matches are "
            "case-insensitive exact strings after whitespace normalization. "
            "These metrics describe only the supplied labeled images."
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=project_root / "output")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output" / "non_chart_extraction_metrics.json",
    )
    args = parser.parse_args()
    report = evaluate(args.ground_truth, args.predictions, args.iou_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved non-chart extraction evaluation to {args.output.resolve()}")


if __name__ == "__main__":
    main()
