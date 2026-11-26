"""Evaluate non-chart YOLO weights on the labeled validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset_utils import validate_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "config" / "non_chart_yolo.yaml",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=None, help="Ultralytics device such as cpu or 0.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output" / "non_chart_detector_metrics.json",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Non-chart model weights not found: {args.model}")
    dataset_report = validate_dataset(args.data)
    image_count = dataset_report["splits"][args.split]["image_count"]

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        device=args.device,
        plots=True,
    )
    box_metrics = metrics.box
    report: dict[str, Any] = {
        "task": "non_chart_object_detection",
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "image_count": image_count,
        "class_names": getattr(model, "names", {}),
        "metrics": {
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
            "mAP50": float(box_metrics.map50),
            "mAP50_95": float(box_metrics.map),
        },
        "dataset_counts": dataset_report["splits"],
        "interpretation": (
            "Metrics are from this labeled split only; they do not establish "
            "performance on other datasets."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved detector evaluation report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
