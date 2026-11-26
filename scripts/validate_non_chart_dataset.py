"""Validate a non-chart YOLO dataset and its normalized bounding-box labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_utils import validate_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "config" / "non_chart_yolo.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.data)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
