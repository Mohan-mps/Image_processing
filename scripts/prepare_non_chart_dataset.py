"""Deterministically split paired images and YOLO labels into train/val/test."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

from .dataset_utils import (
    IMAGE_EXTENSIONS,
    load_dataset_config,
    validate_label_file,
    validate_dataset,
)


def _paired_inputs(images_dir: Path, labels_dir: Path, class_count: int) -> list[tuple[Path, Path]]:
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Source image and label directories must both exist")
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len({path.stem for path in images}) != len(images):
        raise ValueError("Source images must have unique filename stems")
    pairs: list[tuple[Path, Path]] = []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing YOLO annotation for {image.name}: {label}")
        validate_label_file(label, class_count)
        pairs.append((image, label))
    if not pairs:
        raise ValueError(f"No supported source images found in {images_dir}")
    orphan_labels = sorted(
        path.name
        for path in labels_dir.glob("*.txt")
        if path.stem not in {image.stem for image, _ in pairs}
    )
    if orphan_labels:
        raise ValueError(f"Labels have no matching source image: {orphan_labels}")
    return pairs


def _assert_output_is_empty(output_dir: Path) -> None:
    if output_dir.exists():
        existing = [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.name not in {".gitkeep", "README.md"}
        ]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite an existing dataset at {output_dir}; "
                "choose an empty --output directory"
            )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Flat source images directory.")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO .txt labels directory.")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "config" / "non_chart_yolo.yaml",
        help="Dataset YAML whose names define class IDs.",
    )
    parser.add_argument("--output", type=Path, help="Dataset root (defaults to YAML path).")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        parser.error("--val-fraction and --test-fraction must both be positive")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("Validation and test fractions must sum to less than 1")

    data_config, dataset_root, class_names = load_dataset_config(args.data)
    output_root = (args.output or dataset_root).resolve()
    pairs = _paired_inputs(args.images, args.labels, len(class_names))
    if len(pairs) < 3:
        raise ValueError("At least three annotated images are needed for train/val/test")
    _assert_output_is_empty(output_root)

    random.Random(args.seed).shuffle(pairs)
    validation_count = max(1, round(len(pairs) * args.val_fraction))
    test_count = max(1, round(len(pairs) * args.test_fraction))
    train_count = len(pairs) - validation_count - test_count
    if train_count < 1:
        raise ValueError(
            "Split fractions leave no training examples; provide more images "
            "or reduce val/test fractions"
        )
    assignments = {
        "train": pairs[:train_count],
        "val": pairs[train_count : train_count + validation_count],
        "test": pairs[train_count + validation_count :],
    }
    for split, split_pairs in assignments.items():
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, label in split_pairs:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / label.name)

    output_data_yaml = args.data
    if output_root != dataset_root:
        output_data_yaml = output_root / "dataset.yaml"
        output_data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": data_config["names"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    report = validate_dataset(output_data_yaml)
    print(
        "Dataset split complete: "
        + ", ".join(
            f"{split}={len(assignments[split])}"
            for split in ("train", "val", "test")
        )
    )
    print(f"Classes: {class_names}")
    print(f"Validated dataset root: {report['dataset_root']}")
    print(f"Dataset YAML: {output_data_yaml.resolve()}")


if __name__ == "__main__":
    main()
