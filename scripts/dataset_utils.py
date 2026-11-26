"""Validation helpers for Ultralytics detection datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_dataset_config(path: Path) -> tuple[dict[str, Any], Path, dict[int, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    root_value = payload.get("path", ".")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError("Dataset YAML 'path' must be a non-empty path string")
    root = Path(root_value)
    if not root.is_absolute():
        root = (path.resolve().parent / root).resolve()

    raw_names = payload.get("names")
    if isinstance(raw_names, list):
        names = {index: name for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(class_id): name for class_id, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Dataset class IDs in 'names' must be integers") from exc
    else:
        raise ValueError("Dataset YAML must define class names as a list or mapping")
    if (
        not names
        or sorted(names) != list(range(len(names)))
        or not all(isinstance(name, str) and name.strip() for name in names.values())
        or len(set(names.values())) != len(names)
    ):
        raise ValueError("Dataset class names must be unique, non-empty, and zero-based")
    return payload, root, names


def split_paths(
    payload: dict[str, Any], root: Path, split: str
) -> tuple[Path, Path]:
    value = payload.get(split)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset YAML must define the {split!r} split as a path")
    image_path = Path(value)
    if not image_path.is_absolute():
        image_path = (root / image_path).resolve()
    if image_path.parent.name != "images":
        raise ValueError(
            f"{split} image path must be under an 'images' directory: {image_path}"
        )
    label_path = image_path.parent.parent / "labels" / image_path.name
    return image_path, label_path


def validate_label_file(path: Path, class_count: int) -> Counter[int]:
    class_counts: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(
                f"{path}:{line_number}: expected 5 YOLO fields "
                "'class x_center y_center width height'"
            )
        try:
            raw_class_id = float(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: fields must be numeric") from exc
        if not raw_class_id.is_integer():
            raise ValueError(f"{path}:{line_number}: class ID must be an integer")
        class_id = int(raw_class_id)
        if class_id < 0 or class_id >= class_count:
            raise ValueError(
                f"{path}:{line_number}: class ID {class_id} is outside "
                f"0..{class_count - 1}"
            )
        center_x, center_y, box_width, box_height = coordinates
        if not (
            0 <= center_x <= 1
            and 0 <= center_y <= 1
            and 0 < box_width <= 1
            and 0 < box_height <= 1
            and center_x - box_width / 2 >= 0
            and center_x + box_width / 2 <= 1
            and center_y - box_height / 2 >= 0
            and center_y + box_height / 2 <= 1
        ):
            raise ValueError(
                f"{path}:{line_number}: normalized box coordinates must "
                "describe a positive box inside [0, 1]"
            )
        class_counts[class_id] += 1
    return class_counts


def validate_split(
    image_path: Path,
    label_path: Path,
    class_count: int,
    *,
    require_images: bool = True,
) -> dict[str, Any]:
    if not image_path.is_dir():
        raise FileNotFoundError(f"Image split directory not found: {image_path}")
    if not label_path.is_dir():
        raise FileNotFoundError(f"Label split directory not found: {label_path}")
    images = sorted(
        path
        for path in image_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if require_images and not images:
        raise ValueError(f"No supported images found in split: {image_path}")
    image_stems = [path.stem for path in images]
    duplicate_stems = sorted(
        stem for stem, count in Counter(image_stems).items() if count > 1
    )
    if duplicate_stems:
        raise ValueError(f"Image stems must be unique in {image_path}: {duplicate_stems}")

    label_files = {
        path.stem: path
        for path in label_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }
    missing_labels = sorted(set(image_stems) - label_files.keys())
    orphan_labels = sorted(label_files.keys() - set(image_stems))
    if missing_labels or orphan_labels:
        raise ValueError(
            f"Image/label mismatch in {image_path}: missing labels={missing_labels}, "
            f"labels without images={orphan_labels}"
        )
    counts: Counter[int] = Counter()
    for label_file in label_files.values():
        counts.update(validate_label_file(label_file, class_count))
    return {
        "image_count": len(images),
        "label_count": len(label_files),
        "object_count": sum(counts.values()),
        "objects_by_class": {str(class_id): counts[class_id] for class_id in range(class_count)},
        "images_directory": str(image_path),
        "labels_directory": str(label_path),
    }


def validate_dataset(data_yaml: Path) -> dict[str, Any]:
    payload, root, names = load_dataset_config(data_yaml)
    splits = {
        split: validate_split(*split_paths(payload, root, split), len(names))
        for split in ("train", "val", "test")
    }
    return {
        "data_yaml": str(data_yaml.resolve()),
        "dataset_root": str(root),
        "class_names": {str(class_id): name for class_id, name in names.items()},
        "splits": splits,
    }
