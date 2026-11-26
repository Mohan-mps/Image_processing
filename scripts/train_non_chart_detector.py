"""Train an independent YOLO detector for custom non-chart object classes."""

from __future__ import annotations

import argparse
import shutil
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
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ultralytics device such as cpu or 0.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=project_root / "models" / "non_chart" / "runs",
    )
    parser.add_argument("--run-name", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "models" / "non_chart" / "best_non_chart.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs, imgsz, and batch must be positive; workers cannot be negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite trained weights: {args.output}. "
            "Use --overwrite explicitly or select another --output."
        )
    validate_dataset(args.data)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    model = YOLO(args.base_model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.runs_dir),
        name=args.run_name,
        exist_ok=False,
    )
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Ultralytics training completed without a run directory")
    best_weights = Path(save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Training did not produce the expected best checkpoint: {best_weights}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.output)
    print(f"Saved best non-chart detector weights to {args.output.resolve()}")


if __name__ == "__main__":
    main()
