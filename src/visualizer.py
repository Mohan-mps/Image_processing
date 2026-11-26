"""Plot-area visualization for manual review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .plot_area_detector import PlotArea

_PLOT_COLOR = (30, 30, 220)


def save_annotated_chart(
    image: np.ndarray,
    plot: PlotArea,
    destination: Path,
) -> Path:
    """Render the detected plot bounds on a chart crop."""
    annotated = image.copy()
    left, top = round(plot.left), round(plot.top)
    right, bottom = round(plot.right), round(plot.bottom)
    cv2.rectangle(annotated, (left, top), (right, bottom), _PLOT_COLOR, 2)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), annotated):
        raise OSError(f"Could not save annotated chart: {destination}")
    return destination
