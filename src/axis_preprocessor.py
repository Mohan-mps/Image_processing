"""Prepare contextual, enhanced X/Y plot-edge crops for vision models."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .plot_area_detector import PlotArea


@dataclass(frozen=True)
class AxisCrops:
    x_axis: np.ndarray
    y_axis: np.ndarray
    y_axis_rotated: np.ndarray
    x_bounds: tuple[int, int, int, int]
    y_bounds: tuple[int, int, int, int]


class AxisPreprocessor:
    """Crop axis neighborhoods with margins, contrast enhancement, and scaling."""

    def __init__(
        self,
        scale: float = 3.0,
        clahe_clip_limit: float = 2.0,
        denoise: bool = False,
    ) -> None:
        if scale < 1:
            raise ValueError("Axis-crop scale must be at least 1")
        if clahe_clip_limit <= 0:
            raise ValueError("CLAHE clip limit must be positive")
        self.scale = scale
        self.clahe_clip_limit = clahe_clip_limit
        self.denoise = denoise

    def prepare(
        self,
        image: np.ndarray,
        plot: PlotArea,
        x_padding_fraction: float = 0.04,
        y_padding_fraction: float = 0.04,
    ) -> AxisCrops:
        if image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("Axis preprocessing requires a non-empty image")
        if x_padding_fraction < 0 or y_padding_fraction < 0:
            raise ValueError("Axis crop padding fractions cannot be negative")
        height, width = image.shape[:2]
        left, top, right, bottom = (
            int(round(value))
            for value in (plot.left, plot.top, plot.right, plot.bottom)
        )
        x_pad = max(4, round(width * x_padding_fraction))
        y_pad = max(4, round(height * y_padding_fraction))
        x_label_height = max(30, round(height * 0.18))
        # The tick numbers sit above the horizontal axis line.  A pure y_pad
        # margin (4 % of crop height, often < 30 px) can be smaller than a
        # two-row label block (tick numbers + axis title), leaving the numbers
        # outside the top edge of the crop.  Use at least 15 % of crop height
        # (or 60 px minimum) upward from bottom so the number row is always
        # captured regardless of how much space the chart data leaves above
        # the axis line.
        x_tick_upward = max(60, round(height * 0.15))
        x_bounds = (
            max(0, left - x_pad),
            max(0, bottom - x_tick_upward),
            min(width, right + x_pad),
            min(height, round(bottom) + x_label_height),
        )
        # Extend right edge well past the axis line so tick labels are included.
        # left marks the axis line itself; tick numbers sit to its left and may
        # be only a few pixels away, so a 2 px margin is insufficient for charts
        # with narrow left margins.  Use at least 40 px or 15 % of image width,
        # whichever is larger, capped at the image boundary.
        y_tick_margin = max(40, round(width * 0.15))
        # The highest Y tick label is commonly placed noticeably above the
        # detected plot top (the top border of the plotting rectangle).  A
        # symmetric 4% margin can therefore crop that label even when the
        # plot boundary itself is detected correctly.  Reserve at least 12%
        # of the chart height above the plot; keep the caller-provided
        # y_padding_fraction when it requests an even larger margin.
        # Additionally, extend all the way to the top edge (y=0) when the plot
        # is detected near the top, to capture axis scale multipliers like "×10⁴"
        # that are often positioned above the axis title.
        y_tick_top_padding = max(60, round(height * 0.12))
        # If plot top is within the upper 30% of the image, extend to y=0
        # to capture scale multipliers and labels that sit above the plot area
        if top < height * 0.3:
            y_top = 0
        else:
            y_top = max(0, top - max(y_pad, y_tick_top_padding))
        y_bounds = (
            0,
            y_top,
            min(width, left + y_tick_margin),
            min(height, bottom + y_pad),
        )
        x_crop = self._enhance(image[x_bounds[1] : x_bounds[3], x_bounds[0] : x_bounds[2]])
        y_crop = self._enhance(image[y_bounds[1] : y_bounds[3], y_bounds[0] : y_bounds[2]])
        if x_crop.size == 0 or y_crop.size == 0:
            raise ValueError("Plot bounds produced an empty axis crop")
        y_rotated = cv2.rotate(y_crop, cv2.ROTATE_90_CLOCKWISE)
        return AxisCrops(x_crop, y_crop, y_rotated, x_bounds, y_bounds)

    def _enhance(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop.copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if self.denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(gray)
        target_width = max(1, round(enhanced.shape[1] * self.scale))
        target_height = max(1, round(enhanced.shape[0] * self.scale))
        return cv2.resize(
            enhanced,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )
