"""Plot-area localization within a detected chart crop."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlotArea:
    """Plot bounds in chart-crop pixel coordinates."""

    left: float
    top: float
    right: float
    bottom: float
    horizontal_axis_detected: bool
    vertical_axis_detected: bool
    confidence: float
    method: str

    @property
    def x_axis_y(self) -> float:
        return self.bottom

    @property
    def y_axis_x(self) -> float:
        return self.left

    def as_dict(self) -> dict[str, int | float | str | bool]:
        bounds: dict[str, int | float | str | bool] = {
            "left": round(self.left, 2),
            "top": round(self.top, 2),
            "right": round(self.right, 2),
            "bottom": round(self.bottom, 2),
        }
        return bounds


def _line_candidates(
    mask: np.ndarray,
    horizontal: bool,
    min_length: int,
) -> list[tuple[float, float, float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float, float]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        length = width if horizontal else height
        if length < min_length:
            continue
        if horizontal and height <= max(8, mask.shape[0] * 0.025):
            candidates.append((float(y + height / 2), float(x), float(x + width), float(length)))
        elif not horizontal and width <= max(8, mask.shape[1] * 0.025):
            candidates.append((float(x + width / 2), float(y), float(y + height), float(length)))
    return candidates


def _hough_candidates(
    edges: np.ndarray,
    horizontal: bool,
    min_length: int,
) -> list[tuple[float, float, float, float]]:
    height, width = edges.shape[:2]
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(15, min_length // 3),
        minLineLength=min_length,
        maxLineGap=max(4, min(height, width) // 40),
    )
    candidates: list[tuple[float, float, float, float]] = []
    if lines is None:
        return candidates
    for line in lines[:, 0]:
        x1, y1, x2, y2 = (int(value) for value in line)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if horizontal and dx >= min_length and dy <= max(4, dx * 0.035):
            candidates.append(
                (float((y1 + y2) / 2), float(min(x1, x2)), float(max(x1, x2)), float(dx))
            )
        elif not horizontal and dy >= min_length and dx <= max(4, dy * 0.035):
            candidates.append(
                (float((x1 + x2) / 2), float(min(y1, y2)), float(max(y1, y2)), float(dy))
            )
    return candidates


def _deduplicate_lines(
    candidates: list[tuple[float, float, float, float]],
    coordinate_tolerance: float,
) -> list[tuple[float, float, float, float]]:
    result: list[tuple[float, float, float, float]] = []
    for candidate in sorted(candidates, key=lambda line: line[3], reverse=True):
        if not any(abs(candidate[0] - accepted[0]) <= coordinate_tolerance for accepted in result):
            result.append(candidate)
    return result


class PlotAreaDetector:
    """Find plot boundaries from axis lines or a manual region."""

    def detect(
        self,
        image: np.ndarray,
        manual_region: tuple[float, float, float, float] | None = None,
    ) -> PlotArea:
        height, width = image.shape[:2]
        if width < 2 or height < 2:
            raise ValueError(f"Chart crop is too small to locate plot area: {width}x{height}")

        if manual_region is not None:
            left, top, right, bottom = manual_region
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("Manual plot region must be normalized x1,y1,x2,y2 in [0, 1]")
            return PlotArea(
                left * width,
                top * height,
                right * width,
                bottom * height,
                True,
                True,
                1.0,
                "manual",
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 45, 150)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            max(15, (min(width, height) // 8) | 1),
            9,
        )
        min_horizontal = max(20, int(width * 0.32))
        min_vertical = max(20, int(height * 0.32))
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (min_horizontal, 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, min_vertical)
        )
        horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        horizontal = _deduplicate_lines(
            _line_candidates(horizontal_mask, True, min_horizontal)
            + _hough_candidates(edges, True, min_horizontal),
            max(3, height * 0.025),
        )
        vertical = _deduplicate_lines(
            _line_candidates(vertical_mask, False, min_vertical)
            + _hough_candidates(edges, False, min_vertical),
            max(3, width * 0.025),
        )
        horizontal = [
            line for line in horizontal if height * 0.35 <= line[0] <= height * 0.95
        ]
        vertical = [
            line for line in vertical if width * 0.03 <= line[0] <= width * 0.60
        ]

        x_axis = max(horizontal, key=lambda line: (line[0], line[3]), default=None)
        y_axis = min(vertical, key=lambda line: (line[0], -line[3]), default=None)
        horizontal_detected = x_axis is not None
        vertical_detected = y_axis is not None
        if x_axis is not None:
            bottom = x_axis[0]
            right = x_axis[2]
        else:
            bottom = height * 0.82
            right = width * 0.95
        if y_axis is not None:
            left = y_axis[0]
            top = y_axis[1]
        else:
            left = width * 0.15
            top = height * 0.10

        left = float(np.clip(left, 0, width - 1))
        top = float(np.clip(top, 0, height - 1))
        right = float(np.clip(right, left + 1, width))
        bottom = float(np.clip(bottom, top + 1, height))
        methods = [
            "line" if x_axis is not None else "fallback",
            "line" if y_axis is not None else "fallback",
        ]
        detected_count = int(horizontal_detected) + int(vertical_detected)
        confidence = (0.9 if horizontal_detected else 0.2)
        confidence += 0.9 if vertical_detected else 0.2
        confidence /= 2
        result = PlotArea(
            left,
            top,
            right,
            bottom,
            horizontal_detected,
            vertical_detected,
            confidence,
            "+".join(methods),
        )
        LOGGER.debug(
            "Plot area (%s, %s, %s, %s), method=%s, axes=%d/2",
            round(left),
            round(top),
            round(right),
            round(bottom),
            result.method,
            detected_count,
        )
        return result
