"""Task-specific and few-shot prompt construction for axis tick extraction."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROMPT_MODES = (
    "zero-shot",
    "task-specific",
    "few-shot",
    "few-shot-enhanced-crops",
)

_ZERO_SHOT = """First decide whether the image contains an actual plotted graph with X and Y axes and plotted data (such as a line, bar, scatter, or heatmap plot). A photograph, whiteboard note, workflow/fishbone diagram, timeline, infographic, table, or panel of specimens is not a graph. If it is not a graph, set is_graph to false and return empty tick arrays. If it is a graph, set is_graph to true and read only its visible X-axis and left-side Y-axis tick labels. Ignore any right-side vertical scale or Z-axis. Preserve every character exactly, including trailing decimal zeros (for example, keep -1.0 as -1.0, not -1). Multi-line tick labels: if two or more lines of text sit directly beneath or beside the same tick mark, they form ONE label — join them with a single space (e.g. "pH 4" over "Bio I" becomes "pH 4 Bio I"). Return one JSON object with boolean is_graph and string arrays x_axis_tick, y_axis_tick, and uncertain_values. Preserve displayed order. Do not infer missing values."""


@dataclass(frozen=True)
class PromptRequest:
    prompt: str
    images: tuple[np.ndarray, ...]


class PromptManager:
    def __init__(
        self,
        prompt_directory: Path | None = None,
        examples_manifest: Path | None = None,
        maximum_examples: int = 3,
    ) -> None:
        if maximum_examples < 0:
            raise ValueError("Maximum few-shot examples cannot be negative")
        project_root = Path(__file__).resolve().parents[1]
        self.prompt_directory = prompt_directory or project_root / "prompts"
        self.examples_manifest = examples_manifest or (
            project_root / "dataset" / "few_shot_examples" / "examples.json"
        )
        self.maximum_examples = maximum_examples
        self._examples_cache: list[dict[str, Any]] | None = None
        self._example_image_cache: dict[Path, np.ndarray] = {}

    def build_axis(self, axis: str, image: np.ndarray) -> PromptRequest:
        """Build a single-axis request using the dedicated X or Y prompt file."""
        if axis not in {"x", "y"}:
            raise ValueError("Axis must be 'x' or 'y'")
        prompt_path = self.prompt_directory / f"{axis}_axis_prompt.txt"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Axis prompt not found: {prompt_path}")
        return PromptRequest(prompt_path.read_text(encoding="utf-8").strip(), (image,))

    def validate_mode(self, mode: str) -> None:
        if mode not in PROMPT_MODES:
            raise ValueError(f"Unknown prompt mode {mode!r}; expected one of {PROMPT_MODES}")
        if mode in {"few-shot", "few-shot-enhanced-crops"} and not self._load_examples():
            raise ValueError(
                f"Few-shot mode requested, but no examples are configured in "
                f"{self.examples_manifest}. Add manually verified image/JSON examples."
            )

    def build(
        self,
        mode: str,
        chart_image: np.ndarray,
        x_axis_image: np.ndarray | None = None,
        y_axis_image: np.ndarray | None = None,
        y_axis_rotated_image: np.ndarray | None = None,
    ) -> PromptRequest:
        self.validate_mode(mode)
        if mode == "zero-shot":
            return PromptRequest(_ZERO_SHOT, (chart_image,))

        prompt = self._combined_prompt()
        images: list[np.ndarray] = []
        if mode in {"few-shot", "few-shot-enhanced-crops"}:
            examples = self._load_examples()
            example_sections: list[str] = []
            for index, example in enumerate(examples[: self.maximum_examples], start=1):
                image_path = Path(example["image"])
                if not image_path.is_absolute():
                    image_path = self.examples_manifest.parent / image_path
                image_path = image_path.resolve()
                example_image = self._example_image_cache.get(image_path)
                if example_image is None:
                    example_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if example_image is None:
                        raise OSError(f"Could not read few-shot example image: {image_path}")
                    self._example_image_cache[image_path] = example_image
                images.append(example_image)
                expected = self._validate_expected(example["expected"], image_path)
                example_sections.append(
                    f"Reference image {index} is followed by its verified answer:\n"
                    f"{json.dumps(expected, ensure_ascii=False)}"
                )
            prompt = (
                "Use the following manually verified reference examples only to learn "
                "the requested JSON format and visual task. Do not copy their values "
                "unless they are visible in the final chart.\n\n"
                + "\n\n".join(example_sections)
                + "\n\nNow process the final chart images in the order specified below.\n\n"
                + prompt
            )

        if mode == "few-shot-enhanced-crops":
            if x_axis_image is None or y_axis_image is None:
                raise ValueError("Enhanced few-shot mode requires X- and Y-axis crops")
            images.extend(
                [
                    chart_image,
                    x_axis_image,
                    y_axis_image,
                    y_axis_rotated_image
                    if y_axis_rotated_image is not None
                    else y_axis_image,
                ]
            )
            prompt += (
                "\nThe final four images are, in order: the full chart, enhanced "
                "X-axis region, enhanced Y-axis region, and a 90-degree-rotated view "
                "of that same Y-axis region. First classify the full chart as a "
                "plotted graph or not; then read X ticks only from the X-axis region "
                "and Y ticks "
                "only from the left-side Y-axis region. The rotated view is supplementary "
                "and does not add ticks. Ignore any right-side vertical scale or Z-axis. "
                "For X-axis ticks: include labels even if they are printed at a diagonal "
                "or angled orientation (e.g. rotated 30-60 degrees), but do NOT include "
                "text that is printed fully vertically (90 degrees), as that belongs to "
                "the Y-axis. "
                "Preserve exact visible strings, including trailing decimal zeros; for "
                "example, do not change -1.0 to -1. Multi-line tick labels: if two or "
                "more lines of text sit beneath or beside the same tick mark, join them "
                "with a single space into one entry (e.g. \"pH 4\" over \"Bio I\" becomes "
                "\"pH 4 Bio I\"). Return boolean is_graph and "
                "string arrays x_axis_tick, y_axis_tick, and uncertain_values."
            )
        else:
            images.append(chart_image)
            if x_axis_image is not None and y_axis_image is not None:
                images.extend(
                    [
                        x_axis_image,
                        y_axis_image,
                        y_axis_rotated_image
                        if y_axis_rotated_image is not None
                        else y_axis_image,
                    ]
                )
                prompt += (
                    "\nFirst verify the final chart is an actual plotted graph with "
                    "X/Y axes and plotted data, not a photograph, whiteboard note, "
                    "workflow/fishbone diagram, timeline, infographic, table, or "
                    "specimen panel. If it is not a graph, set is_graph to false and "
                    "return empty tick arrays. The images after the chart are, in "
                    "order: enhanced X-axis "
                    "region, enhanced left-side Y-axis region, and a rotated view of "
                    "that same left-side Y-axis region. Use the chart only for context. "
                    "Read X ticks only from the X-axis region and Y ticks only from the "
                    "left-side Y-axis region. Do not include any right-side vertical "
                    "scale, secondary axis, or Z-axis values in y_axis_tick. "
                    "For X-axis ticks: include labels even if they are printed at a "
                    "diagonal or angled orientation (e.g. rotated 30-60 degrees, such "
                    "as '50°C/30min'), but do NOT include text that is printed fully "
                    "vertically (90 degrees), as that belongs to the Y-axis. "
                    "Preserve "
                    "exact visible strings, including trailing decimal zeros; for "
                    "example, keep -1.0 as -1.0, not -1. Multi-line tick labels: if "
                    "two or more lines of text sit beneath or beside the same tick "
                    "mark, join them with a single space into one entry (e.g. "
                    "\"pH 4\" over \"Bio I\" becomes \"pH 4 Bio I\"). Return boolean is_graph and "
                    "string arrays x_axis_tick, y_axis_tick, and uncertain_values."
                )
            else:
                prompt += "\nThe final image is the chart to transcribe."
        return PromptRequest(prompt, tuple(images))

    def _combined_prompt(self) -> str:
        path = self.prompt_directory / "combined_axis_prompt.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Combined axis prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _load_examples(self) -> list[dict[str, Any]]:
        if self._examples_cache is not None:
            return self._examples_cache
        if not self.examples_manifest.is_file():
            raise FileNotFoundError(
                f"Few-shot examples index not found: {self.examples_manifest}"
            )
        payload = json.loads(self.examples_manifest.read_text(encoding="utf-8"))
        examples = payload.get("examples")
        if not isinstance(examples, list):
            raise ValueError("Few-shot index must contain an 'examples' array")
        for example in examples:
            if not isinstance(example, dict) or not {"image", "expected"}.issubset(example):
                raise ValueError("Each few-shot example requires image and expected fields")
        self._examples_cache = examples
        return self._examples_cache

    @staticmethod
    def _validate_expected(
        expected: Any, image_path: Path
    ) -> dict[str, Any]:
        if not isinstance(expected, dict):
            raise ValueError(f"Expected output for {image_path} must be a JSON object")
        result: dict[str, Any] = {"is_graph": True}
        for field in ("x_axis_tick", "y_axis_tick"):
            values = expected.get(field)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError(
                    f"Few-shot {field} in {image_path} must be a list of strings"
                )
            result[field] = values
        uncertain = expected.get("uncertain_values", [])
        if not isinstance(uncertain, list) or not all(
            isinstance(value, str) for value in uncertain
        ):
            raise ValueError(
                f"Few-shot uncertain_values in {image_path} must be a list of strings"
            )
        result["uncertain_values"] = uncertain
        return result


def image_to_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode axis/chart image for Ollama")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
