"""Prompt templates for non-chart scientific image classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NON_CHART_CATEGORIES = [
    "microscopy",
    "medical imaging",
    "histology",
    "gel electrophoresis",
    "scientific photography",
    "SEM / material science",
    "satellite imagery",
    "experimental images",
    "scientific diagrams",
    "other visual content",
    "chart",
    "text-only region",
    "uncertain",
]

PROMPT_A = """You are a scientific figure image classification system.

Analyze the supplied scientific figure or cropped region.
Determine whether the region contains a genuine non-chart scientific visual image.
Identify the image category.

Exclude:
- Charts and graphs.
- Axis labels.
- Legends.
- Titles.
- Captions.
- Panel letters.
- Text-only regions.

Return:
{
  "is_non_chart_image": true,
  "image_type": "microscopy",
  "confidence": 0.95,
  "uncertain": false
}

Do not invent visual information.
"""

PROMPT_B = """You are an expert scientific image detection and classification system.

Identify individual non-chart visual image regions.
Look for visual characteristics such as:
- Cellular structures.
- Microscopic textures.
- Medical scan patterns.
- Experimental sample images.
- Electrophoresis bands.
- Material surface structures.
- Scientific photographs.

Decide if the region is a real scientific image or an artifact such as a chart, legend,
text region, title, or caption. Use the category that best matches the visible evidence.

Return one JSON object with the fields:
{
  "is_non_chart_image": true,
  "image_type": "microscopy",
  "confidence": 0.92,
  "uncertain": false,
  "reason": "clear cellular structures" 
}
"""

PROMPT_C = """You are a conservative scientific visual classifier.

Assess the supplied image region and decide whether it is a genuine non-chart image.
Treat charts, axes, legends, captions, panel letters, and text blocks as non-targets.
Only classify as a scientific image when visible visual evidence is clear.

When uncertain, set:
- is_non_chart_image: false
- image_type: "uncertain"
- confident: false

Return only this JSON schema:
{
  "is_non_chart_image": boolean,
  "image_type": "microscopy|medical imaging|histology|gel electrophoresis|scientific photography|SEM / material science|satellite imagery|experimental images|scientific diagrams|other visual content|chart|text-only region|uncertain",
  "confidence": 0.0,
  "uncertain": true,
  "reason": "brief explanation"
}
"""


@dataclass(frozen=True)
class PromptConfig:
    name: str
    template: str
    description: str


def build_prompt(prompt_name: str, **context: Any) -> str:
    templates = {
        "zero-shot": PROMPT_A,
        "task-specific": PROMPT_B,
        "conservative": PROMPT_C,
    }
    if prompt_name not in templates:
        raise ValueError(f"Unsupported prompt name: {prompt_name!r}")
    prompt = templates[prompt_name]
    for key, value in context.items():
        prompt = prompt.replace("{" + str(key) + "}", str(value))
    return prompt


def get_prompt_configs() -> dict[str, PromptConfig]:
    return {
        "zero-shot": PromptConfig(
            name="zero-shot",
            template=PROMPT_A,
            description="Classify a region without examples.",
        ),
        "task-specific": PromptConfig(
            name="task-specific",
            template=PROMPT_B,
            description="Identify scientific image panels while excluding chart artifacts.",
        ),
        "conservative": PromptConfig(
            name="conservative",
            template=PROMPT_C,
            description="Prefer uncertainty over false positives for ambiguous content.",
        ),
    }
