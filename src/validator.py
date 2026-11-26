"""Validation and evaluation helpers for Qwen axis-tick output."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_SUPER_DIGITS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
        "−": "-",
        "–": "-",
        "—": "-",
    }
)
_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:(?:[eE][+-]?\d+)|(?:\^[+-]?\d+))?$"
)
_NUMERIC_LIKE_PATTERN = re.compile(
    r"^[0-9A-Za-zOIl|.,+\-−–—eExX×·^~()\[\]\s]+$"
)
_MULTIPLIED_POWER_PATTERN = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[xX×·]\s*10\s*\^?\s*([+-]?\d+)$"
)
_POWER_OF_TEN_PATTERN = re.compile(r"^10([+-]\d+)$")


def normalize_tick_value(text: str) -> str | None:
    """Normalize minus/superscript glyphs and accept numeric tick formats."""
    value = "".join(text.translate(_SUPER_DIGITS).split())
    if not value:
        return None
    multiplied_power = _MULTIPLIED_POWER_PATTERN.fullmatch(value)
    if multiplied_power:
        value = f"{multiplied_power.group(1)}e{multiplied_power.group(2)}"
    else:
        power_of_ten = _POWER_OF_TEN_PATTERN.fullmatch(value)
        if power_of_ten:
            value = f"10^{power_of_ten.group(1)}"
    if _NUMBER_PATTERN.fullmatch(value):
        return value
    return None


def validate_vlm_result(payload: Any) -> dict[str, Any]:
    """Validate graph classification and tick strings without inventing text."""
    if not isinstance(payload, dict):
        raise ValueError("VLM output must be a JSON object")
    is_graph = payload.get("is_graph")
    if not isinstance(is_graph, bool):
        raise ValueError("VLM field 'is_graph' must be a boolean")
    normalized: dict[str, Any] = {"is_graph": is_graph}
    uncertain: list[str] = []
    for field in ("x_axis_tick", "y_axis_tick", "uncertain_values"):
        values = payload.get(field)
        if values is None and field == "uncertain_values":
            values = []
        if not isinstance(values, list):
            raise ValueError(f"VLM field {field!r} must be an array")
        cleaned: list[str] = []
        for value in values:
            if isinstance(value, dict):
                value = value.get("value")
            if not isinstance(value, str):
                raise ValueError(f"VLM field {field!r} must contain strings")
            value = value.strip()
            if value and value not in cleaned:
                cleaned.append(value)
        if field == "uncertain_values":
            uncertain.extend(cleaned)
        else:
            normalized[field] = cleaned

    if not is_graph:
        normalized["x_axis_tick"] = []
        normalized["y_axis_tick"] = []

    for field in ("x_axis_tick", "y_axis_tick"):
        for value in normalized[field]:
            if looks_numeric_like(value):
                # Safely handle None return from normalize_tick_value
                normalized_value = normalize_tick_value(value)
                if normalized_value is None and value not in uncertain:
                    uncertain.append(value)
    normalized["uncertain_values"] = uncertain
    return normalized


def looks_numeric_like(text: str) -> bool:
    """Identify numeric-looking strings that may be corrupted tick labels."""
    value = text.strip()
    return (
        bool(value)
        and _NUMERIC_LIKE_PATTERN.fullmatch(value) is not None
        and (any(character.isdigit() for character in value) or any(
            character in "OIl|" for character in value
        ))
        and sum(character.isalpha() for character in value) <= 2
    )


def evaluate_tick_values(
    expected: list[str],
    predicted: list[str],
) -> dict[str, int | float | bool]:
    """Compare ordered tick strings against manually verified values."""
    expected_values = [value.strip() for value in expected]
    predicted_values = [value.strip() for value in predicted]
    matched = sum((Counter(expected_values) & Counter(predicted_values)).values())
    denominator = max(len(expected_values), len(predicted_values))
    precision = matched / len(predicted_values) if predicted_values else 0.0
    recall = matched / len(expected_values) if expected_values else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "expected_count": len(expected_values),
        "predicted_count": len(predicted_values),
        "matched_count": matched,
        "accuracy": matched / denominator if denominator else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_sequence_match": expected_values == predicted_values,
    }
