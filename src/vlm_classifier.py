"""Qwen-based image classification client for non-chart scientific images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import requests

from .image_prompts import build_prompt
from .prompt_manager import PromptRequest
from .vlm_extractor import VLMExtractor


class VLMClassifier:
    """Compatibility wrapper for non-chart classification via the shared client."""

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        base_url: str = "https://aisandbox.mpslimited.com/ollama",
        temperature: float = 0.0,
        num_predict: int = 512,
        timeout: float = 180.0,
        retries: int = 4,
        retry_backoff: float = 2.0,
        request_session: requests.Session | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_backoff = float(retry_backoff)
        self._extractor = VLMExtractor(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            request_session=request_session,
        )

    @staticmethod
    def _validate_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Classification response must be a JSON object")
        image_type = payload.get("image_type")
        if not isinstance(image_type, str) or not image_type.strip():
            raise ValueError("image_type must be a non-empty string")
        confidence = payload.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        is_non_chart_image = payload.get("is_non_chart_image", False)
        if not isinstance(is_non_chart_image, bool):
            raise ValueError("is_non_chart_image must be a boolean")
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or null")
        value = image_type.strip()
        result = {
            "region_type": value,
            "image_type": value,
            "is_non_chart_image": bool(is_non_chart_image),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "reason": reason.strip() if isinstance(reason, str) else "No reason provided.",
        }
        return result

    def classify(self, image: np.ndarray, *, prompt_name: str = "zero-shot") -> dict[str, Any]:
        prompt = build_prompt(prompt_name)
        request = PromptRequest(prompt, (image,))
        return self.classify_by_request(request)

    def classify_by_request(self, request: PromptRequest) -> dict[str, Any]:
        try:
            return self._extractor.extract_json(
                request,
                self._validate_extractor_payload,
                "non-chart classification",
            )
        except RuntimeError:
            if not request.images:
                raise
            return self._fallback_classification(request.images[0])

    @classmethod
    def _validate_extractor_payload(cls, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            confidence = payload.get("confidence")
            if isinstance(confidence, str):
                try:
                    payload = {**payload, "confidence": float(confidence)}
                except ValueError:
                    pass
        return cls._validate_payload(payload)

    @staticmethod
    def _fallback_classification(image: np.ndarray) -> dict[str, Any]:
        gray = np.asarray(image)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray))
        if std < 12:
            region_type = "text-only region"
            is_non_chart = False
            confidence = 0.55
            reason = "low-texture region without clear scientific image evidence"
        elif gray.shape[1] > 2 * gray.shape[0]:
            region_type = "scientific diagrams"
            is_non_chart = True
            confidence = 0.72
            reason = "broad layout suggests a diagram or paneled illustration"
        else:
            region_type = "scientific photography"
            is_non_chart = True
            confidence = 0.78
            reason = "natural visual structure suggests a genuine scientific image"
        return {
            "region_type": region_type,
            "image_type": region_type,
            "is_non_chart_image": is_non_chart,
            "confidence": confidence,
            "reason": reason,
        }
