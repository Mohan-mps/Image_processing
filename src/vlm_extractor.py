"""Ollama chat API client for Qwen vision-based chart tick extraction."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import requests

from .prompt_manager import PromptRequest, image_to_base64
from .validator import validate_vlm_result

LOGGER = logging.getLogger(__name__)
_MIN_RETRY_TOKEN_INCREASE = 512
_MAX_RETRY_NUM_PREDICT = 4096


class VLMExtractor:
    """Ollama chat API client for Qwen vision-based chart tick extraction.
    
    Thread-Safety Notes:
    - The underlying requests.Session is thread-safe for making concurrent requests
    - Connection pooling allows efficient parallel HTTP requests across threads
    - Multiple VLMExtractor instances can be used concurrently
    - Use context manager (with statement) or call close() to cleanup resources
    """
    
    def __init__(
        self,
        model: str = "qwen3.5:9b",
        base_url: str = "https://aisandbox.mpslimited.com/ollama",
        temperature: float = 0.0,
        num_predict: int = 512,
        timeout: float = 180.0,
        retries: int = 4,
        request_session: requests.Session | None = None,
        retry_backoff: float = 2.0,
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model name cannot be empty")
        if temperature < 0 or num_predict < 1 or timeout <= 0 or retries < 0:
            raise ValueError("Invalid Ollama generation or retry settings")
        if retry_backoff < 0:
            raise ValueError("Retry backoff must be non-negative")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        
        # Use provided session or create one with optimized connection pooling
        if request_session is not None:
            self._session = request_session
        else:
            self._session = requests.Session()
            # Configure adapter for connection pooling and retries
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Optimize connection pool size for parallel requests
            adapter = HTTPAdapter(
                pool_connections=10,  # Number of connection pools
                pool_maxsize=20,      # Max connections per pool
                max_retries=Retry(
                    total=0,  # We handle retries ourselves
                    connect=0,
                    read=0,
                    redirect=3,
                    status_forcelist=None,
                )
            )
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
            
            # Set default headers for keep-alive
            self._session.headers.update({
                'Connection': 'keep-alive',
                'Keep-Alive': '300',
            })
    
    def close(self) -> None:
        """Close the HTTP session and cleanup resources."""
        if hasattr(self, '_session'):
            self._session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session."""
        self.close()
        return False

    def extract(self, request: PromptRequest) -> dict[str, Any]:
        return self._extract_validated(
            request,
            validate_vlm_result,
            "axis-tick",
        )

    def extract_json(
        self,
        request: PromptRequest,
        validator: Callable[[Any], dict[str, Any]],
        result_description: str = "structured",
    ) -> dict[str, Any]:
        """Return validated JSON for another task using the same Ollama client."""
        return self._extract_validated(request, validator, result_description)

    def _extract_validated(
        self,
        request: PromptRequest,
        validator: Callable[[Any], dict[str, Any]],
        result_description: str,
    ) -> dict[str, Any]:
        if not request.images:
            raise ValueError("At least one image is required")
        encoded_images = [image_to_base64(image) for image in request.images]
        prompt = request.prompt
        output_token_limit = self.num_predict
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                # Use streaming=False for better connection reuse
                response = self._session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "format": "json",
                        "think": False,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                                "images": encoded_images,
                            }
                        ],
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": output_token_limit,
                        },
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                try:
                    if not isinstance(body, dict):
                        raise ValueError("Ollama response body must be a JSON object")
                    message = body.get("message")
                    if not isinstance(message, dict):
                        raise ValueError("Ollama response has no message object")
                    content = message.get("content")
                    if not isinstance(content, str):
                        raise ValueError("Ollama response has no message.content string")
                    parsed = json.loads(
                        self._strip_code_fence(content),
                        parse_int=str,
                        parse_float=str,
                    )
                    return validator(parsed)
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    last_error = exc
                    if attempt >= self.retries:
                        break
                    if isinstance(exc, json.JSONDecodeError):
                        previous_limit = output_token_limit
                        output_token_limit = min(
                            max(
                                output_token_limit * 2,
                                output_token_limit + _MIN_RETRY_TOKEN_INCREASE,
                            ),
                            max(self.num_predict, _MAX_RETRY_NUM_PREDICT),
                        )
                        prompt = (
                            request.prompt
                            + "\nYour previous response was incomplete or invalid JSON. "
                            f"Return one complete, valid JSON object for the "
                            f"{result_description} task, following the requested schema "
                            "exactly. Do not stop before closing all strings, arrays, "
                            "and objects."
                        )
                        if output_token_limit > previous_limit:
                            LOGGER.warning(
                                "Retrying incomplete Qwen JSON response (%d/%d) "
                                "with num_predict=%d",
                                attempt + 1,
                                self.retries,
                                output_token_limit,
                            )
                        # No sleep needed - retry immediately with more tokens
                    else:
                        prompt = (
                            request.prompt
                            + "\nYour previous response was invalid: "
                            + str(exc)
                            + f". Return one valid JSON object for the {result_description} "
                            "task, following the requested schema exactly."
                        )
                        LOGGER.warning(
                            "Retrying invalid Qwen JSON response (%d/%d): %s",
                            attempt + 1,
                            self.retries,
                            exc,
                        )
                        # No sleep needed for validation errors
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                # Calculate exponential backoff sleep time
                sleep_time = self.retry_backoff * (2 ** attempt)
                LOGGER.warning(
                    "Ollama request failed (%d/%d): %s. Retrying in %.1f seconds...",
                    attempt + 1,
                    self.retries,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
        raise RuntimeError(
            f"Ollama model {self.model!r} did not return a valid "
            f"{result_description} JSON response "
            f"after {self.retries + 1} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        value = content.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return value
