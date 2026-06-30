from __future__ import annotations

from typing import Any

import requests

from citybehavex.config import LLMConfig

from .models import DiaryValidationError, LLMStats


class OpenAICompatibleDiaryClient:
    """Small wrapper around the OpenAI-compatible chat endpoints used for diaries."""

    def __init__(self, config: LLMConfig, *, requests_module=requests) -> None:
        self.config = config
        self.requests = requests_module
        self.base_url = config.base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    def preflight(self) -> None:
        try:
            response = self.requests.get(
                self.models_url,
                headers=self.headers,
                timeout=min(self.config.timeout_seconds, 10.0),
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - converted to domain error.
            raise DiaryValidationError(
                f"LLM server preflight failed at {self.models_url}: {exc}"
            ) from exc

    def generate_json(self, prompt: str, *, stats: LLMStats | None = None) -> Any:
        request_payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You generate strictly valid JSON for mobility simulation.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.config.max_tokens is not None:
            request_payload["max_tokens"] = self.config.max_tokens

        if stats is not None:
            stats.calls += 1
        response = self.requests.post(
            self.chat_url,
            headers=self.headers,
            json=request_payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise DiaryValidationError(
                f"LLM server returned non-JSON response at {self.chat_url}: {response.text[:500]}"
            ) from exc
