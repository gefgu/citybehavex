from __future__ import annotations

from typing import Any, Optional

import requests

from citybehavex.llm.config import LLMConfig
from citybehavex.llm_diaries.models import DiaryValidationError, LLMStats
from citybehavex.utils.http import post_openai_chat_json


class OpenAICompatibleDiaryClient:
    """Small wrapper around the OpenAI-compatible chat endpoints used for diaries."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        base_url: Optional[str] = None,
        requests_module=requests,
    ) -> None:
        self.config = config
        self.requests = requests_module
        effective_url = (base_url or config.base_url or "").rstrip("/")
        self.base_url = effective_url
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"
        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

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
        if stats is not None:
            with stats.lock:
                stats.calls += 1
        return post_openai_chat_json(
            self.base_url,
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You generate strictly valid JSON for mobility simulation.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds,
            retries=1,
            api_key=self.config.api_key,
            requests_module=self.requests,
        )
