from __future__ import annotations

from typing import Any

import requests


def post_json_with_retries(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any],
    timeout: float,
    retries: int = 2,
    requests_module=requests,
) -> Any:
    """POST JSON and return the parsed JSON body, retrying failed attempts."""
    last_error: Exception | None = None
    for _attempt in range(max(1, retries)):
        try:
            response = requests_module.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - caller decides final error semantics.
            last_error = exc
    raise RuntimeError(f"POST {url} failed after {max(1, retries)} attempt(s)") from last_error


def post_openai_chat_json(
    base_url: str,
    *,
    model: str | None,
    messages: list[dict[str, str]],
    timeout: float,
    retries: int = 2,
    api_key: str | None = None,
    temperature: float = 0.0,
    response_format: dict[str, str] | None = None,
    max_tokens: int | None = None,
    requests_module=requests,
) -> Any:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    return post_json_with_retries(
        base_url.rstrip("/") + "/v1/chat/completions",
        headers=headers,
        payload=payload,
        timeout=timeout,
        retries=retries,
        requests_module=requests_module,
    )
