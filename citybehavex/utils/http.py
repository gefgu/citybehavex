from __future__ import annotations

import random
import time
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
    backoff_base_seconds: float = 0.5,
    backoff_max_seconds: float = 8.0,
    sleep=time.sleep,
) -> Any:
    """POST JSON and return the parsed JSON body, retrying failed attempts.

    Sleeps between attempts (never before the first or after the last) with
    exponential backoff plus jitter, so a briefly-refusing/overloaded server
    gets a real chance to recover instead of being hit by a tight retry loop.
    """
    last_error: Exception | None = None
    attempts = max(1, retries)
    for attempt in range(attempts):
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
            if attempt < attempts - 1:
                delay = min(backoff_max_seconds, backoff_base_seconds * (2**attempt))
                sleep(delay + random.uniform(0, delay * 0.5))
    raise RuntimeError(f"POST {url} failed after {attempts} attempt(s)") from last_error


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


# generate_json() previously hardcoded retries=1 here (zero backoff room --
# a flaky/briefly-refusing remote host got exactly one shot per outer-loop
# attempt). Diary generation and profile-weight calibration each already
# wrap this in their own outer retry loop (config.retries, no delay between
# iterations), so bumping this inner count gives real exponential-backoff
# retries per outer attempt without changing either caller.
CHAT_COMPLETION_INNER_RETRIES = 4
