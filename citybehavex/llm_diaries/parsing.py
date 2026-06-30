from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .models import ChatCompletionResponse, Diary, DiaryBatch, DiaryValidationError


def parse_chat_completion_response(payload: Any) -> ChatCompletionResponse:
    try:
        return ChatCompletionResponse.model_validate(payload)
    except ValidationError as exc:
        raise DiaryValidationError(f"invalid OpenAI-compatible response: {exc}") from exc


def parse_diary_content(content: str) -> DiaryBatch:
    try:
        payload = _loads_model_json(content)
    except json.JSONDecodeError as exc:
        raise DiaryValidationError(f"diary content is not valid JSON: {exc}") from exc
    try:
        return DiaryBatch.model_validate(payload)
    except ValidationError as exc:
        raise DiaryValidationError(f"invalid diary payload: {exc}") from exc


def parse_diary_response(payload: Any) -> DiaryBatch:
    response = parse_chat_completion_response(payload)
    return parse_diary_content(response.choices[0].message.content)


def parse_single_diary_content(content: str) -> Diary:
    try:
        payload = _loads_model_json(content)
    except json.JSONDecodeError as exc:
        raise DiaryValidationError(f"diary content is not valid JSON: {exc}") from exc
    if isinstance(payload, dict) and "diary" in payload:
        payload = payload["diary"]
    elif isinstance(payload, dict) and "diaries" in payload:
        diaries = payload["diaries"]
        if not isinstance(diaries, list) or len(diaries) != 1:
            raise DiaryValidationError("single-diary response must contain exactly one diary")
        payload = diaries[0]
    try:
        return Diary.model_validate(payload)
    except ValidationError as exc:
        raise DiaryValidationError(f"invalid diary payload: {exc}") from exc


def parse_single_diary_response(payload: Any) -> Diary:
    response = parse_chat_completion_response(payload)
    return parse_single_diary_content(response.choices[0].message.content)


def diary_schema() -> dict[str, Any]:
    return DiaryBatch.model_json_schema()


def _loads_model_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)
