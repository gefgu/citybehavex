from __future__ import annotations

import json

import pandas as pd
import pytest

from citybehavex.config import LLMConfig
from citybehavex.diaries import diary_batch_to_markov_training
from citybehavex.llm_diaries import (
    DiaryBatch,
    DiaryValidationError,
    fetch_diary_batch,
    parse_diary_response,
    parse_single_diary_response,
)


def _diary(day_id: int = 1) -> dict:
    return {
        "diary_id": f"routine-{day_id}",
        "episodes": [
            {"start": "00:00", "end": "07:00", "purpose": "HOME"},
            {"start": "07:00", "end": "09:00", "purpose": "OTHER"},
            {"start": "09:00", "end": "17:00", "purpose": "WORK"},
            {"start": "17:00", "end": "19:00", "purpose": "PURCHASE"},
            {"start": "19:00", "end": "24:00", "purpose": "HOME"},
        ],
    }


def _batch() -> dict:
    return {
        "representative_day": "2026-01-01",
        "diaries": [_diary(i) for i in range(10)],
    }


def _chat(payload: dict) -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
    }


def test_parse_valid_openai_compatible_response():
    batch = parse_diary_response(_chat(_batch()))
    assert isinstance(batch, DiaryBatch)
    assert len(batch.diaries) == 10
    assert batch.diaries[0].episodes[0].purpose == "HOME"


def test_parse_single_diary_response_accepts_fenced_json():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "```json\n" + json.dumps(_diary(1)) + "\n```"
                }
            }
        ]
    }
    diary = parse_single_diary_response(payload)
    assert diary.diary_id == "routine-1"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": "not json"}}]},
    ],
)
def test_parse_malformed_response_fails(payload):
    with pytest.raises(DiaryValidationError):
        parse_diary_response(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda batch: batch["diaries"].pop(),
        lambda batch: batch["diaries"][0]["episodes"].__setitem__(0, {"start": "00:00", "end": "01:00", "purpose": "WORK"}),
        lambda batch: batch["diaries"][0]["episodes"].__setitem__(1, {"start": "06:00", "end": "09:00", "purpose": "OTHER"}),
        lambda batch: batch["diaries"][0]["episodes"].__setitem__(2, {"start": "09:00", "end": "17:00", "purpose": "BAD"}),
    ],
)
def test_invalid_diary_payload_fails(mutate):
    payload = _batch()
    mutate(payload)
    with pytest.raises(DiaryValidationError):
        parse_diary_response(_chat(payload))


def test_cache_fallback_after_llm_failure(monkeypatch, tmp_path):
    valid_cache = tmp_path / "validated_diaries.json"
    valid_cache.write_text(json.dumps(_batch()), encoding="utf-8")

    def raise_request(*args, **kwargs):
        raise RuntimeError("server down")

    monkeypatch.setattr("citybehavex.llm_diaries.requests.get", raise_request)
    config = LLMConfig(
        base_url="http://localhost:8000",
        api_key="test",
        model="test-model",
        cache_dir=str(tmp_path),
        validated_diaries_path=str(valid_cache),
    )
    batch = fetch_diary_batch(
        config,
        city_profile="test city",
        representative_day="2026-01-01",
    )
    assert len(batch.diaries) == 10


def test_fetch_diary_batch_calls_llm_once_per_diary(monkeypatch, tmp_path):
    calls = []

    class Response:
        text = ""

        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return Response(_chat(_diary(len(calls))))

    monkeypatch.setattr("citybehavex.llm_diaries.requests.get", lambda *args, **kwargs: Response({"data": []}))
    monkeypatch.setattr("citybehavex.llm_diaries.requests.post", post)
    config = LLMConfig(
        base_url="http://localhost:8000",
        api_key="test",
        model="test-model",
        cache_dir=str(tmp_path),
        validated_diaries_path=str(tmp_path / "validated_diaries.json"),
        diary_count=10,
    )

    batch = fetch_diary_batch(
        config,
        city_profile="test city",
        representative_day="2026-01-01",
    )

    assert len(batch.diaries) == 10
    assert len(calls) == 10
    assert (tmp_path / "raw_response_001.json").exists()
    assert (tmp_path / "prompt_001.txt").exists()


def test_markov_training_requires_validated_batch():
    batch = DiaryBatch.model_validate(_batch())
    training = diary_batch_to_markov_training(batch, representative_day="2026-01-01")
    assert list(training.columns) == ["uid", "datetime", "location", "purpose"]
    assert training["uid"].nunique() == 10
    assert pd.api.types.is_datetime64_any_dtype(training["datetime"])
