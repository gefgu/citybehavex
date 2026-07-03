from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from citybehavex.llm import LLMConfig
from citybehavex.llm_diaries.training import annotate_trajectory_purposes_ddcrp, diary_batch_to_markov_training
from citybehavex.llm_diaries import (
    Diary,
    DiaryBatch,
    DiaryValidationError,
    LLMStats,
    allocate_location_counts,
    build_single_diary_prompt,
    fetch_diary_batch,
    load_validated_diary_cache,
    lognormal_location_probabilities,
    parse_diary_response,
    parse_single_diary_response,
)

DEFAULT_COUNTS = allocate_location_counts(1.0, 0.5, 6, 10)


def _diary(day_id: int = 1) -> dict:
    return {
        "diary_id": f"routine-{day_id}",
        "episodes": [
            {"start": "00:00", "end": "07:00", "purpose": "HOME"},
            {"start": "07:00", "end": "09:00", "purpose": "OTHER"},
            {"start": "09:00", "end": "17:00", "purpose": "WORK"},
            {"start": "17:00", "end": "19:00", "purpose": "OTHER"},
            {"start": "19:00", "end": "24:00", "purpose": "HOME"},
        ],
    }


def _home_diary(day_id: int = 1) -> dict:
    return {
        "diary_id": f"routine-{day_id}",
        "episodes": [
            {"start": "00:00", "end": "24:00", "purpose": "HOME"},
        ],
    }


def _batch(
    *,
    location_counts: list[int] | None = None,
    mu: float = 1.0,
    sigma: float = 0.5,
    max_locations: int = 6,
) -> dict:
    counts = location_counts or DEFAULT_COUNTS
    return {
        "representative_day": "2026-01-01",
        "location_count_distribution": {
            "mu": mu,
            "sigma": sigma,
            "max_locations": max_locations,
        },
        "target_location_counts": counts,
        "diaries": [
            _home_diary(i) if count == 1 else _diary(i)
            for i, count in enumerate(counts, start=1)
        ],
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


def test_obsolete_diary_fields_are_ignored_and_not_serialized():
    payload = _batch()
    payload["diaries"][0]["description"] = "obsolete top-level diary text"
    payload["diaries"][0]["episodes"][0]["duration_minutes"] = None
    payload["diaries"][0]["episodes"][0]["notes"] = "obsolete episode text"

    batch = parse_diary_response(_chat(payload))
    dumped = batch.model_dump()

    assert "description" not in dumped["diaries"][0]
    assert "duration_minutes" not in dumped["diaries"][0]["episodes"][0]
    assert "notes" not in dumped["diaries"][0]["episodes"][0]


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


def test_parse_single_diary_response_normalizes_known_purpose_aliases():
    payload = _diary(1)
    payload["episodes"][1]["purpose"] = "PURCHASE"

    diary = parse_single_diary_response(_chat(payload))

    assert diary.episodes[1].purpose == "OTHER"


def test_truncated_lognormal_probabilities_and_allocations():
    probabilities = lognormal_location_probabilities(1.0, 0.5, 6)

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[2] == max(probabilities.values())
    assert allocate_location_counts(1.0, 0.5, 6, 10) == [
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        5,
    ]
    assert allocate_location_counts(1.0, 0.5, 6, 20) == [
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
        5,
        6,
    ]


def test_one_location_prompt_requires_single_home_episode():
    prompt = build_single_diary_prompt(
        diary_number=1,
        diary_count=10,
        city_profile="test city",
        representative_day="2026-01-01",
        location_count=1,
    )

    assert "HOME is the only visited location" in prompt
    assert "exactly one episode from 00:00 to 24:00" in prompt


def test_prompt_lists_previous_schedules_to_avoid_duplicates():
    previous = [Diary.model_validate(_diary(1)), Diary.model_validate(_diary(2))]
    prompt = build_single_diary_prompt(
        diary_number=3,
        diary_count=10,
        city_profile="test city",
        representative_day="2026-01-01",
        location_count=4,
        previous_diaries=previous,
    )
    assert "already been generated" in prompt
    assert "do NOT" in prompt
    # The compact episode summary of a prior schedule is echoed back.
    assert "00:00-07:00 HOME | 07:00-09:00 OTHER" in prompt


def test_one_location_prompt_ignores_previous_schedules():
    previous = [Diary.model_validate(_diary(1))]
    prompt = build_single_diary_prompt(
        diary_number=2,
        diary_count=10,
        city_profile="test city",
        representative_day="2026-01-01",
        location_count=1,
        previous_diaries=previous,
    )
    assert "already been generated" not in prompt


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
        lambda batch: batch["diaries"][1]["episodes"].__setitem__(1, {"start": "06:00", "end": "09:00", "purpose": "OTHER"}),
        lambda batch: batch["diaries"][1]["episodes"].__setitem__(2, {"start": "09:00", "end": "17:00", "purpose": "BAD"}),
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
        diary_count=10,
    )
    stats = LLMStats()
    batch = fetch_diary_batch(
        config,
        city_profile="test city",
        representative_day="2026-01-01",
        location_counts=DEFAULT_COUNTS,
        stats=stats,
    )
    assert len(batch.diaries) == 10
    assert stats.calls == 0
    assert stats.cache_hits == 1


def test_legacy_and_mismatched_caches_are_rejected(tmp_path):
    legacy_path = tmp_path / "legacy.json"
    legacy = _batch()
    legacy.pop("location_count_distribution")
    legacy.pop("target_location_counts")
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(DiaryValidationError):
        load_validated_diary_cache(legacy_path)

    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(_batch(mu=1.1)), encoding="utf-8")
    config = LLMConfig(
        cache_dir=str(tmp_path),
        validated_diaries_path=str(mismatch_path),
        diary_count=10,
    )
    with pytest.raises(DiaryValidationError):
        fetch_diary_batch(
            config,
            city_profile="test city",
            representative_day="2026-01-01",
            location_counts=DEFAULT_COUNTS,
        )


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

    stats = LLMStats()
    batch = fetch_diary_batch(
        config,
        city_profile="test city",
        representative_day="2026-01-01",
        location_counts=[2] * 10,
        stats=stats,
    )

    assert len(batch.diaries) == 10
    assert len(calls) == 10
    assert stats.calls == 10
    assert batch.target_location_counts == [2] * 10
    assert batch.location_count_distribution.model_dump() == {
        "mu": 1.0,
        "sigma": 0.5,
        "max_locations": 6,
    }
    assert not (tmp_path / "raw_response.json").exists()
    assert not (tmp_path / "raw_response_001.json").exists()
    assert not (tmp_path / "prompt.txt").exists()
    assert not (tmp_path / "prompt_001.txt").exists()
    assert (tmp_path / "validated_diaries.json").exists()


def test_one_location_diary_retries_until_home_only(monkeypatch, tmp_path):
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

    responses = [_diary(1), _home_diary(1)] + [_diary(i) for i in range(2, 11)]

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return Response(_chat(responses[len(calls) - 1]))

    monkeypatch.setattr(
        "citybehavex.llm_diaries.requests.get",
        lambda *args, **kwargs: Response({"data": []}),
    )
    monkeypatch.setattr("citybehavex.llm_diaries.requests.post", post)
    config = LLMConfig(
        base_url="http://localhost:8000",
        api_key="test",
        model="test-model",
        cache_dir=str(tmp_path),
        diary_count=10,
        retries=2,
    )

    batch = fetch_diary_batch(
        config,
        city_profile="test city",
        representative_day="2026-01-01",
        location_counts=[1] + [2] * 9,
    )

    assert len(calls) == 11
    assert all(episode.purpose == "HOME" for episode in batch.diaries[0].episodes)


def test_markov_training_requires_validated_batch():
    batch = DiaryBatch.model_validate(_batch())
    training = diary_batch_to_markov_training(batch, representative_day="2026-01-01")
    assert list(training.columns) == ["uid", "datetime", "location", "purpose"]
    assert training["uid"].nunique() == 10
    assert pd.api.types.is_datetime64_any_dtype(training["datetime"])


def test_ddcrp_annotation_preserves_engine_purpose_column():
    batch = DiaryBatch.model_validate(_batch())
    traj = pd.DataFrame(
        {
            "uid": [1],
            "datetime": [pd.Timestamp("2026-01-01 08:30:00")],
            "purpose": ["OTHER"],
        }
    )

    annotated = annotate_trajectory_purposes_ddcrp(
        traj,
        bank=type("Bank", (), {"diaries": batch.diaries})(),
        chosen=np.array([[0]], dtype=np.int64),
        start_date=pd.Timestamp("2026-01-01"),
    )

    assert annotated["purpose"].tolist() == ["OTHER"]
