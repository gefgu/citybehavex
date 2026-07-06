from __future__ import annotations

import numpy as np
import pytest

from citybehavex.schedules.alignment import _extract_scores, _load_cache, _save_cache, score_alignment_matrix
from citybehavex.schedules import ScheduleConfig
from citybehavex.llm_diaries import Diary


def _diary(diary_id: str, away_purpose: str, away_start: str, away_end: str) -> Diary:
    return Diary.model_validate(
        {
            "diary_id": diary_id,
            "episodes": [
                {"start": "00:00", "end": away_start, "purpose": "HOME"},
                {"start": away_start, "end": away_end, "purpose": away_purpose},
                {"start": away_end, "end": "24:00", "purpose": "HOME"},
            ],
        }
    )


def test_extract_scores_accepts_common_tei_shapes():
    assert _extract_scores([0.1, 0.9], 2) == [0.1, 0.9]
    assert _extract_scores({"scores": [0.2, 0.8]}, 2) == [0.2, 0.8]
    assert _extract_scores(
        [{"index": 1, "score": 0.7}, {"index": 0, "score": 0.3}], 2
    ) == [0.3, 0.7]


def test_score_alignment_matrix_uses_rerank_endpoint(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"index": 0, "score": 0.25}, {"index": 1, "score": 1.2}]

    def fake_post(url, headers, json, timeout):
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setattr("citybehavex.schedules.alignment.requests.post", fake_post)

    diaries = [
        _diary("d1", "WORK", "09:00", "17:00"),
        _diary("d2", "OTHER", "12:00", "18:00"),
    ]
    cfg = ScheduleConfig(
        similarity_backend="alignment_model",
        alignment_base_url="http://tei.local",
        alignment_model="models/modernbert-schedule-aligner",
        alignment_batch_size=4,
    )
    scores = score_alignment_matrix(["profile text"], diaries, cfg)

    assert scores is not None
    np.testing.assert_allclose(scores, np.array([[0.25, 1.0]]))
    assert calls[0][0] == "http://tei.local/rerank"
    assert calls[0][1]["query"] == "profile text"
    assert calls[0][1]["model"] == "models/modernbert-schedule-aligner"


def test_score_alignment_matrix_returns_none_on_bad_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"unexpected": True}

    monkeypatch.setattr(
        "citybehavex.schedules.alignment.requests.post",
        lambda *args, **kwargs: Response(),
    )

    scores = score_alignment_matrix(
        ["profile"],
        [_diary("d1", "WORK", "09:00", "17:00")],
        ScheduleConfig(
            similarity_backend="alignment_model",
            alignment_base_url="http://tei.local",
        ),
    )

    assert scores is None


def test_score_alignment_matrix_only_refetches_missing_cache_entries(monkeypatch, tmp_path):
    calls = []

    class Response:
        def __init__(self, texts):
            self._texts = texts

        def raise_for_status(self):
            return None

        def json(self):
            return [{"index": i, "score": 0.5} for i in range(len(self._texts))]

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return Response(json["texts"])

    monkeypatch.setattr("citybehavex.schedules.alignment.requests.post", fake_post)

    diaries = [
        _diary("d1", "WORK", "09:00", "17:00"),
        _diary("d2", "OTHER", "12:00", "18:00"),
    ]
    cache_path = tmp_path / "schedule_alignment_cache.npz"
    cfg = ScheduleConfig(
        similarity_backend="alignment_model",
        alignment_base_url="http://tei.local",
        alignment_model="models/modernbert-schedule-aligner",
        alignment_cache_path=str(cache_path),
    )

    first = score_alignment_matrix(["profile a", "profile b"], diaries, cfg)
    assert first is not None
    assert len(calls) == 2  # one rerank call per profile, all diaries missing

    # A second run with the same cache path (simulating a new simulation
    # reusing the same profiles/diaries) must not re-request anything --
    # everything it needs is already cached.
    calls.clear()
    second = score_alignment_matrix(["profile a", "profile b"], diaries, cfg)
    assert second is not None
    assert calls == []
    np.testing.assert_allclose(first, second)

    # A third run adding one new profile must only fetch that profile's row.
    third = score_alignment_matrix(["profile a", "profile b", "profile c"], diaries, cfg)
    assert third is not None
    assert len(calls) == 1
    assert calls[0]["query"] == "profile c"


def test_save_cache_is_atomic_on_failure(tmp_path, monkeypatch):
    cache_path = tmp_path / "schedule_alignment_cache.npz"
    _save_cache(cache_path, {"existing-key": 0.42})
    assert cache_path.exists()

    original_savez = np.savez

    def broken_savez(fh, **kwargs):
        original_savez(fh, **kwargs)
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez", broken_savez)
    with pytest.raises(OSError):
        _save_cache(cache_path, {"new-key": 0.99})

    reloaded = _load_cache(cache_path)
    assert reloaded.keys() == {"existing-key"}
    assert reloaded["existing-key"] == pytest.approx(0.42)
    assert not (tmp_path / "schedule_alignment_cache.npz.tmp").exists()
