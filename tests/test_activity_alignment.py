from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from citybehavex.activities.alignment import (
    _save_cache,
    cluster_profile_embeddings,
    expand_cluster_scores,
    score_activity_alignment,
    score_poi_semantic_alignment,
    score_poi_type_alignment,
    _cache_key,
    _period_cache_key,
    _poi_cache_key,
    _poi_type_cache_key,
)
from citybehavex.activities.poi_semantic import build_poi_semantic_activity_data
from citybehavex.activities.config import ActivitiesConfig
from citybehavex.llm_diaries import Diary


def _diary(diary_id: str = "schedule-030") -> Diary:
    return Diary.model_validate(
        {
            "diary_id": diary_id,
            "episodes": [
                {"start": "00:00", "end": "08:00", "purpose": "HOME"},
                {"start": "08:00", "end": "17:00", "purpose": "WORK"},
                {"start": "17:00", "end": "24:00", "purpose": "HOME"},
            ],
        }
    )


def _diary_with_other(diary_id: str = "schedule-031") -> Diary:
    return Diary.model_validate(
        {
            "diary_id": diary_id,
            "episodes": [
                {"start": "00:00", "end": "08:00", "purpose": "HOME"},
                {"start": "08:00", "end": "17:00", "purpose": "OTHER"},
                {"start": "17:00", "end": "24:00", "purpose": "WORK"},
            ],
        }
    )


def test_profile_clustering_reuses_scores_and_expands_to_agents():
    narratives = ["worker a", "worker b", "student"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    clusters = cluster_profile_embeddings(narratives, embeddings, threshold=0.94)
    assert clusters.labels.tolist() == [0, 0, 1]
    assert clusters.narratives == ["worker a", "student"]

    cluster_scores = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float64)
    expanded = expand_cluster_scores(cluster_scores, clusters.labels)
    np.testing.assert_allclose(expanded, [[0.1, 0.9], [0.1, 0.9], [0.8, 0.2]])


def test_activity_alignment_scores_only_valid_block_activities(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"index": i, "score": 0.5} for i in range(len(calls[-1]["pairs"]))]

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    result = score_activity_alignment(
        ["profile cluster"],
        [_diary()],
        ActivitiesConfig(
            enabled=True,
            alignment_backend="rerank",
            alignment_base_url="http://tei.local",
            alignment_model="activity-aligner",
            alignment_batch_size=100,
        ),
    )

    assert result is not None
    scores, blocks, metadata = result
    assert scores.shape[:3] == (1, 3, 26)
    assert [b.purpose for b in blocks] == ["HOME", "WORK", "HOME"]
    work_rows = metadata[metadata["purpose"] == "WORK"]
    assert set(work_rows["activity"]).issubset({"eatdrink", "paidwork", "commute"})
    assert "sleep" not in set(work_rows["activity"])
    assert calls[0]["model"] == "activity-aligner"


def test_activity_alignment_skips_other_blocks(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"index": i, "score": 0.5} for i in range(len(calls[-1]["pairs"]))]

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    result = score_activity_alignment(
        ["profile cluster"],
        [_diary_with_other()],
        ActivitiesConfig(
            enabled=True,
            alignment_backend="rerank",
            alignment_base_url="http://tei.local",
            alignment_model="activity-aligner",
            alignment_batch_size=100,
        ),
    )

    assert result is not None
    scores, blocks, metadata = result
    other_block = next(block for block in blocks if block.purpose == "OTHER")
    assert not np.any(scores[:, other_block.block_id])
    assert "OTHER" not in set(metadata["purpose"])


def test_poi_semantic_alignment_scores_only_masked_activities(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"index": i, "score": 0.5} for i in range(len(calls[-1]["pairs"]))]

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    poi_data = build_poi_semantic_activity_data()
    result = score_poi_semantic_alignment(
        ["profile cluster"],
        ActivitiesConfig(
            enabled=True,
            alignment_backend="rerank",
            alignment_base_url="http://tei.local",
            alignment_model="activity-aligner",
            alignment_batch_size=10_000,
        ),
        poi_data,
    )

    assert result is not None
    scores, metadata = result
    assert scores.shape == (1, len(poi_data.semantic_clusters), 25)
    food_id = poi_data.cluster_to_id["food_drink"]
    food = metadata[metadata["semantic_cluster"] == "food_drink"]
    assert set(food["activity"]) == {"eatdrink", "shopserv", "read", "compint", "goout", "leisure"}
    assert not np.any(scores[0, food_id, [0, 3, 16, 17]])
    all_texts = [text for call in calls for _query, text in call["pairs"]]
    assert not any(text.startswith("travel:") or text.startswith("commute:") for text in all_texts)


def test_poi_type_alignment_scores_only_other_blocks(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"index": i, "score": 0.5} for i in range(len(calls[-1]["pairs"]))]

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    poi_data = build_poi_semantic_activity_data()
    food_id = poi_data.cluster_to_id["food_drink"]
    education_id = poi_data.cluster_to_id["education"]
    result = score_poi_type_alignment(
        ["profile cluster"],
        [_diary_with_other()],
        ActivitiesConfig(
            enabled=True,
            alignment_backend="rerank",
            alignment_base_url="http://tei.local",
            alignment_model="activity-aligner",
            alignment_batch_size=100,
            poi_type_choice_enabled=True,
        ),
        poi_data,
        available_cluster_ids=[food_id, education_id],
    )

    assert result is not None
    scores, blocks, metadata = result
    assert scores.shape == (1, 3, len(poi_data.semantic_clusters))
    other_block = next(block for block in blocks if block.purpose == "OTHER")
    home_block = next(block for block in blocks if block.purpose == "HOME")
    assert np.any(scores[0, other_block.block_id, [food_id, education_id]])
    assert not np.any(scores[0, home_block.block_id])
    assert set(metadata["purpose"]) == {"OTHER"}
    assert set(metadata["semantic_cluster"]) == {"food_drink", "education"}
    all_queries = " ".join(pair[0] for call in calls for pair in call["pairs"])
    assert "OTHER from 08:00 to 17:00" in all_queries
    assert "HOME" not in all_queries


def test_alignment_cache_keys_distinguish_score_products():
    diary = _diary_with_other()
    block = score_activity_alignment.__globals__["diary_activity_blocks"]([diary])[1]
    catalog = score_activity_alignment.__globals__["build_catalog"]()
    activity_text = "eatdrink: Eating and drinking"

    keys = {
        _cache_key("model", "profile", block, -1, activity_text),
        _period_cache_key("model", "profile", "WORK", 2, -1, activity_text),
        _poi_cache_key("model", "profile", "food_drink", activity_text),
        _poi_type_cache_key("model", "profile", block, "food_drink"),
    }
    assert len(keys) == 4


def test_activity_alignment_reuses_period_scores_for_matching_raw_blocks(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"index": i, "score": _deterministic_score(query, text)}
                for i, (query, text) in enumerate(calls[-1]["pairs"])
            ]

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    one_diary = [_diary("routine-001")]
    two_matching_diaries = [_diary("routine-001"), _diary("routine-002")]
    config = ActivitiesConfig(
        enabled=True,
        alignment_backend="rerank",
        alignment_base_url="http://tei.local",
        alignment_model="activity-aligner",
        alignment_batch_size=10_000,
    )

    first = score_activity_alignment(["worker profile"], one_diary, config)
    one_diary_pairs = sum(len(call["pairs"]) for call in calls)
    calls.clear()
    result = score_activity_alignment(["worker profile"], two_matching_diaries, config)
    two_diary_pairs = sum(len(call["pairs"]) for call in calls)

    assert result is not None
    scores, blocks, metadata = result
    assert first is not None
    assert one_diary_pairs == two_diary_pairs
    assert {"period_index", "period_label"}.issubset(metadata.columns)

    by_id = {block.diary_id: [] for block in blocks}
    for block in blocks:
        by_id[block.diary_id].append(block.block_id)
    routine_1, routine_2 = by_id["routine-001"], by_id["routine-002"]
    np.testing.assert_allclose(scores[0, routine_1[0]], scores[0, routine_2[0]])
    np.testing.assert_allclose(scores[0, routine_1[1]], scores[0, routine_2[1]])
    np.testing.assert_allclose(scores[0, routine_1[2]], scores[0, routine_2[2]])
    assert np.any(scores[0, routine_1[0]])

    all_queries = " ".join(pair[0] for call in calls for pair in call["pairs"])
    assert "routine-001" not in all_queries
    assert "routine-002" not in all_queries
    assert "HOME blocks mostly in the 00-06 period" in all_queries
    assert "WORK blocks mostly in the 12-18 period" in all_queries


def test_activity_alignment_period_groups_merge_previous_activity_candidates(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return _FakeResponse([0.5 for _query, _text in json["pairs"]])

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    diaries = [
        Diary.model_validate(
            {
                "diary_id": "after-other",
                "episodes": [
                    {"start": "00:00", "end": "01:00", "purpose": "OTHER"},
                    {"start": "01:00", "end": "05:00", "purpose": "HOME"},
                    {"start": "05:00", "end": "24:00", "purpose": "HOME"},
                ],
            }
        ),
        Diary.model_validate(
            {
                "diary_id": "after-work",
                "episodes": [
                    {"start": "00:00", "end": "01:00", "purpose": "WORK"},
                    {"start": "01:00", "end": "05:00", "purpose": "HOME"},
                    {"start": "05:00", "end": "24:00", "purpose": "HOME"},
                ],
            }
        ),
    ]

    result = score_activity_alignment(
        ["profile"],
        diaries,
        ActivitiesConfig(
            enabled=True,
            alignment_backend="rerank",
            alignment_base_url="http://tei.local",
            alignment_model="activity-aligner",
            alignment_batch_size=10_000,
        ),
    )

    assert result is not None
    all_queries = " ".join(pair[0] for call in calls for pair in call["pairs"])
    assert "HOME blocks mostly in the 00-06 period" in all_queries
    assert "previous micro-activity was paidwork" in all_queries
    assert "previous micro-activity was shopserv" in all_queries


class _FakeResponse:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return [{"index": i, "score": s} for i, s in enumerate(self._scores)]


def _deterministic_score(query: str, text: str) -> float:
    import hashlib

    digest = hashlib.sha256(f"{query}\x00{text}".encode()).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def _many_diaries_config(**overrides) -> ActivitiesConfig:
    defaults = dict(
        enabled=True,
        alignment_backend="rerank",
        alignment_base_url="http://tei.local",
        alignment_model="activity-aligner",
        alignment_batch_size=5,
    )
    defaults.update(overrides)
    return ActivitiesConfig(**defaults)


def test_activity_alignment_concurrency_matches_sequential(monkeypatch):
    # Each call derives its scores purely from the request payload (no shared
    # mutable state), so this fake is safe to call from multiple threads, and
    # jittered per-call sleeps force out-of-order completion under concurrency.
    def fake_post(url, headers, json, timeout):
        pairs = json["pairs"]
        time.sleep(0.001 * (len(pairs) % 3))
        scores = [_deterministic_score(q, t) for q, t in pairs]
        return _FakeResponse(scores)

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    diaries = [_diary("routine-001"), _diary("routine-002")]
    narratives = ["profile cluster a", "profile cluster b", "profile cluster c"]

    seq_result = score_activity_alignment(
        narratives, diaries, _many_diaries_config(alignment_concurrency=1)
    )
    conc_result = score_activity_alignment(
        narratives, diaries, _many_diaries_config(alignment_concurrency=4)
    )

    assert seq_result is not None and conc_result is not None
    seq_scores, _seq_blocks, seq_metadata = seq_result
    conc_scores, _conc_blocks, conc_metadata = conc_result
    np.testing.assert_allclose(seq_scores, conc_scores)
    pd_sort_cols = ["cluster", "block_id", "previous_activity", "activity_idx"]
    seq_sorted = seq_metadata.sort_values(pd_sort_cols).reset_index(drop=True)
    conc_sorted = conc_metadata.sort_values(pd_sort_cols).reset_index(drop=True)
    assert seq_sorted.equals(conc_sorted)


def test_activity_alignment_retries_transient_failure(monkeypatch):
    attempts: dict[str, int] = {}
    lock = threading.Lock()

    def fake_post(url, headers, json, timeout):
        pairs = json["pairs"]
        key = pairs[0][0]  # first pair's query text identifies this chunk
        with lock:
            attempts[key] = attempts.get(key, 0) + 1
            attempt_count = attempts[key]
        if attempt_count == 1:
            raise ConnectionError("simulated transient failure")
        return _FakeResponse([_deterministic_score(q, t) for q, t in pairs])

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    result = score_activity_alignment(
        ["profile cluster"],
        [_diary()],
        _many_diaries_config(alignment_concurrency=1, alignment_retries=2),
    )

    assert result is not None
    # Every chunk should have failed exactly once before its retry succeeded.
    assert attempts and all(count >= 2 for count in attempts.values())


def test_activity_alignment_returns_none_when_retries_exhausted(monkeypatch):
    def fake_post(url, headers, json, timeout):
        raise ConnectionError("simulated permanent failure")

    monkeypatch.setattr("citybehavex.activities.alignment.requests.post", fake_post)

    result = score_activity_alignment(
        ["profile cluster"],
        [_diary()],
        _many_diaries_config(alignment_concurrency=1, alignment_retries=2),
    )

    assert result is None


def test_save_cache_is_atomic_on_failure(tmp_path, monkeypatch):
    cache_path = tmp_path / "activity_alignment_cache.npz"
    _save_cache(cache_path, {"existing-key": 0.42})
    assert cache_path.exists()

    original_savez = np.savez

    def broken_savez(fh, **kwargs):
        original_savez(fh, **kwargs)
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez", broken_savez)
    with pytest.raises(OSError):
        _save_cache(cache_path, {"new-key": 0.99})

    # The original cache must survive a failed write untouched, and no stray
    # temp file should be left behind.
    from citybehavex.activities.alignment import _load_cache

    reloaded = _load_cache(cache_path)
    assert reloaded.keys() == {"existing-key"}
    assert reloaded["existing-key"] == pytest.approx(0.42)
    assert not (tmp_path / "activity_alignment_cache.npz.tmp").exists()
