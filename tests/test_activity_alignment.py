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
)
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


def test_activity_alignment_visited_pairs_prunes_unreachable_cluster_block_combos(monkeypatch):
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

    diaries = [_diary("routine-001"), _diary("routine-002")]
    narratives = ["worker profile", "student profile"]
    # routine-001 is blocks 0,1,2; routine-002 is blocks 3,4,5 (diary_activity_blocks
    # numbers block_id sequentially across all diaries). Only allow cluster 0's
    # blocks from routine-001 -- everything else must be pruned away.
    visited = {(0, 0), (0, 1), (0, 2)}

    result = score_activity_alignment(
        narratives,
        diaries,
        ActivitiesConfig(
            enabled=True,
            alignment_backend="rerank",
            alignment_base_url="http://tei.local",
            alignment_model="activity-aligner",
            alignment_batch_size=100,
        ),
        visited_pairs=visited,
    )

    assert result is not None
    scores, blocks, metadata = result

    # Excluded cluster (1) must be entirely untouched (zero-initialized default).
    assert not np.any(scores[1])
    # Excluded blocks (routine-002, block_id 3/4/5) for the included cluster must
    # also be entirely zero.
    excluded_block_ids = [b.block_id for b in blocks if b.diary_id == "routine-002"]
    for block_id in excluded_block_ids:
        assert not np.any(scores[0, block_id])
    # The one allowed (cluster, block) combo must have real (non-zero) scores.
    included_block_ids = [b.block_id for b in blocks if b.diary_id == "routine-001"]
    assert any(np.any(scores[0, block_id]) for block_id in included_block_ids)

    # No metadata rows for pruned combinations.
    assert set(zip(metadata["cluster"], metadata["diary_id"])) == {(0, "routine-001")}

    # No network calls should ever have carried pruned-combo content.
    all_queries = " ".join(
        pair[0] for call in calls for pair in call["pairs"]
    )
    assert "student profile" not in all_queries
    assert "routine-002" not in all_queries


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
