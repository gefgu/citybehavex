from __future__ import annotations

import numpy as np

from citybehavex.activities.alignment import (
    cluster_profile_embeddings,
    expand_cluster_scores,
    score_activity_alignment,
)
from citybehavex.activities.config import ActivitiesConfig
from citybehavex.llm_diaries import Diary


def _diary() -> Diary:
    return Diary.model_validate(
        {
            "diary_id": "schedule-030",
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
            return [{"index": i, "score": 0.5} for i in range(len(calls[-1]["texts"]))]

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
