from __future__ import annotations

import numpy as np

from citybehavex.activities.alignment import ProfileClusters
from citybehavex.profiles.config import AgentProfilesConfig
from citybehavex.profiles.ownership_alignment import (
    expand_vehicle_scores,
    score_vehicle_ownership_alignment,
)


def test_score_vehicle_ownership_alignment_scores_car_and_bike(monkeypatch):
    calls = []

    def fake_score_chunk(base_url, model, pairs, *, timeout, retries):
        calls.append((base_url, model, pairs, timeout, retries))
        return [0.8 if "private car" in text else 0.3 for _query, text in pairs]

    monkeypatch.setattr(
        "citybehavex.profiles.ownership_alignment._score_chunk_with_retries",
        fake_score_chunk,
    )
    config = AgentProfilesConfig(
        ownership_alignment_backend="rerank",
        ownership_alignment_base_url="http://localhost:8084",
        ownership_alignment_model="models/modernbert-vehicle-ownership-aligner",
        ownership_alignment_batch_size=8,
        ownership_alignment_concurrency=1,
    )

    result = score_vehicle_ownership_alignment(
        ["transport-neutral profile"],
        config,
        city_profile="dense metro",
    )

    assert result is not None
    scores, metadata = result
    assert scores.tolist() == [[0.8, 0.3]]
    assert metadata["vehicle"].tolist() == ["car", "bike"]
    assert "City context: dense metro" in metadata.loc[0, "query_text"]
    assert len(calls) == 1


def test_expand_vehicle_scores_uses_profile_cluster_labels():
    clusters = ProfileClusters(
        labels=np.array([1, 0, 1], dtype=np.int64),
        narratives=["cluster 0", "cluster 1"],
        representative_indices=np.array([1, 0], dtype=np.int64),
    )
    scores = np.array([[0.2, 0.4], [0.8, 0.1]], dtype=np.float64)

    expanded = expand_vehicle_scores(scores, clusters)

    assert expanded.tolist() == [[0.8, 0.1], [0.2, 0.4], [0.8, 0.1]]
