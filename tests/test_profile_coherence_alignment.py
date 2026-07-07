from __future__ import annotations

import numpy as np

from citybehavex.activities.alignment import ProfileClusters
from citybehavex.profiles.coherence_alignment import (
    COHERENCE_CANDIDATE_TEXT,
    expand_coherence_scores,
    score_profile_coherence_alignment,
)
from citybehavex.profiles.config import AgentProfilesConfig


def test_score_profile_coherence_alignment_scores_clusters(monkeypatch):
    calls = []

    def fake_score_chunk(base_url, model, pairs, *, timeout, retries):
        calls.append((base_url, model, pairs, timeout, retries))
        return [0.9 if "adult" in query else 0.1 for query, _text in pairs]

    monkeypatch.setattr(
        "citybehavex.profiles.coherence_alignment._score_chunk_with_retries",
        fake_score_chunk,
    )
    config = AgentProfilesConfig(
        coherence_alignment_backend="rerank",
        coherence_alignment_base_url="http://localhost:8085",
        coherence_alignment_model="models/modernbert-profile-coherence-aligner",
        coherence_alignment_batch_size=8,
        coherence_alignment_concurrency=1,
    )

    result = score_profile_coherence_alignment(
        ["coherent adult profile", "teen doctorate manager profile"],
        config,
        city_profile="dense metro",
    )

    assert result is not None
    scores, metadata = result
    assert scores.tolist() == [0.9, 0.1]
    assert metadata["candidate_text"].tolist() == [
        COHERENCE_CANDIDATE_TEXT,
        COHERENCE_CANDIDATE_TEXT,
    ]
    assert "City context: dense metro" in metadata.loc[0, "query_text"]
    assert len(calls) == 1


def test_expand_coherence_scores_uses_profile_cluster_labels():
    clusters = ProfileClusters(
        labels=np.array([1, 0, 1], dtype=np.int64),
        narratives=["cluster 0", "cluster 1"],
        representative_indices=np.array([1, 0], dtype=np.int64),
    )
    scores = np.array([0.2, 0.8], dtype=np.float64)

    expanded = expand_coherence_scores(scores, clusters)

    assert expanded.tolist() == [0.8, 0.2, 0.8]
