from __future__ import annotations

import json

import numpy as np
import pandas as pd

from citybehavex.reports.network_validation import (
    build_network_validation,
    clustering_coefficients,
    encounters_sidecar_path,
    graph_from_edges,
    social_network_sidecar_path,
    topological_overlap,
)


def test_clustering_coefficients_are_per_node():
    graph = graph_from_edges(4, {(0, 1), (1, 2), (0, 2)})

    values = clustering_coefficients(graph)

    np.testing.assert_allclose(values, [1.0, 1.0, 1.0, 0.0])


def test_topological_overlap_is_per_edge_jaccard():
    graph = graph_from_edges(4, {(0, 1), (1, 2), (0, 2), (2, 3)})

    values = dict(zip(sorted(graph.edges), topological_overlap(graph)))

    assert values[(0, 1)] == 1.0 / 3.0
    assert values[(2, 3)] == 0.0
    assert values[(0, 2)] == 0.25


def test_build_network_validation_computes_distribution_wasserstein(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)
    social_network_sidecar_path(synthetic).write_text(
        json.dumps(
            {
                "kind": "initial_profile_similarity",
                "node_count": 3,
                "edge_count": 2,
                "layout": "profile_svd",
                "directed": True,
                "social_graph_k": 2,
                "nodes": [[0.0, 0.0, 8.0, 1], [1.0, 0.0, 8.0, 2], [0.0, 1.0, 8.0, 3]],
                "edges": [[0, 1, 1.0], [1, 2, 1.0]],
                "degrees": [1, 2, 1],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "agent": [0, 0, 1, 1],
            "contact": [1, 1, 2, 2],
            "tile": [10, 10, 11, 11],
            "ts": [1, 2, 1, 1],
        }
    ).to_parquet(encounters_sidecar_path(synthetic), index=False)

    payload, warnings = build_network_validation(synthetic, seed=7)

    assert payload is not None
    assert payload["comparison"] == "synthetic_vs_random"
    assert payload["distributions"]["synthetic"]["clustering_coefficient"]["count"] == 3
    assert payload["distributions"]["synthetic"]["edge_persistence"]["count"] == 2
    assert payload["distributions"]["synthetic"]["edge_persistence"]["mean"] == 0.75
    assert set(payload["wasserstein"]) == {
        "clustering_coefficient",
        "edge_persistence",
        "topological_overlap",
    }


def test_build_network_validation_marks_persistence_unavailable_without_encounters(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)
    social_network_sidecar_path(synthetic).write_text(
        json.dumps(
            {
                "kind": "initial_profile_similarity",
                "node_count": 2,
                "edge_count": 1,
                "layout": "profile_svd",
                "directed": True,
                "social_graph_k": 1,
                "nodes": [[0.0, 0.0, 8.0, 1], [1.0, 0.0, 8.0, 2]],
                "edges": [[0, 1, 1.0]],
                "degrees": [1, 1],
            }
        ),
        encoding="utf-8",
    )

    payload, warnings = build_network_validation(synthetic, seed=7)

    assert payload is not None
    assert payload["wasserstein"]["edge_persistence"] is None
    assert any("edge persistence unavailable" in warning for warning in warnings)
