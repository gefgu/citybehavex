from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl

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
    block = payload["synthetic_vs_random"]
    assert block["comparison"] == "synthetic_vs_random"
    assert block["distributions"]["synthetic"]["clustering_coefficient"]["count"] == 3
    assert block["distributions"]["synthetic"]["edge_persistence"]["count"] == 2
    assert block["distributions"]["synthetic"]["edge_persistence"]["mean"] == 0.75
    assert set(block["wasserstein"]) == {
        "degree",
        "clustering_coefficient",
        "edge_persistence",
        "topological_overlap",
    }
    degree_summary = block["distributions"]["synthetic"]["degree"]
    assert degree_summary["count"] == 3
    np.testing.assert_allclose(degree_summary["mean"], (1 + 2 + 1) / 3)
    assert degree_summary["median"] is not None
    assert degree_summary["std"] is not None
    assert degree_summary["p10"] is not None
    assert degree_summary["p90"] is not None


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
    assert payload["synthetic_vs_random"]["wasserstein"]["edge_persistence"] is None
    assert any("edge persistence unavailable" in warning for warning in warnings)


def test_build_network_validation_can_be_disabled(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)

    payload, warnings = build_network_validation(synthetic, enabled=False)

    assert payload is None
    assert warnings == []


def test_observed_validation_uses_location_day_contacts(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)
    observed = pl.DataFrame(
        {
            "user_id": ["a", "b", "a", "b", "c"],
            "timestamp": pl.Series(
                ["2026-01-01 08:00", "2026-01-01 09:00", "2026-01-02 08:00", "2026-01-02 09:00", "2026-01-02 10:00"]
            ).str.to_datetime(),
            "venueId": ["x", "x", "x", "x", "x"],
            "lat": [0.0] * 5,
            "lon": [0.0] * 5,
        }
    )

    payload, warnings = build_network_validation(
        synthetic,
        observed_df=observed,
        enabled=True,
        synthetic_enabled=False,
        observed_enabled=True,
        location_mode="location_col",
        location_col="venueId",
        seed=7,
    )

    assert payload is not None
    block = payload["observed_vs_random"]
    assert block["comparison"] == "observed_vs_random"
    assert block["distributions"]["observed"]["edge_persistence"]["count"] == 3
    assert block["distributions"]["observed"]["edge_persistence"]["mean"] == 2.0 / 3.0


def test_observed_validation_skips_oversized_groups(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)
    observed = pl.DataFrame(
        {
            "uid": [1, 2, 3],
            "timestamp": pl.Series(["2026-01-01 08:00"] * 3).str.to_datetime(),
            "location_id": ["x", "x", "x"],
        }
    )

    payload, warnings = build_network_validation(
        synthetic,
        observed_df=observed,
        enabled=True,
        synthetic_enabled=False,
        observed_enabled=True,
        max_group_size=2,
        seed=7,
    )

    assert payload is not None
    assert payload["observed_vs_random"]["distributions"]["observed"]["edge_persistence"]["count"] == 0
    assert any("larger than max_group_size=2" in warning for warning in warnings)


def test_observed_validation_uses_h3_contacts(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)
    observed = pl.DataFrame(
        {
            "uid": [1, 2, 3],
            "timestamp": pl.Series(["2026-01-01 08:00", "2026-01-01 09:00", "2026-01-01 10:00"]).str.to_datetime(),
            "lat": [35.0, 35.0, 35.5],
            "lon": [137.0, 137.0, 137.5],
        }
    )

    payload, _warnings = build_network_validation(
        synthetic,
        observed_df=observed,
        enabled=True,
        synthetic_enabled=False,
        observed_enabled=True,
        location_mode="h3",
        h3_resolution=9,
        seed=7,
    )

    assert payload is not None
    assert payload["observed_vs_random"]["source_network"]["edge_count"] == 1
