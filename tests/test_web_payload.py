from __future__ import annotations

import json

import pandas as pd

from citybehavex.simulation.core import social_network_sidecar_path
from web.backend.app.payload import _load_social_network_sidecar


def test_load_social_network_sidecar_returns_none_when_absent(tmp_path):
    synthetic = tmp_path / "trajectories_20260101T010203.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)

    assert _load_social_network_sidecar(str(synthetic)) is None


def test_load_social_network_sidecar_validates_and_returns_payload(tmp_path):
    synthetic = tmp_path / "trajectories_20260101T010203.parquet"
    pd.DataFrame({"uid": [1]}).to_parquet(synthetic, index=False)
    sidecar = social_network_sidecar_path(synthetic)
    payload = {
        "kind": "initial_profile_similarity",
        "node_count": 2,
        "edge_count": 1,
        "layout": "profile_svd",
        "directed": True,
        "social_graph_k": 1,
        "nodes": [[0.0, 0.0, 8.0, 1, "worker"], [1.0, 1.0, 8.0, 2, "student"]],
        "edges": [[0, 1, 0.75]],
        "degrees": [1, 0],
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_social_network_sidecar(str(synthetic)) == payload
