from __future__ import annotations

from web.backend.app.cache import _key


def test_payload_cache_key_stays_within_filename_limits(tmp_path):
    synthetic = tmp_path / "gparis_simulation_core_trajectories_20260704T104640.parquet"
    observed = tmp_path / "gparis_observed_reference_trajectories_20260704T104640.parquet"
    extras = tuple(
        tmp_path / f"gparis_simulation_core_trajectories_20260704T104640_{suffix}.parquet"
        for suffix in (
            "social_network-1783172814",
            "activities-1783172814",
            "road_graph_nodes-1783171256",
            "road_graph_edges-1783171256",
        )
    )
    for path in (synthetic, observed, *extras):
        path.write_text("x", encoding="utf-8")

    cache_key = _key(
        "gparis_simulation_core",
        "20260704T104640__1783172814__1782927162",
        synthetic,
        observed,
        extras,
    )

    assert cache_key.endswith(".json")
    assert len(cache_key.encode("utf-8")) < 255


def test_payload_cache_key_changes_when_extra_inputs_change(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    observed = tmp_path / "observed.parquet"
    extra_a = tmp_path / "extra-a.parquet"
    extra_b = tmp_path / "extra-b.parquet"
    for path in (synthetic, observed, extra_a, extra_b):
        path.write_text("x", encoding="utf-8")

    key_a = _key("exp", "run", synthetic, observed, (extra_a,))
    key_b = _key("exp", "run", synthetic, observed, (extra_b,))

    assert key_a != key_b
