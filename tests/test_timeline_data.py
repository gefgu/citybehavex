from __future__ import annotations

import pandas as pd

from web.backend.app.timeline_data import _build_legs_index, query_agent_trips


def _trajectory(category: bool) -> pd.DataFrame:
    rows = {
        "uid": [1, 1],
        "datetime": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00"]),
        "lat": [48.8566, 48.8580],
        "lng": [2.3522, 2.3540],
        "arrival": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00"]),
        "departure": pd.to_datetime(["2026-01-01 08:30", "2026-01-01 10:00"]),
        "trip_duration_minutes": [0.0, 10.0],
        "dwell_minutes": [30.0, 60.0],
        "purpose": ["HOME", "OTHER"],
    }
    if category:
        rows["category"] = [None, "cafe"]
    return pd.DataFrame(rows)


def test_query_agent_trips_returns_null_category_for_legacy_runs(tmp_path):
    path = tmp_path / "trajectory.parquet"
    _trajectory(category=False).to_parquet(path, index=False)

    trips = query_agent_trips(path, 1)

    assert trips[0]["category"] is None
    assert trips[1]["purpose"] == "OTHER"


def test_timeline_legs_index_preserves_category(tmp_path):
    trajectory_path = tmp_path / "trajectory.parquet"
    legs_path = tmp_path / "legs.parquet"
    _trajectory(category=True).to_parquet(trajectory_path, index=False)

    _build_legs_index(trajectory_path, legs_path)

    legs = pd.read_parquet(legs_path)
    assert "category" in legs.columns
    assert "cafe" in set(legs["category"].dropna())
