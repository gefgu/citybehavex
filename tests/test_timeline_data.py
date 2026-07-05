from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from web.backend.app.timeline_data import (
    _build_legs_index,
    _build_moving_index,
    group_trips_by_location,
    query_active_legs,
    query_activity_at_stop,
    query_agent_crp,
    query_agent_encounter_counts,
    query_agent_social_friends,
    query_agent_trips,
    query_stop_activities,
)


def _trajectory(category: bool) -> pd.DataFrame:
    rows = {
        "uid": [1, 1],
        "stop_id": [0, 1],
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


def test_query_agent_crp_filters_by_agent_and_sorts_by_usage(tmp_path):
    path = tmp_path / "run_crp.parquet"
    pd.DataFrame(
        {
            "agent": [0, 0, 0, 1, 1],
            "diary_id": ["wd-0", "wd-1", "we-0", "wd-0", "wd-1"],
            "day_type": ["weekday", "weekday", "weekend", "weekday", "weekday"],
            "sim": [0.2, 0.9, 0.5, 0.1, 0.8],
            "usage_count": [1, 5, 2, 3, 3],
            "T_a": [0.31, 0.31, 0.31, 0.5, 0.5],
            "alpha_a": [0.15, 0.15, 0.15, 0.2, 0.2],
        }
    ).to_parquet(path, index=False)

    rows = query_agent_crp(path, 0)

    assert [r["diary_id"] for r in rows] == ["wd-1", "we-0", "wd-0"]
    assert rows[0]["T_a"] == 0.31 and rows[0]["alpha_a"] == 0.15
    assert rows[1]["day_type"] == "weekend"


def test_query_agent_social_friends_maps_zero_based_graph_to_display_uids(tmp_path):
    path = tmp_path / "run_social_network.json"
    path.write_text(
        json.dumps(
            {
                "kind": "initial_profile_similarity",
                "node_count": 3,
                "edge_count": 3,
                "layout": "profile_svd",
                "directed": True,
                "social_graph_k": 2,
                "nodes": [[0, 0, 4, 1], [1, 0, 4, 2], [2, 0, 4, 3]],
                "edges": [[0, 1, 0.8], [0, 2, 0.4], [1, 0, 0.7]],
                "degrees": [2, 1, 0],
            }
        ),
        encoding="utf-8",
    )

    params, friends, warnings = query_agent_social_friends(path, 1)

    assert warnings == []
    assert params["degree"] == 2
    assert params["total_social_strength"] == pytest.approx(1.2)
    assert [friend["uid"] for friend in friends] == [2, 3]
    assert friends[0]["social_strength"] == 0.8
    assert friends[0]["embedding_similarity"] == 0.8
    assert friends[0]["reciprocated"] is True
    assert friends[1]["reciprocated"] is False


def test_query_agent_encounter_counts_counts_both_directions(tmp_path):
    path = tmp_path / "run_encounters.parquet"
    pd.DataFrame(
        {
            "agent": [1, 2, 1, 3],
            "contact": [2, 1, 3, 1],
            "tile": [10, 10, 11, 11],
            "ts": [1, 2, 3, 4],
        }
    ).to_parquet(path, index=False)

    counts = query_agent_encounter_counts(path, 1)

    assert counts == {2: 2, 3: 2}


def test_query_agent_trips_returns_null_category_for_legacy_runs(tmp_path):
    path = tmp_path / "trajectory.parquet"
    _trajectory(category=False).to_parquet(path, index=False)

    trips = query_agent_trips(path, 1)

    assert trips[0]["category"] is None
    assert trips[1]["purpose"] == "OTHER"


def test_group_trips_by_location_merges_zero_travel_continuations():
    # Simulates a workday split by the engine into several zero-travel stops
    # at the same tile, each with its own (possibly inconsistent) purpose,
    # followed by a real trip elsewhere.
    trips = [
        {
            "arrival": datetime(2026, 1, 1, 9, 0),
            "departure": datetime(2026, 1, 1, 10, 0),
            "lat": 48.85,
            "lng": 2.35,
            "purpose": "WORK",
            "category": "office",
            "activity": 1,
            "trip_duration_minutes": 5.0,
            "dwell_minutes": 60.0,
        },
        {
            "arrival": datetime(2026, 1, 1, 10, 0),
            "departure": datetime(2026, 1, 1, 12, 0),
            "lat": 48.85,
            "lng": 2.35,
            "purpose": "OTHER",
            "category": "office",
            "activity": 2,
            "trip_duration_minutes": 0.0,
            "dwell_minutes": 120.0,
        },
        {
            "arrival": datetime(2026, 1, 1, 12, 5),
            "departure": datetime(2026, 1, 1, 12, 30),
            "lat": 48.90,
            "lng": 2.40,
            "purpose": "OTHER",
            "category": "cafe",
            "activity": 3,
            "trip_duration_minutes": 5.0,
            "dwell_minutes": 25.0,
        },
    ]

    grouped = group_trips_by_location(trips)

    assert len(grouped) == 2
    first = grouped[0]
    assert first["arrival"] == datetime(2026, 1, 1, 9, 0)
    assert first["departure"] == datetime(2026, 1, 1, 12, 0)
    assert first["purpose"] == "WORK"
    assert first["dwell_minutes"] == 180.0
    assert len(first["activities"]) == 2
    assert grouped[1]["lat"] == 48.90
    assert len(grouped[1]["activities"]) == 1


def _activities(tmp_path):
    path = tmp_path / "activities.parquet"
    pd.DataFrame(
        {
            "uid": [1, 1, 1],
            "stop_id": [0, 0, 1],
            "seq": [0, 1, 0],
            "activity": [0, 4, 9],
            "arrival": pd.to_datetime(
                ["2026-01-01 08:00", "2026-01-01 08:30", "2026-01-01 09:00"]
            ),
            "departure": pd.to_datetime(
                ["2026-01-01 08:30", "2026-01-01 09:00", "2026-01-01 10:00"]
            ),
        }
    ).to_parquet(path, index=False)
    return path


def test_query_stop_activities_groups_by_stop_id(tmp_path):
    path = _activities(tmp_path)

    by_stop = query_stop_activities(path, 1)

    assert set(by_stop.keys()) == {0, 1}
    assert [a["seq"] for a in by_stop[0]] == [0, 1]
    assert [a["activity"] for a in by_stop[0]] == [0, 4]
    assert len(by_stop[1]) == 1


def test_query_stop_activities_scopes_to_requested_uid(tmp_path):
    path = tmp_path / "activities.parquet"
    pd.DataFrame(
        {
            "uid": [1, 2],
            "stop_id": [0, 0],
            "seq": [0, 0],
            "activity": [0, 1],
            "arrival": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 08:00"]),
            "departure": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:00"]),
        }
    ).to_parquet(path, index=False)

    by_stop = query_stop_activities(path, 2)

    assert set(by_stop.keys()) == {0}
    assert by_stop[0][0]["activity"] == 1


def test_query_activity_at_stop_picks_the_window_containing_ts(tmp_path):
    path = _activities(tmp_path)

    hit = query_activity_at_stop(path, 1, 0, pd.Timestamp("2026-01-01 08:45").to_pydatetime())
    assert hit is not None
    assert hit["activity"] == 4  # the second activity within stop 0, [08:30, 09:00)

    miss = query_activity_at_stop(path, 1, 0, pd.Timestamp("2026-01-01 07:00").to_pydatetime())
    assert miss is None


def test_timeline_legs_index_preserves_category(tmp_path):
    trajectory_path = tmp_path / "trajectory.parquet"
    legs_path = tmp_path / "legs.parquet"
    _trajectory(category=True).to_parquet(trajectory_path, index=False)

    _build_legs_index(trajectory_path, legs_path)

    legs = pd.read_parquet(legs_path)
    assert "category" in legs.columns
    assert "cafe" in set(legs["category"].dropna())


def test_query_active_legs_attaches_moving_waypoints(tmp_path):
    trajectory_path = tmp_path / "trajectory.parquet"
    moving_raw_path = tmp_path / "moving_raw.parquet"
    legs_path = tmp_path / "legs.parquet"
    moving_path = tmp_path / "moving.parquet"
    _trajectory(category=True).to_parquet(trajectory_path, index=False)
    pd.DataFrame(
        {
            "uid": [1, 1, 1],
            "stop_id": [1, 1, 1],
            "seq": [0, 1, 2],
            "lat": [48.8566, 48.8572, 48.8580],
            "lng": [2.3522, 2.3531, 2.3540],
            "t": pd.to_datetime(
                ["2026-01-01 08:50", "2026-01-01 08:55", "2026-01-01 09:00"]
            ),
        }
    ).to_parquet(moving_raw_path, index=False)

    _build_legs_index(trajectory_path, legs_path)
    _build_moving_index(moving_raw_path, moving_path)

    segments, truncated = query_active_legs(
        legs_path,
        pd.Timestamp("2026-01-01 08:45").to_pydatetime(),
        pd.Timestamp("2026-01-01 09:05").to_pydatetime(),
        (48.85, 2.35, 48.86, 2.36),
        10,
        moving_path,
    )

    leg = next(s for s in segments if s["kind"] == "leg")
    assert truncated is False
    assert [w["lng"] for w in leg["waypoints"]] == [2.3522, 2.3531, 2.3540]
