from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from citybehavex.reports import (
    _motif_visits,
    _visits_for_comparison,
    load_trajectory,
    waiting_times_minutes,
)


def test_waiting_times_minutes_converts_skmob2_seconds():
    traj = SimpleNamespace(
        df=pd.DataFrame(
            {
                "uid": [1, 1, 1],
                "datetime": pd.to_datetime(
                    [
                        "2026-01-01 00:00:00",
                        "2026-01-01 01:00:00",
                        "2026-01-01 03:00:00",
                    ]
                ),
                "lat": [48.85, 48.86, 48.87],
                "lng": [2.35, 2.36, 2.37],
            }
        ),
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )

    assert waiting_times_minutes(traj) == [60.0, 120.0]


def test_load_trajectory_detects_common_column_names(tmp_path):
    path = tmp_path / "trajectories.parquet"
    pd.DataFrame(
        {
            "user_id": ["u1"],
            "start_timestamp": pd.to_datetime(["2026-01-01 08:00:00"]),
            "latitude": [48.85],
            "longitude": [2.35],
        }
    ).to_parquet(path, index=False)

    traj = load_trajectory(str(path))

    assert traj.uid_col == "user_id"
    assert traj.datetime_col == "start_timestamp"
    assert traj.lat_col == "latitude"
    assert traj.lng_col == "longitude"


def test_motif_visits_use_h3_locations_and_binary_purposes():
    source = pd.DataFrame(
        {
            "uid": [1, 1],
            "datetime": pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 08:00:00"]
            ),
            "lat": [48.85, 48.86],
            "lng": [2.35, 2.36],
            "purpose": ["HOME", "WORK"],
        }
    )

    visits = _visits_for_comparison(
        source,
        uid_col="uid",
        datetime_col="datetime",
        activity_col="purpose",
        location_resolution=10,
    )
    motif_visits = _motif_visits(visits)

    assert motif_visits["location_id"].str.startswith("8a").all()
    assert motif_visits["purpose"].tolist() == ["HOME", "VISIT"]
