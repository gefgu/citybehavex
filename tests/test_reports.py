from __future__ import annotations

from types import SimpleNamespace

import h3
import pandas as pd

from citybehavex.reports import (
    _activity_comparison_section_html,
    _activities_sidecar_path,
    _collapse_explicit_purposes,
    _common_part_of_commuters,
    _daily_location_lognormal_dataset,
    _derive_purpose_groups_from_heuristic,
    _metrics_section_html,
    _micro_activity_daily_usage_figure,
    _micro_activity_section_html,
    _mobility_law_visits,
    _mobility_laws_section_html,
    _motif_visits,
    _prepare_activity_visits,
    _trajectory_od_matrix,
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


def test_trajectory_od_matrix_orders_users_and_excludes_invalid_and_self_loops():
    source = pd.DataFrame(
        {
            "uid": ["u1", "u2", "u1", "u1", "u2", "u1"],
            "datetime": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 11:00:00",
                ]
            ),
            "lat": [48.90, 48.90, 48.85, 48.85, 48.85, 999.0],
            "lng": [2.45, 2.45, 2.35, 2.35, 2.35, 2.50],
        }
    )

    matrix = _trajectory_od_matrix(
        source,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        resolution=9,
    )

    origin = h3.latlng_to_cell(48.85, 2.35, 9)
    destination = h3.latlng_to_cell(48.90, 2.45, 9)
    assert matrix.loc[origin, destination] == 2.0
    assert float(matrix.to_numpy().sum()) == 2.0
    assert origin not in matrix.columns


def test_common_part_of_commuters_uses_trajectory_cpc(monkeypatch):
    traj = SimpleNamespace(
        df=pd.DataFrame(
            {
                "uid": [1, 1],
                "datetime": pd.to_datetime(
                    ["2026-01-01 08:00:00", "2026-01-01 09:00:00"]
                ),
                "lat": [48.85, 48.90],
                "lng": [2.35, 2.45],
            }
        ),
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
    )
    calls = []

    def cpc(synthetic_traj, observed_traj, *, resolution):
        calls.append((synthetic_traj, observed_traj, resolution))
        return 0.75

    monkeypatch.setattr(
        "citybehavex.reports.comparison.trajectory_common_part_of_commuters",
        cpc,
    )

    values = _common_part_of_commuters(traj, traj)

    assert values == [(7, 0.75), (8, 0.75), (9, 0.75)]
    assert calls == [(traj, traj, 7), (traj, traj, 8), (traj, traj, 9)]


def test_metrics_section_html_shows_cpc_at_all_resolutions():
    html = _metrics_section_html(
        [("Jump lengths", "1.2345", "km")],
        [("Activity distribution", "0.1234", "")],
        [(7, 0.1), (8, 0.25), (9, 1.0)],
    )

    assert "Common Part of Commuters" in html
    assert "<td>H3 7</td><td>0.1000</td>" in html
    assert "<td>H3 8</td><td>0.2500</td>" in html
    assert "<td>H3 9</td><td>1.0000</td>" in html


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


def test_explicit_purpose_collapse_uses_home_work_other_only():
    visits = pd.DataFrame(
        {
            "purpose": ["HOME", " work ", "SHOP", "PURCHASE", None, "unknown"],
        }
    )

    grouped = _collapse_explicit_purposes(visits)

    assert grouped["purpose"].tolist() == [
        "HOME",
        "WORK",
        "OTHER",
        "OTHER",
        "OTHER",
        "OTHER",
    ]


def test_heuristic_purpose_derivation_uses_time_location_anchors():
    visits = pd.DataFrame(
        {
            "uid": ["u1"] * 5,
            "start_timestamp": pd.to_datetime(
                [
                    "2026-01-01 02:30",
                    "2026-01-01 05:00",
                    "2026-01-01 10:00",
                    "2026-01-01 15:00",
                    "2026-01-01 20:00",
                ]
            ),
            "location_id": ["home", "home", "work", "work", "shop"],
        }
    )

    derived = _derive_purpose_groups_from_heuristic(visits)

    assert derived["purpose"].tolist() == ["HOME", "HOME", "WORK", "WORK", "OTHER"]


def test_prepare_activity_visits_warns_when_using_heuristic():
    source = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pd.to_datetime(
                ["2026-01-01 03:00", "2026-01-01 10:00", "2026-01-01 18:00"]
            ),
            "location_id": ["home", "work", "shop"],
        }
    )

    result = _prepare_activity_visits(
        source,
        label="survey",
        uid_col="uid",
        datetime_col="datetime",
        activity_col=None,
        location_col="location_id",
        lat_col=None,
        lng_col=None,
    )

    assert result is not None
    assert result.used_heuristic is True
    assert "survey has no explicit purpose column" in result.warning
    assert set(result.visits["purpose"]).issubset({"HOME", "WORK", "OTHER"})


def test_activity_comparison_section_uses_comparison_plots(monkeypatch):
    observed_visits = pd.DataFrame({"source": ["observed"]})
    synthetic_visits = pd.DataFrame({"source": ["synthetic"]})
    calls = []

    class Figure:
        def __init__(self, name):
            self.name = name

        def _repr_html_(self):
            return f"<iframe>{self.name}</iframe>"

    def purpose(datasets, **kwargs):
        calls.append(("purpose", datasets))
        return Figure("purpose")

    def transition(first, second, *, labels, **kwargs):
        calls.append(("transition", first, second, labels))
        return Figure("transition")

    def daily(first, second, *, labels, **kwargs):
        calls.append(("daily", first, second, labels))
        return Figure("daily")

    monkeypatch.setattr("citybehavex.reports.comparison.plot_visit_purpose_comparison", purpose)
    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_activity_transition_difference",
        transition,
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_daily_activity_difference",
        daily,
    )

    html = _activity_comparison_section_html(
        observed_visits,
        synthetic_visits,
        "survey",
    )

    assert len(calls) == 3
    assert calls[0][0] == "purpose"
    assert list(calls[0][1]) == ["survey", "synthetic"]
    assert calls[0][1]["survey"] is observed_visits
    assert calls[0][1]["synthetic"] is synthetic_visits
    assert calls[1][0] == "transition"
    assert calls[1][1] is observed_visits
    assert calls[1][2] is synthetic_visits
    assert calls[1][3] == ("survey", "synthetic")
    assert calls[2][0] == "daily"
    assert calls[2][1] is observed_visits
    assert calls[2][2] is synthetic_visits
    assert calls[2][3] == ("survey", "synthetic")
    assert html.count("Activity comparison") == 1
    assert html.count("<iframe>") == 3


def test_activity_comparison_section_shows_heuristic_warning(monkeypatch):
    class Figure:
        def _repr_html_(self):
            return "<iframe>chart</iframe>"

    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_visit_purpose_comparison",
        lambda *args, **kwargs: Figure(),
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_activity_transition_difference",
        lambda *args, **kwargs: Figure(),
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_daily_activity_difference",
        lambda *args, **kwargs: Figure(),
    )

    visits = pd.DataFrame(
        {
            "uid": ["u1"],
            "start_timestamp": pd.to_datetime(["2026-01-01 08:00"]),
            "end_timestamp": pd.to_datetime(["2026-01-01 09:00"]),
            "location_id": ["home"],
            "purpose": ["HOME"],
        }
    )

    html = _activity_comparison_section_html(
        visits,
        visits,
        "survey",
        ["survey has no explicit purpose column; derived HOME/WORK/OTHER."],
    )

    assert "Purpose heuristic warning" in html
    assert "survey has no explicit purpose column" in html


def test_activity_comparison_section_requires_both_datasets():
    visits = pd.DataFrame({"source": ["observed"]})

    assert _activity_comparison_section_html(visits, None, "survey") == ""
    assert _activity_comparison_section_html(None, visits, "survey") == ""


def test_mobility_law_visits_use_existing_locations_or_h3_fallback():
    source = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(
                ["2026-01-01 08:00:00", "2026-01-01 10:00:00"]
            ),
            "lat": [48.85, 48.86],
            "lng": [2.35, 2.36],
            "tile_id": ["home", None],
            "purpose": ["HOME", "WORK"],
        }
    )

    existing = _mobility_law_visits(
        source,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        location_col="tile_id",
        activity_col="purpose",
    )
    fallback = _mobility_law_visits(
        source,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
    )

    assert existing["location_id"].iloc[0] == "home"
    assert existing["location_id"].iloc[1].startswith("8a")
    assert existing["purpose"].tolist() == ["HOME", "WORK"]
    assert fallback["location_id"].str.startswith("8a").all()


def test_daily_location_lognormal_dataset_counts_distinct_locations():
    visits = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u2"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 10:00:00",
                    "2026-01-02 08:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                ]
            ),
            "location_id": ["home", "work", "work", "home", "home", "shop"],
        }
    )

    x_points, probabilities, mu, sigma, label = (
        _daily_location_lognormal_dataset(visits, "observed")
    )

    assert x_points.tolist() == [1.0, 2.0]
    assert probabilities.tolist() == [1 / 3, 2 / 3]
    assert mu > 0
    assert sigma > 0
    assert label == "observed"


def test_mobility_laws_section_renders_all_four_charts(monkeypatch):
    calls = []

    class Figure:
        def __init__(self, name):
            self.name = name

        def _repr_html_(self):
            return f"<iframe>{self.name}</iframe>"

    monkeypatch.setattr(
        "citybehavex.reports.comparison._truncated_powerlaw_dataset",
        lambda values, label: ((1.0, 2.0, 3.0, 4.0), [1.0], [0.5], label),
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison._daily_location_lognormal_dataset",
        lambda visits, label: ([1.0], [0.5], 0.6, 0.7, label),
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison._distance_frequency_dataset",
        lambda visits, label: ([1.0], [0.5], 1.8, 2.5, label),
    )

    def truncated(*datasets, **kwargs):
        calls.append(("truncated", datasets, kwargs))
        return Figure(kwargs["title"])

    def lognormal(*datasets, **kwargs):
        calls.append(("lognormal", datasets, kwargs))
        return Figure("lognormal")

    def distance_frequency(*datasets, **kwargs):
        calls.append(("distance-frequency", datasets, kwargs))
        return Figure("distance-frequency")

    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_truncated_powerlaw_fits",
        truncated,
    )
    monkeypatch.setattr("citybehavex.reports.comparison.plot_lognormal_fits", lognormal)
    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_distance_frequency_law",
        distance_frequency,
    )

    visits = pd.DataFrame({"row": [1, 2]})
    html = _mobility_laws_section_html(
        observed_visits=visits,
        synthetic_visits=visits,
        observed_jumps=[1.0, 2.0],
        synthetic_jumps=[2.0, 3.0],
        observed_rog=[3.0, 4.0],
        synthetic_rog=[4.0, 5.0],
        observed_label="survey",
    )

    assert [call[0] for call in calls] == [
        "truncated",
        "truncated",
        "lognormal",
        "distance-frequency",
    ]
    for _, datasets, _ in calls:
        assert datasets[0][-1] == "survey"
        assert datasets[1][-1] == "synthetic"
    assert calls[1][2]["x_label"] == "radius of gyration · km"
    assert "Mobility laws" in html
    assert html.count("<iframe>") == 4
    assert html.count('class="fit-parameters"') == 4
    assert "c=1" in html
    assert "r0=2" in html
    assert "beta=3" in html
    assert "kappa=4" in html
    assert "mu=0.6" in html
    assert "sigma=0.7" in html
    assert "eta=1.8" in html
    assert "mu=2.5" in html


def test_mobility_laws_section_skips_only_failed_chart(monkeypatch):
    class Figure:
        def _repr_html_(self):
            return "<iframe>chart</iframe>"

    monkeypatch.setattr(
        "citybehavex.reports.comparison._truncated_powerlaw_dataset",
        lambda values, label: ((1.0, 2.0, 3.0, 4.0), [1.0], [0.5], label),
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison._daily_location_lognormal_dataset",
        lambda visits, label: ([1.0], [0.5], 0.6, 0.7, label),
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison._distance_frequency_dataset",
        lambda visits, label: ([1.0], [0.5], 1.8, 2.5, label),
    )

    truncated_calls = 0

    def truncated(*datasets, **kwargs):
        nonlocal truncated_calls
        truncated_calls += 1
        if truncated_calls == 1:
            raise ValueError("insufficient travel distances")
        return Figure()

    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_truncated_powerlaw_fits",
        truncated,
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_lognormal_fits",
        lambda *datasets, **kwargs: Figure(),
    )
    monkeypatch.setattr(
        "citybehavex.reports.comparison.plot_distance_frequency_law",
        lambda *datasets, **kwargs: Figure(),
    )

    visits = pd.DataFrame({"row": [1, 2]})
    html = _mobility_laws_section_html(
        observed_visits=visits,
        synthetic_visits=visits,
        observed_jumps=[1.0, 2.0],
        synthetic_jumps=[2.0, 3.0],
        observed_rog=[3.0, 4.0],
        synthetic_rog=[4.0, 5.0],
        observed_label="survey",
    )

    assert html.count("<iframe>") == 3
    assert html.count('class="fit-parameters"') == 3


def test_activities_sidecar_path_uses_synthetic_stem():
    assert (
        _activities_sidecar_path("data/run/synthetic.parquet")
        == "data/run/synthetic_activities.parquet"
    )


def test_micro_activity_daily_usage_figure_uses_catalog_labels():
    activities = pd.DataFrame(
        {
            "uid": [1, 1],
            "activity": [0, 4],
            "arrival": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00"]),
            "departure": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 09:00"]),
        }
    )

    fig = _micro_activity_daily_usage_figure(activities)

    trace_names = {trace.name for trace in fig.data}
    assert "sleep" in trace_names
    assert "paid_work" in trace_names


def test_micro_activity_section_skips_missing_sidecar(capsys, tmp_path):
    html = _micro_activity_section_html(str(tmp_path / "missing_activities.parquet"))

    assert html == ""
    assert "micro-activity chart skipped" in capsys.readouterr().err


def test_micro_activity_section_skips_empty_sidecar(capsys, tmp_path):
    path = tmp_path / "empty_activities.parquet"
    pd.DataFrame(columns=["uid", "activity", "arrival", "departure"]).to_parquet(
        path,
        index=False,
    )

    html = _micro_activity_section_html(str(path))

    assert html == ""
    assert "activities table is empty" in capsys.readouterr().err
