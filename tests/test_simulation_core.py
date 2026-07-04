from __future__ import annotations

import numpy as np
import pandas as pd
import h3

import citybehavex._core as core
from citybehavex.activities import (
    N_ACTIVITIES,
    activity_duration_arrays,
    build_eligibility_csr,
)
from citybehavex.config.root import CityBehavExConfig
from citybehavex.profiles import generate_profiles
from citybehavex.profiles.config import AgentProfilesConfig
from citybehavex.simulation.core import simulate_agents
from citybehavex.simulation.core import build_social_graph_artifact
from citybehavex.simulation.runner import (
    _append_home_anchors,
    _append_work_scores,
    _derive_home_anchor_candidates_from_tessellation,
    _home_anchors_output_path,
)

_SLOT = 900
_SPEED = 50.0


def test_build_social_graph_artifact_packs_csr_graph():
    artifact = build_social_graph_artifact(
        np.array([0, 2, 3, 3], dtype=np.int64),
        np.array([1, 2, 2], dtype=np.int64),
        np.array([0.9, 0.4, 0.8], dtype=np.float64),
        n_agents=3,
        random_state=7,
        social_graph_k=2,
        profile_embeddings=np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float64),
        profile_types=["worker", "student", "retired"],
    )
    payload = artifact.to_dict()

    assert payload["node_count"] == 3
    assert payload["edge_count"] == 3
    assert payload["layout"] == "profile_svd"
    assert payload["degrees"] == [2, 1, 0]
    assert payload["nodes"][0][3:] == [1, "worker"]
    assert payload["edges"][0] == [0, 1, 0.9]


def _run(
    lats,
    lngs,
    abs_locs,
    slot_times,
    *,
    end_ts,
    rho=1.0,
    gamma=0.21,
    relevances=None,
    act_dur_mu=None,
    act_dur_sigma=None,
    purpose_act_starts=None,
    purpose_acts=None,
    work_tile=1,
):
    diary_ts = np.asarray(slot_times, dtype=np.int64)
    diary_loc = np.asarray(abs_locs, dtype=np.int32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([len(diary_ts)], dtype=np.int64)
    rels = np.ones(len(lats), dtype=float) if relevances is None else np.asarray(relevances, dtype=float)
    # Returns a 3-tuple of tuples: (11 trip arrays), (7 path arrays), (6 activity arrays).
    # Trip: agents, lats, lngs, arrival, departure, duration,
    #       enc_agent, enc_contact, enc_tile, enc_ts, stop_abstract_loc
    # Paths: stop_id, path_agent, path_stop_id, path_seq, path_lat, path_lng, path_t
    # Activities: act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure
    return core.simulation_core_simulate_agents(
        latitudes=np.asarray(lats, dtype=float),
        longitudes=np.asarray(lngs, dtype=float),
        relevances=rels,
        distances=np.empty(0, dtype=np.float64),
        neighbor_starts=np.array([0, 0], dtype=np.int64),
        neighbors=np.empty(0, dtype=np.int64),
        diary_timestamps=diary_ts,
        diary_abs_locs=diary_loc,
        diary_starts=starts,
        diary_ends=ends,
        rho=rho,
        gamma=gamma,
        alpha=0.0,
        start_ts=0,
        end_ts=end_ts,
        indipendency_window_s=1800,
        dt_update_mob_sim_s=3600,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=1,
        master_seed=42,
        starting_locs=np.array([0], dtype=np.int64),
        starting_locs_mode_relevance=False,
        work_tiles=np.array([work_tile], dtype=np.int64),
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )


def test_simulation_core_long_trip_is_centered_on_slot_boundary():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    trip, _, _ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    ag, _, _, arr, dep, dur, *_ = trip

    assert len(ag) == 3
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)
    assert np.all(np.diff(arr) >= 0)
    assert np.all((dep - arr) >= 0)
    assert dur[1] > _SLOT
    assert dep[0] < 8 * 3600
    assert arr[1] > 8 * 3600
    assert abs((8 * 3600 - dep[0]) - (arr[1] - 8 * 3600)) <= 1


def test_simulation_core_short_trip_arrives_within_the_slot():
    lats = [48.8566, 48.8580]
    lngs = [2.3522, 2.3540]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    trip, _, _ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    _, _, _, arr, dep, dur, *_ = trip
    arr, dep, dur = np.asarray(arr), np.asarray(dep), np.asarray(dur)

    assert dur[1] < _SLOT
    assert dep[0] == 8 * 3600
    assert 8 * 3600 <= arr[1] < 8 * 3600 + _SLOT


def test_simulation_core_trip_durations_are_off_the_hourly_grid():
    lats = [48.8566, 48.95]
    lngs = [2.3522, 2.55]
    slot_times = [0, 8 * 3600, 18 * 3600]
    abs_locs = [0, 1, 0]
    trip, _, _ = _run(lats, lngs, abs_locs, slot_times, end_ts=86400)
    _, _, _, arr, *_ = trip
    assert any(int(a) % _SLOT != 0 for a in np.asarray(arr))


def test_simulation_core_keeps_one_location_for_continuous_abstract_block():
    lats = [48.8566, 48.8580, 48.8610, 48.8640]
    lngs = [2.3522, 2.3540, 2.3580, 2.3620]
    slot_times = [0, 8 * 3600, 8 * 3600 + _SLOT, 8 * 3600 + 2 * _SLOT, 18 * 3600]
    abs_locs = [0, 1, 1, 1, 0]

    trip, _, _ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=1.0,
        gamma=0.0,
    )

    ag = trip[0]
    assert len(ag) == 3


def test_work_code_always_resolves_to_fixed_work_tile():
    """WORK (abstract code 1) is pinned to a single tile per agent for the
    whole simulation -- every occurrence of the WORK code, even non-adjacent
    ones on the same day, must resolve to that same fixed tile."""
    lats = [48.8566, 48.8580, 48.8610, 48.8640]
    lngs = [2.3522, 2.3540, 2.3580, 2.3620]
    slot_times = [
        0,
        8 * 3600,
        8 * 3600 + _SLOT,
        10 * 3600,
        14 * 3600,
        14 * 3600 + _SLOT,
        18 * 3600,
    ]
    abs_locs = [0, 1, 1, 0, 1, 1, 0]

    trip, _, _ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=1.0,
        gamma=0.0,
    )

    out_lats = np.asarray(trip[1])
    out_lngs = np.asarray(trip[2])
    assert len(out_lats) == 5
    assert out_lats[1] == out_lats[3]
    assert out_lngs[1] == out_lngs[3]


def test_other_code_does_not_reuse_a_cached_location_within_a_day():
    """OTHER abstract codes (2-6) must resolve fresh via the EPR return/
    exploration decision every time, never memoized per day. With rho=1.0
    (always explore), exploration excludes already-visited tiles, so two
    non-adjacent occurrences of the same OTHER code must land on different
    physical tiles -- a per-day cache would have forced them to match."""
    lats = [48.8566, 48.8580, 48.8610, 48.8640]
    lngs = [2.3522, 2.3540, 2.3580, 2.3620]
    slot_times = [
        0,
        8 * 3600,
        8 * 3600 + _SLOT,
        10 * 3600,
        14 * 3600,
        14 * 3600 + _SLOT,
        18 * 3600,
    ]
    abs_locs = [0, 2, 2, 0, 2, 2, 0]

    trip, _, _ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=1.0,
        gamma=0.0,
    )

    out_lats = np.asarray(trip[1])
    out_lngs = np.asarray(trip[2])
    assert len(out_lats) == 5
    assert out_lats[1] != out_lats[3] or out_lngs[1] != out_lngs[3]


def test_same_physical_location_across_abstract_codes_yields_one_stop():
    """Different abstract-location codes that resolve (e.g. via preferential
    return, rho=0) to the agent's *current* physical tile must not fragment
    the stop table -- a new stop row only appears on a real relocation."""
    lats = [48.8566, 48.8580, 48.8700]
    lngs = [2.3522, 2.3540, 2.4000]
    slot_times = [0, 8 * 3600, 12 * 3600, 18 * 3600]
    abs_locs = [0, 1, 2, 0]

    trip, _, _ = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=0.0,
        gamma=0.0,
        work_tile=0,
    )
    ag, out_lats, _, arr, dep, *_ = trip

    assert len(ag) == 1
    assert arr[0] == 0
    assert dep[0] == 86400
    assert out_lats[0] == lats[0]


def test_same_physical_location_still_samples_multiple_activities():
    """Even though the stop table collapses to one row, each abstract-
    location change that lands back on the same tile should still record its
    own micro-activity in the separate activities table, in order, with
    non-overlapping [arrival, departure) windows spanning the whole stay."""
    lats = [48.8566, 48.8580, 48.8700]
    lngs = [2.3522, 2.3540, 2.4000]
    slot_times = [0, 8 * 3600, 12 * 3600, 18 * 3600]
    abs_locs = [0, 1, 2, 0]
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    purpose_act_starts, purpose_acts = build_eligibility_csr()

    trip, _, acts = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=86400,
        rho=0.0,
        gamma=0.0,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
        work_tile=0,
    )
    ag = trip[0]
    assert len(ag) == 1  # still one physical stop

    act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure = (
        np.asarray(a) for a in acts
    )
    assert len(act_agent) >= 4
    assert (act_stop_id == 0).all()
    assert list(act_seq) == list(range(len(act_seq)))
    assert (act_activity >= 0).all()
    assert (act_activity < N_ACTIVITIES).all()
    # Contiguous, non-overlapping, covering the whole simulated window.
    assert act_arrival[0] == 0
    assert act_departure[-1] == 86400
    assert list(act_arrival[1:]) == list(act_departure[:-1])
    assert (act_departure >= act_arrival).all()
    assert 8 * 3600 in set(act_arrival)
    assert 12 * 3600 in set(act_arrival)
    assert 18 * 3600 in set(act_arrival)


def test_simulate_agents_returns_trip_columns():
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "lat": [48.8566, 48.95],
            "lng": [2.3522, 2.55],
            "relevance": [1.0, 1.0],
        }
    )
    diary_arrays = (
        np.array([0, 8 * 3600, 18 * 3600], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([3], dtype=np.int64),
    )
    df, encounters, moving, activities = simulate_agents(
        tess,
        "relevance",
        diary_arrays,
        start_ts=0,
        end_ts=86400,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=1,
        random_state=42,
    )
    for column in (
        "uid",
        "datetime",
        "lat",
        "lng",
        "arrival",
        "departure",
        "trip_duration_minutes",
        "dwell_minutes",
        "purpose",
    ):
        assert column in df.columns
    assert "activity" not in df.columns
    assert (df["dwell_minutes"] >= 0).all()
    assert (df["trip_duration_minutes"] >= 0).all()
    assert pd.api.types.is_datetime64_any_dtype(df["arrival"])
    assert isinstance(moving, pd.DataFrame)
    assert isinstance(activities, pd.DataFrame)


def test_simulate_agents_purpose_uses_engine_abstract_location_not_arrival_window():
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "lat": [48.8566, 48.95],
            "lng": [2.3522, 2.55],
            "relevance": [1.0, 1.0],
        }
    )
    diary_arrays = (
        np.array([0, 8 * 3600, 8 * 3600 + _SLOT], dtype=np.int64),
        np.array([0, 2, 0], dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([3], dtype=np.int64),
    )

    df, _, _, _ = simulate_agents(
        tess,
        "relevance",
        diary_arrays,
        start_ts=0,
        end_ts=10 * 3600,
        slot_seconds=_SLOT,
        car_speed_kmh=10.0,
        n_agents=1,
        random_state=42,
        rho=1.0,
        gamma=0.0,
        starting_locs=np.array([0], dtype=np.int64),
    )

    other_stop = df.iloc[1]
    assert other_stop["datetime"] >= pd.Timestamp("1970-01-01 08:15:00")
    assert other_stop["purpose"] == "OTHER"


def test_simulate_agents_can_return_social_graph_artifact():
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "lat": [48.8566, 48.95],
            "lng": [2.3522, 2.55],
            "relevance": [1.0, 1.0],
        }
    )
    diary_arrays = (
        np.array([0, 8 * 3600, 18 * 3600], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int64),
        np.array([3, 3], dtype=np.int64),
    )
    default_result = simulate_agents(
        tess,
        "relevance",
        diary_arrays,
        start_ts=0,
        end_ts=86400,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=2,
        random_state=42,
        social_graph_k=1,
    )
    assert len(default_result) == 4

    result = simulate_agents(
        tess,
        "relevance",
        diary_arrays,
        start_ts=0,
        end_ts=86400,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=2,
        random_state=42,
        social_graph_k=1,
        return_social_graph=True,
        social_node_profiles=["worker", "student"],
    )
    assert len(result) == 5
    artifact = result[4].to_dict()
    assert artifact["node_count"] == 2
    assert artifact["edge_count"] == 2
    assert artifact["nodes"][0][4] == "worker"


def test_simulate_agents_encounters_has_expected_columns():
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1],
            "lat": [48.8566, 48.95],
            "lng": [2.3522, 2.55],
            "relevance": [1.0, 1.0],
        }
    )
    diary_arrays = (
        np.array([0, 8 * 3600, 18 * 3600], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int64),
        np.array([3, 3], dtype=np.int64),
    )
    _, encounters, _, _ = simulate_agents(
        tess,
        "relevance",
        diary_arrays,
        start_ts=0,
        end_ts=86400,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=2,
        random_state=42,
    )
    assert isinstance(encounters, pd.DataFrame)
    for col in ("agent", "contact", "tile", "ts"):
        assert col in encounters.columns


# --- Phase 4: activity CRP + early/late exit timing -------------------------

def _diary_arrays_single(abs_locs, slot_times):
    return (
        np.asarray(slot_times, dtype=np.int64),
        np.asarray(abs_locs, dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([len(slot_times)], dtype=np.int64),
    )


def test_activity_column_present_when_enabled():
    """activities.enabled -> a non-empty activities DataFrame is returned,
    with the stop table itself left untouched (no inline activity column)."""
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.95],
        "lng": [2.3522, 2.55],
        "relevance": [1.0, 1.0],
    })
    diary_arrays = _diary_arrays_single([0, 1, 0], [0, 8 * 3600, 18 * 3600])
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    df, _, _, activities = simulate_agents(
        tess, "relevance", diary_arrays,
        start_ts=0, end_ts=86400,
        slot_seconds=_SLOT, car_speed_kmh=_SPEED,
        n_agents=1, random_state=42,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )
    assert "activity" not in df.columns
    assert len(activities) > 0
    assert (activities["activity"] >= 0).all()
    assert (activities["activity"] < N_ACTIVITIES).all()
    assert set(activities["stop_id"]).issubset(set(range(len(df))))


def test_activities_chain_until_macro_departure_deadline():
    lats = [48.8566, 48.8566]
    lngs = [2.3522, 2.3522]
    slot_times = [0, 7 * 3600, 12 * 3600]
    abs_locs = [0, 1, 0]
    ten_min_hours = 10 / 60
    act_dur_mu = np.log(np.array([ten_min_hours, ten_min_hours, ten_min_hours], dtype=np.float64))
    act_dur_sigma = np.zeros(3, dtype=np.float64)
    purpose_act_starts = np.array([0, 1, 2, 3], dtype=np.int64)
    purpose_acts = np.array([0, 1, 2], dtype=np.int64)

    trip, _, acts = _run(
        lats,
        lngs,
        abs_locs,
        slot_times,
        end_ts=14 * 3600,
        rho=1.0,
        gamma=0.0,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )

    _, _, _, arr, dep, dur, *_ = (np.asarray(a) for a in trip)
    assert len(arr) == 3
    assert dep[0] == 7 * 3600 - 900
    assert arr[1] == 7 * 3600 - 900
    assert dur[1] == 0

    act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure = (
        np.asarray(a) for a in acts
    )
    first_stop = act_stop_id == 0
    assert first_stop.sum() > 1
    assert act_arrival[first_stop][0] == 0
    assert act_departure[first_stop][-1] == 7 * 3600 - 900
    assert list(act_arrival[first_stop][1:]) == list(act_departure[first_stop][:-1])
    assert act_departure[first_stop][-1] - act_arrival[first_stop][-1] == 5 * 60


def test_home_anchors_are_appended_and_profiles_use_only_home_pool(tmp_path):
    anchor_path = tmp_path / "home_anchors.parquet"
    pd.DataFrame({"lat": [48.1, 48.2, 48.3], "lng": [2.1, 2.2, 2.3]}).to_parquet(anchor_path, index=False)
    tess = pd.DataFrame(
        {
            "tile_id": ["poi_1", "poi_2"],
            "lat": [48.8, 48.9],
            "lng": [2.3, 2.4],
            "category": ["restaurant", "corporate_office"],
            "purpose": ["OTHER", "WORK"],
            "relevance": [10.0, 20.0],
        }
    )
    config = CityBehavExConfig.model_validate(
        {
            "simulation": {"agents": 3, "random_state": 7},
            "profiles": {"enabled": True, "home_anchors_path": str(anchor_path)},
            "road_network": {"enabled": False},
        }
    )

    augmented, home_pool = _append_home_anchors(config, tess, "relevance")
    assert home_pool.tolist() == [2, 3, 4]
    assert augmented.loc[home_pool, "category"].tolist() == ["residential"] * 3
    assert augmented.loc[home_pool, "purpose"].tolist() == ["HOME"] * 3

    profiles = generate_profiles(
        20,
        AgentProfilesConfig(enabled=True),
        np.random.default_rng(3),
        augmented,
        "relevance",
        home_tile_pool=home_pool,
    )
    assert {p.home_tile for p in profiles}.issubset(set(home_pool.tolist()))
    assert {p.work_tile for p in profiles}.issubset({0, 1})


def test_poi_building_work_scores_favor_high_poi_and_building_cells(tmp_path):
    resolution = 8
    low_lat, low_lng = 48.8566, 2.3522
    high_lat, high_lng = 48.92, 2.48
    high_cell = h3.latlng_to_cell(high_lat, high_lng, resolution)
    building_path = tmp_path / "buildings.parquet"
    pd.DataFrame({"h3_cell": [high_cell], "building_count": [25]}).to_parquet(building_path, index=False)
    tess = pd.DataFrame(
        {
            "tile_id": ["low", "high"],
            "lat": [low_lat, high_lat],
            "lng": [low_lng, high_lng],
            "purpose": ["OTHER", "WORK"],
            "relevance": [1.0, 10.0],
        }
    )
    config = CityBehavExConfig.model_validate(
        {
            "tessellation": {"resolution": resolution},
            "profiles": {
                "enabled": True,
                "overture_building_features_path": str(building_path),
            },
            "road_network": {"enabled": False},
        }
    )

    enriched, relevance_column = _append_work_scores(config, tess, "relevance")

    assert relevance_column == "work_score"
    assert "building_count" in enriched.columns
    assert enriched.loc[1, "building_count"] == 25
    assert enriched.loc[1, "work_score"] > enriched.loc[0, "work_score"]


def test_poi_building_home_anchors_use_cached_buildings(tmp_path):
    resolution = 8
    min_lng, min_lat, max_lng, max_lat = 2.30, 48.82, 2.53, 48.95
    boundary = h3.LatLngPoly(
        [
            (min_lat, min_lng),
            (min_lat, max_lng),
            (max_lat, max_lng),
            (max_lat, min_lng),
        ]
    )
    cells = sorted(h3.polygon_to_cells(boundary, resolution))
    building_cell = cells[len(cells) // 2]
    poi_cell = cells[0] if cells[0] != building_cell else cells[-1]
    building_lat, building_lng = h3.cell_to_latlng(building_cell)
    poi_lat, poi_lng = h3.cell_to_latlng(poi_cell)
    building_path = tmp_path / "buildings.parquet"
    pd.DataFrame({"h3_cell": [building_cell], "building_count": [100]}).to_parquet(building_path, index=False)
    tess = pd.DataFrame(
        {
            "tile_id": ["poi_dense", "residential_like"],
            "lat": [poi_lat, building_lat],
            "lng": [poi_lng, building_lng],
            "relevance": [100.0, 0.0],
        }
    )
    config = CityBehavExConfig.model_validate(
        {
            "simulation": {
                "agents": 200,
                "random_state": 11,
                "min_lon": min_lng,
                "min_lat": min_lat,
                "max_lon": max_lng,
                "max_lat": max_lat,
            },
            "tessellation": {"resolution": resolution},
            "profiles": {
                "enabled": True,
                "home_anchor_h3_resolution": resolution,
                "home_poi_inverse_weight": 0.0,
                "overture_building_features_path": str(building_path),
            },
            "road_network": {"enabled": False},
        }
    )

    anchors = _derive_home_anchor_candidates_from_tessellation(config, tess, "relevance", 200)
    anchor_cells = pd.Series(
        h3.latlng_to_cell(lat, lng, resolution)
        for lat, lng in zip(anchors["lat"], anchors["lng"])
    )

    assert len(anchors) == 200
    assert anchor_cells.value_counts().idxmax() == building_cell
    assert (anchor_cells == building_cell).sum() > (anchor_cells == poi_cell).sum()


def test_poi_building_home_anchors_exclude_empty_no_poi_cells(tmp_path):
    resolution = 8
    min_lng, min_lat, max_lng, max_lat = 2.30, 48.82, 2.53, 48.95
    boundary = h3.LatLngPoly(
        [
            (min_lat, min_lng),
            (min_lat, max_lng),
            (max_lat, max_lng),
            (max_lat, min_lng),
        ]
    )
    cells = sorted(h3.polygon_to_cells(boundary, resolution))
    building_cell = cells[len(cells) // 2]
    empty_cell = cells[0] if cells[0] != building_cell else cells[-1]
    building_lat, building_lng = h3.cell_to_latlng(building_cell)
    empty_lat, empty_lng = h3.cell_to_latlng(empty_cell)
    building_path = tmp_path / "buildings.parquet"
    pd.DataFrame({"h3_cell": [building_cell], "building_count": [1]}).to_parquet(building_path, index=False)
    tess = pd.DataFrame(
        {
            "tile_id": ["commercial_poi"],
            "lat": [building_lat],
            "lng": [building_lng],
            "relevance": [100.0],
        }
    )
    config = CityBehavExConfig.model_validate(
        {
            "simulation": {
                "agents": 50,
                "random_state": 13,
                "min_lon": min_lng,
                "min_lat": min_lat,
                "max_lon": max_lng,
                "max_lat": max_lat,
            },
            "tessellation": {"resolution": resolution},
            "profiles": {
                "enabled": True,
                "home_anchor_h3_resolution": resolution,
                "overture_building_features_path": str(building_path),
            },
            "road_network": {"enabled": False},
        }
    )

    anchors = _derive_home_anchor_candidates_from_tessellation(config, tess, "relevance", 50)
    anchor_cells = {
        h3.latlng_to_cell(lat, lng, resolution)
        for lat, lng in zip(anchors["lat"], anchors["lng"])
    }

    assert empty_cell not in anchor_cells
    assert anchor_cells == {building_cell}
    assert (empty_lat, empty_lng) != (building_lat, building_lng)


def test_default_home_anchor_cache_path_includes_method_and_resolution():
    config = CityBehavExConfig.model_validate(
        {
            "profiles": {
                "enabled": True,
                "output": "data/example_profiles.parquet",
                "home_anchor_h3_resolution": 8,
            },
            "road_network": {"enabled": False},
        }
    )

    assert _home_anchors_output_path(config).name == "example_profiles_home_anchors_poi_building_v3_h3r8.parquet"


def test_activity_column_absent_when_disabled():
    """Without activity params, no activities are sampled -> empty table."""
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.95],
        "lng": [2.3522, 2.55],
        "relevance": [1.0, 1.0],
    })
    diary_arrays = _diary_arrays_single([0, 1, 0], [0, 8 * 3600, 18 * 3600])
    df, _, _, activities = simulate_agents(
        tess, "relevance", diary_arrays,
        start_ts=0, end_ts=86400,
        slot_seconds=_SLOT, car_speed_kmh=_SPEED,
        n_agents=1, random_state=42,
    )
    assert "activity" not in df.columns
    assert len(activities) == 0


def test_activities_produce_non_trivial_dwell():
    """With activity CRP, departure should differ from slot timestamps for at least some records."""
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.95],
        "lng": [2.3522, 2.55],
        "relevance": [1.0, 1.0],
    })
    # 3 days of HOME/WORK/HOME pattern
    slots, locs = [], []
    for d in range(3):
        base = d * 86400
        slots += [base, base + 9 * 3600, base + 17 * 3600]
        locs += [0, 1, 0]
    diary_arrays = _diary_arrays_single(locs, slots)
    act_dur_mu, act_dur_sigma = activity_duration_arrays()
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    df, _, _, _ = simulate_agents(
        tess, "relevance", diary_arrays,
        start_ts=0, end_ts=3 * 86400,
        slot_seconds=_SLOT, car_speed_kmh=_SPEED,
        n_agents=1, random_state=7,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
    )
    assert (df["dwell_minutes"] >= 0).all()
    # dwell_minutes should vary (not all zero)
    assert df["dwell_minutes"].std() > 0


def test_activities_catalog_coverage():
    """Every purpose code 0-2 has at least one eligible activity."""
    purpose_act_starts, purpose_acts = build_eligibility_csr()
    assert len(purpose_act_starts) == 4  # 3 purposes + sentinel
    for p in range(3):
        n_eligible = int(purpose_act_starts[p + 1] - purpose_act_starts[p])
        assert n_eligible > 0, f"Purpose {p} has no eligible activities"


def _run_multi_agent_multi_day(*, on_day_flush=None):
    """2 agents alternating HOME/WORK/OTHER across 3 simulated days -- enough
    relocations (and day boundaries) to exercise the per-day waypoint flush."""
    lats = [48.8566, 48.9000, 48.9500]
    lngs = [2.3522, 2.4000, 2.4500]
    n_agents = 2

    def diary_for_agent(offset):
        slots, locs = [], []
        for d in range(3):
            base = d * 86400
            slots += [base + offset, base + 8 * 3600 + offset, base + 18 * 3600 + offset]
            locs += [0, 1, 2]
        return locs, slots

    diary_ts: list[int] = []
    diary_loc: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for agent in range(n_agents):
        locs, slots = diary_for_agent(agent * 60)
        starts.append(len(diary_ts))
        diary_ts.extend(slots)
        diary_loc.extend(locs)
        ends.append(len(diary_ts))

    return core.simulation_core_simulate_agents(
        latitudes=np.asarray(lats, dtype=float),
        longitudes=np.asarray(lngs, dtype=float),
        relevances=np.ones(len(lats), dtype=float),
        distances=np.empty(0, dtype=np.float64),
        neighbor_starts=np.zeros(n_agents + 1, dtype=np.int64),
        neighbors=np.empty(0, dtype=np.int64),
        diary_timestamps=np.asarray(diary_ts, dtype=np.int64),
        diary_abs_locs=np.asarray(diary_loc, dtype=np.int32),
        diary_starts=np.asarray(starts, dtype=np.int64),
        diary_ends=np.asarray(ends, dtype=np.int64),
        rho=1.0,
        gamma=0.21,
        alpha=0.0,
        start_ts=0,
        end_ts=3 * 86400,
        indipendency_window_s=1800,
        dt_update_mob_sim_s=3600,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=n_agents,
        master_seed=42,
        starting_locs=np.zeros(n_agents, dtype=np.int64),
        starting_locs_mode_relevance=False,
        work_tiles=np.ones(n_agents, dtype=np.int64),
        on_day_flush=on_day_flush,
    )


def test_on_day_flush_none_matches_baseline_return_shape():
    """The default (no callback) path must be unaffected by the new parameter."""
    trip, paths, activities = _run_multi_agent_multi_day()
    assert len(paths[0]) > 0  # sanity: this fixture does generate waypoints


def test_on_day_flush_chunks_plus_tail_reproduce_the_no_callback_paths():
    # `paths` is (stop_id, path_agent, path_stop_id, path_seq, path_lat,
    # path_lng, path_t) -- `stop_id` belongs to the (unflushed, Phase 2 scope)
    # stops table, so it must be identical regardless of streaming. The
    # remaining 6 columns are RoadPathOutputBuffers itself -- what Phase 1
    # flushes per day -- and are what the callback/tail reconstruction must match.
    baseline_trip, baseline_paths, _ = _run_multi_agent_multi_day()

    chunks: list[tuple] = []
    _, streamed_paths, _ = _run_multi_agent_multi_day(
        on_day_flush=lambda *arrays: chunks.append(tuple(np.asarray(a) for a in arrays))
    )

    # At least one mid-run flush must have happened, given 3 simulated days.
    assert len(chunks) > 0

    assert np.array_equal(np.asarray(baseline_paths[0]), np.asarray(streamed_paths[0]))

    streamed_columns = [np.asarray(col) for col in streamed_paths[1:]]
    reconstructed = [
        np.concatenate([chunk[col_idx] for chunk in chunks] + [streamed_columns[col_idx]])
        for col_idx in range(len(streamed_columns))
    ]

    for baseline_col, reconstructed_col in zip(baseline_paths[1:], reconstructed):
        assert np.array_equal(np.asarray(baseline_col), reconstructed_col)
