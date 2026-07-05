from __future__ import annotations

import numpy as np
import pandas as pd
import h3

import citybehavex._core as core
from citybehavex.activities import (
    N_ACTIVITIES,
    activity_duration_arrays,
    build_catalog,
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
    # Returns a 3-tuple of tuples: (10 trip arrays), (7 path arrays), (6 activity arrays).
    # Trip: agents, loc_id, arrival, departure, duration,
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
    ag, _, arr, dep, dur, *_ = trip

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
    _, _, arr, dep, dur, *_ = trip
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
    _, _, arr, *_ = trip
    assert any(int(a) % _SLOT != 0 for a in np.asarray(arr))


def _run_mode_case(
    *,
    has_car: bool,
    has_bike: bool,
    walk_threshold_km: float,
    bike_threshold_km: float,
    rail: bool = False,
):
    kwargs = {}
    if rail:
        kwargs.update(
            rail_edge_from=np.array([0], dtype=np.int64),
            rail_edge_to=np.array([1], dtype=np.int64),
            rail_edge_weight_ds=np.array([100], dtype=np.int64),
            rail_node_lats=np.array([48.8566, 48.95], dtype=np.float64),
            rail_node_lngs=np.array([2.3522, 2.55], dtype=np.float64),
            location_rail_node=np.array([0, 1], dtype=np.int64),
            max_rail_leg_waypoints=8,
        )
    trip, paths, _ = core.simulation_core_simulate_agents(
        latitudes=np.array([48.8566, 48.95], dtype=float),
        longitudes=np.array([2.3522, 2.55], dtype=float),
        relevances=np.ones(2, dtype=float),
        distances=np.empty(0, dtype=np.float64),
        neighbor_starts=np.array([0, 0], dtype=np.int64),
        neighbors=np.empty(0, dtype=np.int64),
        diary_timestamps=np.array([0, 8 * 3600], dtype=np.int64),
        diary_abs_locs=np.array([0, 1], dtype=np.int32),
        diary_starts=np.array([0], dtype=np.int64),
        diary_ends=np.array([2], dtype=np.int64),
        rho=1.0,
        gamma=0.0,
        alpha=0.0,
        start_ts=0,
        end_ts=86400,
        indipendency_window_s=1800,
        dt_update_mob_sim_s=3600,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=1,
        master_seed=42,
        starting_locs=np.array([0], dtype=np.int64),
        starting_locs_mode_relevance=False,
        work_tiles=np.array([1], dtype=np.int64),
        walking_speed_kmh=5.0,
        bike_speed_kmh=15.0,
        has_car=np.array([has_car], dtype=np.bool_),
        has_bike=np.array([has_bike], dtype=np.bool_),
        walking_threshold_km=np.array([walk_threshold_km], dtype=np.float64),
        bike_threshold_km=np.array([bike_threshold_km], dtype=np.float64),
        **kwargs,
    )
    assert len(trip[0]) == 2
    return np.asarray(paths[7], dtype=np.uint8)


def test_transport_mode_selection_uses_walk_car_bike_rail_and_fallback():
    # Path mode codes: 1=car, 2=walk, 3=bike, 4=rail.
    assert set(_run_mode_case(has_car=True, has_bike=False, walk_threshold_km=100.0, bike_threshold_km=0.0)) == {2}
    assert set(_run_mode_case(has_car=True, has_bike=True, walk_threshold_km=0.0, bike_threshold_km=100.0)) == {1}
    assert set(_run_mode_case(has_car=False, has_bike=True, walk_threshold_km=0.0, bike_threshold_km=100.0)) == {3}
    assert set(_run_mode_case(has_car=False, has_bike=False, walk_threshold_km=0.0, bike_threshold_km=0.0, rail=True)) == {4}
    assert set(_run_mode_case(has_car=False, has_bike=False, walk_threshold_km=0.0, bike_threshold_km=0.0)) == {1}


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

    loc_id = np.asarray(trip[1])
    assert len(loc_id) == 5
    assert loc_id[1] == loc_id[3]


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

    loc_id = np.asarray(trip[1])
    assert len(loc_id) == 5
    assert loc_id[1] != loc_id[3]


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
    ag, loc_id, arr, dep, *_ = trip

    assert len(ag) == 1
    assert arr[0] == 0
    assert dep[0] == 86400
    assert loc_id[0] == 0


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


def test_simulate_agents_defaults_to_colocation_graph_when_home_tiles_and_embeddings_present():
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
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float64)

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
        starting_locs=np.array([0, 0], dtype=np.int64),
        work_tiles=np.array([0, 0], dtype=np.int64),
        profile_embeddings=embeddings,
        degree_mu_ln=np.log(2),
        degree_sigma_ln=0.2,
        max_degree=2,
        home_h3_resolution=7,
        work_h3_resolution=7,
        return_social_graph=True,
    )
    artifact = result[4].to_dict()
    assert artifact["node_count"] == 2
    # Both agents share the same home and work tile, so the only possible
    # edge is colocation-eligible; the colocation builder (not the profile
    # kNN fallback) is exercised whenever starting_locs is supplied.
    assert artifact["edge_count"] <= 2


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

    _, _, arr, dep, dur, *_ = (np.asarray(a) for a in trip)
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


def test_contextual_activity_alignment_uses_previous_activity_with_separate_micro_crp():
    tess = pd.DataFrame({
        "tile_id": [0, 1],
        "lat": [48.8566, 48.8580],
        "lng": [2.3522, 2.3540],
        "relevance": [1.0, 1.0],
    })
    slot_times = np.array([0, 2 * 3600], dtype=np.int64)
    abs_locs = np.array([0, 1], dtype=np.int32)
    block_ids = np.array([0, 1], dtype=np.int32)
    diary_arrays = (
        slot_times,
        abs_locs,
        np.array([0], dtype=np.int64),
        np.array([2], dtype=np.int64),
        block_ids,
    )
    ten_min_hours = 10 / 60
    act_dur_mu = np.log(np.full(N_ACTIVITIES, ten_min_hours, dtype=np.float64))
    act_dur_sigma = np.zeros(N_ACTIVITIES, dtype=np.float64)
    cleanetc = 6
    foodprep = 5
    paidwork = 3
    purpose_act_starts = np.array([0, 2, 3, 3], dtype=np.int64)
    purpose_acts = np.array([cleanetc, foodprep, paidwork], dtype=np.int64)
    scores = np.zeros((1, 2, N_ACTIVITIES + 1, N_ACTIVITIES), dtype=np.float64)
    scores[0, 0, 0, cleanetc] = 1.0
    scores[0, 0, cleanetc + 1, foodprep] = 1.0
    scores[0, 0, foodprep + 1, foodprep] = 1.0

    _df, _encounters, _moving, activities = simulate_agents(
        tess,
        "relevance",
        diary_arrays,
        start_ts=0,
        end_ts=3 * 3600,
        slot_seconds=_SLOT,
        car_speed_kmh=_SPEED,
        n_agents=1,
        random_state=42,
        act_dur_mu=act_dur_mu,
        act_dur_sigma=act_dur_sigma,
        purpose_act_starts=purpose_act_starts,
        purpose_acts=purpose_acts,
        act_temp=0.01,
        activity_alignment_scores=scores,
        activity_cluster_labels=np.array([0], dtype=np.int64),
    )

    first_stop = activities[activities["stop_id"] == 0]["activity"].tolist()
    assert first_stop[:2] == [cleanetc, foodprep]


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
        AgentProfilesConfig(enabled=True, work_from_home_probability=0.0),
        np.random.default_rng(3),
        augmented,
        "relevance",
        home_tile_pool=home_pool,
    )
    assert {p.home_tile for p in profiles}.issubset(set(home_pool.tolist()))
    assert {p.work_tile for p in profiles}.issubset({0, 1})


def test_work_distance_model_none_preserves_relevance_weighted_sampling():
    tess = pd.DataFrame(
        {
            "tile_id": ["near_work", "far_work", "home"],
            "lat": [0.01, 0.50, 0.0],
            "lng": [0.0, 0.0, 0.0],
            "purpose": ["WORK", "WORK", "HOME"],
            "relevance": [1.0, 100.0, 1.0],
        }
    )
    config = AgentProfilesConfig(
        enabled=True,
        work_distance_model="none",
        work_from_home_probability=1.0,
    )

    profiles = generate_profiles(
        300,
        config,
        np.random.default_rng(6),
        tess,
        "relevance",
        home_tile_pool=np.array([2]),
    )
    work_counts = pd.Series([p.work_tile for p in profiles]).value_counts()

    assert work_counts.get(1, 0) > work_counts.get(0, 0) * 20
    assert {p.home_tile for p in profiles} == {2}


def test_conditional_work_sampling_excludes_home_rows_from_work_pool():
    tess = pd.DataFrame(
        {
            "tile_id": ["work", "home_as_attractive", "home"],
            "lat": [0.20, 0.0, 0.0],
            "lng": [0.0, 0.0, 0.0],
            "purpose": ["WORK", "HOME", "HOME"],
            "relevance": [1.0, 100000.0, 1.0],
        }
    )
    config = AgentProfilesConfig(enabled=True, work_from_home_probability=0.0)

    profiles = generate_profiles(
        50,
        config,
        np.random.default_rng(7),
        tess,
        "relevance",
        home_tile_pool=np.array([2]),
    )

    assert {p.home_tile for p in profiles} == {2}
    assert {p.work_tile for p in profiles} == {0}


def test_conditional_work_sampling_expands_when_radius_has_no_candidates():
    tess = pd.DataFrame(
        {
            "tile_id": ["far_work", "home"],
            "lat": [1.0, 0.0],
            "lng": [0.0, 0.0],
            "purpose": ["WORK", "HOME"],
            "relevance": [1.0, 1.0],
        }
    )
    config = AgentProfilesConfig(
        enabled=True,
        work_distance_max_km=1.0,
        work_distance_fallback="expand",
        work_from_home_probability=0.0,
    )

    profiles = generate_profiles(
        20,
        config,
        np.random.default_rng(8),
        tess,
        "relevance",
        home_tile_pool=np.array([1]),
    )

    assert {p.work_tile for p in profiles} == {0}


def test_work_from_home_probability_assigns_work_to_home_tile():
    tess = pd.DataFrame(
        {
            "tile_id": ["work", "home"],
            "lat": [0.20, 0.0],
            "lng": [0.0, 0.0],
            "purpose": ["WORK", "HOME"],
            "relevance": [100.0, 1.0],
        }
    )
    config = AgentProfilesConfig(enabled=True, work_from_home_probability=1.0)

    profiles = generate_profiles(
        30,
        config,
        np.random.default_rng(9),
        tess,
        "relevance",
        home_tile_pool=np.array([1]),
    )

    assert {p.home_tile for p in profiles} == {1}
    assert {p.work_tile for p in profiles} == {1}


def test_exponential_work_distance_favors_nearer_candidates_monotonically():
    tess = pd.DataFrame(
        {
            "tile_id": ["near_work", "mid_work", "far_work", "home"],
            "lat": [0.01, 0.05, 0.20, 0.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
            "purpose": ["WORK", "WORK", "WORK", "HOME"],
            "relevance": [1.0, 1.0, 1.0, 1.0],
        }
    )
    config = AgentProfilesConfig(
        enabled=True,
        work_distance_model="exponential",
        work_distance_exponential_lambda=0.3,
        work_distance_density_correction_power=0.0,
        work_from_home_probability=0.0,
    )

    profiles = generate_profiles(
        1000,
        config,
        np.random.default_rng(11),
        tess,
        "relevance",
        home_tile_pool=np.array([3]),
    )
    counts = pd.Series([p.work_tile for p in profiles]).value_counts()

    assert counts.get(0, 0) > counts.get(1, 0) > counts.get(2, 0)


def test_density_correction_reduces_far_ring_selection():
    tess = pd.DataFrame(
        {
            "tile_id": ["near_work", "far_work_1", "far_work_2", "far_work_3", "home"],
            "lat": [0.01, 0.20, 0.20, 0.20, 0.0],
            "lng": [0.0, 0.00, 0.01, -0.01, 0.0],
            "purpose": ["WORK", "WORK", "WORK", "WORK", "HOME"],
            "relevance": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    base_config = AgentProfilesConfig(
        enabled=True,
        work_distance_model="exponential",
        work_distance_exponential_lambda=0.05,
        work_distance_max_km=100.0,
        work_distance_density_correction_power=0.0,
        work_from_home_probability=0.0,
    )
    corrected_config = base_config.model_copy(update={"work_distance_density_correction_power": 1.0})

    base_profiles = generate_profiles(
        1200,
        base_config,
        np.random.default_rng(12),
        tess,
        "relevance",
        home_tile_pool=np.array([4]),
    )
    corrected_profiles = generate_profiles(
        1200,
        corrected_config,
        np.random.default_rng(12),
        tess,
        "relevance",
        home_tile_pool=np.array([4]),
    )
    base_far_share = np.mean([p.work_tile in {1, 2, 3} for p in base_profiles])
    corrected_far_share = np.mean([p.work_tile in {1, 2, 3} for p in corrected_profiles])

    assert corrected_far_share < base_far_share * 0.5


def test_log1p_attractiveness_keeps_local_work_competitive_with_far_megahub():
    tess = pd.DataFrame(
        {
            "tile_id": ["near_work", "far_megahub", "home"],
            "lat": [0.01, 0.10, 0.0],
            "lng": [0.0, 0.0, 0.0],
            "purpose": ["WORK", "WORK", "HOME"],
            "relevance": [10.0, 10000.0, 1.0],
        }
    )
    config = AgentProfilesConfig(
        enabled=True,
        work_distance_model="exponential",
        work_distance_exponential_lambda=0.1,
        work_distance_density_correction_power=1.0,
        work_from_home_probability=0.0,
    )

    profiles = generate_profiles(
        1000,
        config,
        np.random.default_rng(13),
        tess,
        "relevance",
        home_tile_pool=np.array([2]),
    )
    work_counts = pd.Series([p.work_tile for p in profiles]).value_counts()

    assert work_counts.get(0, 0) > work_counts.get(1, 0)


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


def test_activities_catalog_uses_25_mtus_categories():
    catalog = build_catalog()

    assert N_ACTIVITIES == 25
    assert [activity.name for activity in catalog] == [
        "sleep",
        "eatdrink",
        "selfcare",
        "paidwork",
        "educatn",
        "foodprep",
        "cleanetc",
        "maintain",
        "shopserv",
        "garden",
        "petcare",
        "eldcare",
        "pkidcare",
        "ikidcare",
        "religion",
        "volorgwk",
        "commute",
        "travel",
        "sportex",
        "tvradio",
        "read",
        "compint",
        "goout",
        "leisure",
        "missing",
    ]
    assert catalog[-1].name == "missing"
    assert catalog[-1].eligible_purposes == []


def _run_multi_agent_multi_day(
    *,
    on_day_flush=None,
    on_trip_day_flush=None,
    on_activity_day_flush=None,
    with_activities=False,
):
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

    act_kwargs = {}
    if with_activities:
        act_dur_mu, act_dur_sigma = activity_duration_arrays()
        purpose_act_starts, purpose_acts = build_eligibility_csr()
        act_kwargs = dict(
            act_dur_mu=act_dur_mu,
            act_dur_sigma=act_dur_sigma,
            purpose_act_starts=purpose_act_starts,
            purpose_acts=purpose_acts,
        )

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
        on_trip_day_flush=on_trip_day_flush,
        on_activity_day_flush=on_activity_day_flush,
        **act_kwargs,
    )


def test_on_trip_and_activity_day_flush_none_matches_baseline_return_shape():
    """The default (no callback) path must be unaffected by the new params."""
    trip, _, activities = _run_multi_agent_multi_day(with_activities=True)
    assert len(trip[0]) > 0
    assert len(activities[0]) > 0


def test_on_trip_day_flush_chunks_plus_tail_reproduce_the_no_callback_trip():
    # trip tuple: agents, loc_id, arrival, departure, duration,
    #             enc_agent, enc_contact, enc_tile, enc_ts, stop_abstract_loc
    # paths tuple: stop_id, path_agent, ...
    # A row's flush position depends on when it *finally* closes (which may
    # be a later day than when it was pushed), not its original push order,
    # so -- like encounters -- compare as a multiset of (agent, stop_id,
    # arrival, departure, duration, abstract_loc) rows rather than
    # positionally. `stop_id` alone guarantees no row is double-counted or
    # dropped; including arrival/departure/duration/abstract_loc also
    # verifies the departure-patching relationship survived compaction
    # (a still-open row flushed at the wrong time would show a placeholder
    # departure == arrival instead of its real, later-patched value).
    baseline_trip, baseline_paths, _ = _run_multi_agent_multi_day()

    chunks: list[tuple] = []
    streamed_trip, streamed_paths, _ = _run_multi_agent_multi_day(
        on_trip_day_flush=lambda *arrays: chunks.append(tuple(np.asarray(a) for a in arrays))
    )

    # At least one mid-run flush must have happened, given 3 simulated days
    # and 3 relocations/day/agent.
    assert len(chunks) > 0

    # Chunk field order: agent, loc_id, arrival, departure, duration, stop_id, abstract_loc.
    streamed_tail = [np.asarray(col) for col in streamed_trip[0:5]] + [
        np.asarray(streamed_paths[0]),
        np.asarray(streamed_trip[9]),
    ]
    reconstructed_cols = [
        np.concatenate([chunk[col_idx] for chunk in chunks] + [streamed_tail[col_idx]])
        for col_idx in range(7)
    ]
    reconstructed_rows = sorted(zip(*reconstructed_cols))

    baseline_cols = [np.asarray(col) for col in baseline_trip[0:5]] + [
        np.asarray(baseline_paths[0]),
        np.asarray(baseline_trip[9]),
    ]
    baseline_rows = sorted(zip(*baseline_cols))
    assert baseline_rows == reconstructed_rows


def test_on_activity_day_flush_chunks_plus_tail_reproduce_the_no_callback_activities():
    # activities tuple: act_agent, act_stop_id, act_seq, act_activity, act_arrival, act_departure
    # Same unordered-multiset comparison as trips, for the same reason.
    _, _, baseline_acts = _run_multi_agent_multi_day(with_activities=True)

    chunks: list[tuple] = []
    _, _, streamed_acts = _run_multi_agent_multi_day(
        with_activities=True,
        on_activity_day_flush=lambda *arrays: chunks.append(
            tuple(np.asarray(a) for a in arrays)
        ),
    )

    assert len(chunks) > 0

    streamed_tail = [np.asarray(col) for col in streamed_acts]
    reconstructed_cols = [
        np.concatenate([chunk[col_idx] for chunk in chunks] + [streamed_tail[col_idx]])
        for col_idx in range(6)
    ]
    reconstructed_rows = sorted(zip(*reconstructed_cols))
    baseline_rows = sorted(zip(*(np.asarray(col) for col in baseline_acts)))
    assert baseline_rows == reconstructed_rows


def _run_multi_agent_multi_day_with_social(*, on_encounter_day_flush=None):
    """6 fully-connected agents, high social mixing (alpha=0.9) and moderate
    exploration (rho=0.3) across 3 simulated days -- enough social-choice
    encounters (and day boundaries) to exercise the per-day encounter flush."""
    lats = [48.8566, 48.9000, 48.9500, 48.86, 48.87, 48.88]
    lngs = [2.3522, 2.4000, 2.4500, 2.40, 2.41, 2.42]
    n_agents = 6

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

    neighbor_starts = np.arange(
        0, n_agents * (n_agents - 1) + 1, n_agents - 1, dtype=np.int64
    )
    neighbors = np.array(
        [b for a in range(n_agents) for b in range(n_agents) if b != a], dtype=np.int64
    )
    edge_sim = np.ones(len(neighbors), dtype=np.float64)

    return core.simulation_core_simulate_agents(
        latitudes=np.asarray(lats, dtype=float),
        longitudes=np.asarray(lngs, dtype=float),
        relevances=np.ones(len(lats), dtype=float),
        distances=np.empty(0, dtype=np.float64),
        neighbor_starts=neighbor_starts,
        neighbors=neighbors,
        diary_timestamps=np.asarray(diary_ts, dtype=np.int64),
        diary_abs_locs=np.asarray(diary_loc, dtype=np.int32),
        diary_starts=np.asarray(starts, dtype=np.int64),
        diary_ends=np.asarray(ends, dtype=np.int64),
        rho=0.3,
        gamma=0.21,
        alpha=0.9,
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
        edge_profile_sim=edge_sim,
        on_encounter_day_flush=on_encounter_day_flush,
    )


def test_on_encounter_day_flush_none_matches_baseline_return_shape():
    """The default (no callback) path must be unaffected by the new parameter."""
    trip, _, _ = _run_multi_agent_multi_day_with_social()
    assert len(trip[5]) > 0  # sanity: this fixture does generate encounters


def test_on_encounter_day_flush_chunks_plus_tail_reproduce_the_no_callback_encounters():
    # trip tuple: agents, loc_id, arrival, departure, duration,
    #             enc_agent, enc_contact, enc_tile, enc_ts, stop_abstract_loc
    baseline_trip, _, _ = _run_multi_agent_multi_day_with_social()

    chunks: list[tuple] = []
    streamed_trip, _, _ = _run_multi_agent_multi_day_with_social(
        on_encounter_day_flush=lambda *arrays: chunks.append(
            tuple(np.asarray(a) for a in arrays)
        )
    )

    # At least one mid-run flush must have happened, given 3 simulated days.
    assert len(chunks) > 0

    # Encounters are unordered records: the baseline flattens once at the end
    # (grouped by agent, chronological within agent), while streaming
    # flattens per day (grouped by day, then agent) -- same multiset of
    # records, different order. Compare as sorted tuples rather than
    # positionally.
    streamed_tail = [np.asarray(col) for col in streamed_trip[5:9]]
    reconstructed_cols = [
        np.concatenate([chunk[col_idx] for chunk in chunks] + [streamed_tail[col_idx]])
        for col_idx in range(4)
    ]
    reconstructed_rows = sorted(zip(*reconstructed_cols))
    baseline_rows = sorted(zip(*(np.asarray(col) for col in baseline_trip[5:9])))
    assert baseline_rows == reconstructed_rows


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
