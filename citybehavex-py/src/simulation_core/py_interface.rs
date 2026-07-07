use h3o::Resolution;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::simulation_core::engine::simulate;
use crate::simulation_core::h3_batch::batch_latlng_to_cells;
use crate::simulation_core::inputs::{
    ActivityInputs, CoreInputs, DiaryInputs, InitialLocationInputs, LocationInputs,
    RoadNetworkInputs, SimulationParams, SocialGraphInputs, TransportInputs,
};
use crate::simulation_core::network_graph::{build_co_presence_edges, compute_graph_metrics};
use crate::simulation_core::outputs::{
    ActivityOutputBuffers, RoadPathOutputBuffers, TripOutputBuffers,
};
use crate::simulation_core::roads::{RoadGraph, batch_road_distances};

/// Batch lat/lng (degrees) -> H3 cell index conversion at a fixed resolution,
/// in parallel. Returns `u64` cell indices (`INVALID_CELL` == `u64::MAX` for
/// non-finite/out-of-range input rows) rather than the hex-string form
/// `h3.latlng_to_cell` returns in Python -- callers only use this to group/
/// compare locations, and comparing `u64`s avoids re-introducing a per-row
/// Python loop just to format each cell back into a string.
#[pyfunction]
#[pyo3(name = "batch_latlng_to_cells")]
pub fn batch_latlng_to_cells_py<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray1<'py, f64>,
    lngs: PyReadonlyArray1<'py, f64>,
    resolution: u8,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let lat_slice = lats.as_slice()?;
    let lng_slice = lngs.as_slice()?;
    if lat_slice.len() != lng_slice.len() {
        return Err(PyValueError::new_err(format!(
            "lats and lngs must have the same length, got {} and {}",
            lat_slice.len(),
            lng_slice.len()
        )));
    }
    let res = Resolution::try_from(resolution)
        .map_err(|e| PyValueError::new_err(format!("invalid H3 resolution {resolution}: {e}")))?;
    let cells = py.detach(|| batch_latlng_to_cells(lat_slice, lng_slice, res));
    Ok(cells.into_pyarray(py))
}

/// Borrowed slice from an optional numpy array, or an empty slice when absent.
fn opt_slice<'a, T: numpy::Element>(v: &'a Option<PyReadonlyArray1<'_, T>>) -> PyResult<&'a [T]> {
    match v {
        Some(arr) => Ok(arr.as_slice()?),
        None => Ok(&[]),
    }
}

/// A required i64 numpy array, clamped non-negative and cast to `usize`
/// (indices/counts from Python are never negative in practice, but numpy
/// int arrays don't enforce that at the type level).
fn i64_as_usize_vec(arr: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    Ok(arr.as_slice()?.iter().map(|&x| x.max(0) as usize).collect())
}

/// Same clamp-and-cast as `i64_as_usize_vec`, for an optional numpy array;
/// `None` when the array itself is absent.
fn opt_i64_as_usize_vec(v: &Option<PyReadonlyArray1<'_, i64>>) -> PyResult<Option<Vec<usize>>> {
    match v {
        Some(arr) => Ok(Some(
            arr.as_slice()?.iter().map(|&x| x.max(0) as usize).collect(),
        )),
        None => Ok(None),
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (
    latitudes, longitudes, relevances, distances,
    neighbor_starts, neighbors,
    diary_timestamps, diary_abs_locs, diary_starts, diary_ends,
    rho, gamma, alpha,
    start_ts, end_ts, indipendency_window_s, dt_update_mob_sim_s,
    slot_seconds, car_speed_kmh,
    n_agents, master_seed=None, diary_block_ids=None, starting_locs=None,
    starting_locs_mode_relevance=false,
    work_tiles=None,
    edge_profile_sim=None,
    act_embs=None, act_dur_mu=None, act_dur_sigma=None,
    purpose_act_starts=None, purpose_acts=None,
    profile_embs=None, emb_dim=0usize,
    act_kappa=1.0f64, act_temp=0.5f64,
    profile_act_sims=None,
    activity_alignment_scores=None, activity_cluster_labels=None,
    activity_alignment_clusters=0usize, activity_alignment_blocks=0usize,
    activity_alignment_previous=0usize, activity_history_weight=1.0f64,
    materialize_travel=true,
    road_edge_from=None, road_edge_to=None, road_edge_weight_ds=None,
    road_node_lats=None, road_node_lngs=None, location_road_node=None,
    max_leg_waypoints=16usize,
    gravity_deterrence_exponent=-2.0f64, gravity_origin_exponent=1.0f64,
    gravity_destination_exponent=1.0f64,
    walking_speed_kmh=4.8f64, bike_speed_kmh=15.0f64,
    has_car=None, has_bike=None, walking_threshold_km=None, bike_threshold_km=None,
    rail_edge_from=None, rail_edge_to=None, rail_edge_weight_ds=None,
    rail_node_lats=None, rail_node_lngs=None, location_rail_node=None,
    max_rail_leg_waypoints=16usize,
    on_day_flush=None,
    on_encounter_day_flush=None,
    on_trip_day_flush=None,
    on_activity_day_flush=None
))]
pub fn simulation_core_simulate_agents<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    distances: PyReadonlyArray1<'py, f64>,
    neighbor_starts: PyReadonlyArray1<'py, i64>,
    neighbors: PyReadonlyArray1<'py, i64>,
    diary_timestamps: PyReadonlyArray1<'py, i64>,
    diary_abs_locs: PyReadonlyArray1<'py, i32>,
    diary_starts: PyReadonlyArray1<'py, i64>,
    diary_ends: PyReadonlyArray1<'py, i64>,
    rho: f64,
    gamma: f64,
    alpha: f64,
    start_ts: i64,
    end_ts: i64,
    indipendency_window_s: i64,
    dt_update_mob_sim_s: i64,
    slot_seconds: i64,
    car_speed_kmh: f64,
    n_agents: usize,
    master_seed: Option<u64>,
    diary_block_ids: Option<PyReadonlyArray1<'py, i32>>,
    starting_locs: Option<PyReadonlyArray1<'py, i64>>,
    starting_locs_mode_relevance: bool,
    work_tiles: Option<PyReadonlyArray1<'py, i64>>,
    edge_profile_sim: Option<PyReadonlyArray1<'py, f64>>,
    act_embs: Option<PyReadonlyArray1<'py, f64>>,
    act_dur_mu: Option<PyReadonlyArray1<'py, f64>>,
    act_dur_sigma: Option<PyReadonlyArray1<'py, f64>>,
    purpose_act_starts: Option<PyReadonlyArray1<'py, i64>>,
    purpose_acts: Option<PyReadonlyArray1<'py, i64>>,
    profile_embs: Option<PyReadonlyArray1<'py, f64>>,
    emb_dim: usize,
    act_kappa: f64,
    act_temp: f64,
    profile_act_sims: Option<PyReadonlyArray1<'py, f64>>,
    activity_alignment_scores: Option<PyReadonlyArray1<'py, f64>>,
    activity_cluster_labels: Option<PyReadonlyArray1<'py, i64>>,
    activity_alignment_clusters: usize,
    activity_alignment_blocks: usize,
    activity_alignment_previous: usize,
    activity_history_weight: f64,
    materialize_travel: bool,
    road_edge_from: Option<PyReadonlyArray1<'py, i64>>,
    road_edge_to: Option<PyReadonlyArray1<'py, i64>>,
    road_edge_weight_ds: Option<PyReadonlyArray1<'py, i64>>,
    road_node_lats: Option<PyReadonlyArray1<'py, f64>>,
    road_node_lngs: Option<PyReadonlyArray1<'py, f64>>,
    location_road_node: Option<PyReadonlyArray1<'py, i64>>,
    max_leg_waypoints: usize,
    gravity_deterrence_exponent: f64,
    gravity_origin_exponent: f64,
    gravity_destination_exponent: f64,
    walking_speed_kmh: f64,
    bike_speed_kmh: f64,
    has_car: Option<PyReadonlyArray1<'py, bool>>,
    has_bike: Option<PyReadonlyArray1<'py, bool>>,
    walking_threshold_km: Option<PyReadonlyArray1<'py, f64>>,
    bike_threshold_km: Option<PyReadonlyArray1<'py, f64>>,
    rail_edge_from: Option<PyReadonlyArray1<'py, i64>>,
    rail_edge_to: Option<PyReadonlyArray1<'py, i64>>,
    rail_edge_weight_ds: Option<PyReadonlyArray1<'py, i64>>,
    rail_node_lats: Option<PyReadonlyArray1<'py, f64>>,
    rail_node_lngs: Option<PyReadonlyArray1<'py, f64>>,
    location_rail_node: Option<PyReadonlyArray1<'py, i64>>,
    max_rail_leg_waypoints: usize,
    on_day_flush: Option<Py<PyAny>>,
    on_encounter_day_flush: Option<Py<PyAny>>,
    on_trip_day_flush: Option<Py<PyAny>>,
    on_activity_day_flush: Option<Py<PyAny>>,
) -> PyResult<(
    (
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u8>>,
    ),
    (
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<u8>>,
    ),
    (
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
    ),
)> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let rels = relevances.as_slice()?;
    let dists = distances.as_slice()?;
    let dt_raw = diary_timestamps.as_slice()?;
    let da_raw = diary_abs_locs.as_slice()?;
    let default_block_ids;
    let db_raw = match &diary_block_ids {
        Some(arr) => arr.as_slice()?,
        None => {
            default_block_ids = vec![0i32; da_raw.len()];
            &default_block_ids
        }
    };

    let ns = i64_as_usize_vec(&neighbor_starts)?;
    let nb = i64_as_usize_vec(&neighbors)?;
    let ds = i64_as_usize_vec(&diary_starts)?;
    let de = i64_as_usize_vec(&diary_ends)?;

    let sl_buf = opt_i64_as_usize_vec(&starting_locs)?;
    let sl: Option<&[usize]> = sl_buf.as_deref();

    let wt_buf = opt_i64_as_usize_vec(&work_tiles)?.unwrap_or_default();
    let wt: &[usize] = &wt_buf;

    // Owned copy (unlike the other optional f64 arrays below) since it's
    // built from a slice borrowed from a short-lived `Option` match arm.
    let eps_buf = opt_slice(&edge_profile_sim)?.to_vec();
    let eps: &[f64] = &eps_buf;

    let act_embs_s = opt_slice(&act_embs)?;
    let act_dur_mu_s = opt_slice(&act_dur_mu)?;
    let act_dur_sigma_s = opt_slice(&act_dur_sigma)?;
    let profile_embs_s = opt_slice(&profile_embs)?;
    let profile_act_sims_s = opt_slice(&profile_act_sims)?;
    let activity_alignment_scores_s = opt_slice(&activity_alignment_scores)?;
    let activity_cluster_labels_v =
        opt_i64_as_usize_vec(&activity_cluster_labels)?.unwrap_or_default();

    let purpose_act_starts_v = opt_i64_as_usize_vec(&purpose_act_starts)?.unwrap_or_default();
    let purpose_acts_v = opt_i64_as_usize_vec(&purpose_acts)?.unwrap_or_default();

    let road_edge_from_v = opt_i64_as_usize_vec(&road_edge_from)?.unwrap_or_default();
    let road_edge_to_v = opt_i64_as_usize_vec(&road_edge_to)?.unwrap_or_default();
    let road_edge_weight_v = opt_i64_as_usize_vec(&road_edge_weight_ds)?.unwrap_or_default();
    let road_node_lats_s = opt_slice(&road_node_lats)?;
    let road_node_lngs_s = opt_slice(&road_node_lngs)?;
    let location_road_node_s = opt_slice(&location_road_node)?;

    let rail_edge_from_v = opt_i64_as_usize_vec(&rail_edge_from)?.unwrap_or_default();
    let rail_edge_to_v = opt_i64_as_usize_vec(&rail_edge_to)?.unwrap_or_default();
    let rail_edge_weight_v = opt_i64_as_usize_vec(&rail_edge_weight_ds)?.unwrap_or_default();
    let rail_node_lats_s = opt_slice(&rail_node_lats)?;
    let rail_node_lngs_s = opt_slice(&rail_node_lngs)?;
    let location_rail_node_s = opt_slice(&location_rail_node)?;

    let default_has_car = vec![true; n_agents];
    let default_has_bike = vec![false; n_agents];
    let default_walk_threshold = vec![0.0; n_agents];
    let default_bike_threshold = vec![0.0; n_agents];
    let has_car_s = match &has_car {
        Some(arr) => arr.as_slice()?,
        None => &default_has_car,
    };
    let has_bike_s = match &has_bike {
        Some(arr) => arr.as_slice()?,
        None => &default_has_bike,
    };
    let walking_threshold_s = match &walking_threshold_km {
        Some(arr) => arr.as_slice()?,
        None => &default_walk_threshold,
    };
    let bike_threshold_s = match &bike_threshold_km {
        Some(arr) => arr.as_slice()?,
        None => &default_bike_threshold,
    };

    // Marshal one day's worth of closed waypoint rows to the Python callback
    // (if given) as numpy arrays, in the same column order as the final
    // `path_*` return tuple below, so callers can share one DataFrame-
    // building helper for both the streamed chunks and the final tail.
    let mut on_day_flush_closure = on_day_flush.map(|callback| {
        move |chunk: RoadPathOutputBuffers| -> Result<(), String> {
            let agent = chunk.agent.into_pyarray(py);
            let dest_stop_id = chunk.dest_stop_id.into_pyarray(py);
            let seq = chunk.seq.into_pyarray(py);
            let lat = chunk.lat.into_pyarray(py);
            let lng = chunk.lng.into_pyarray(py);
            let t = chunk.t.into_pyarray(py);
            let mode = chunk.mode.into_pyarray(py);
            callback
                .call1(py, (agent, dest_stop_id, seq, lat, lng, t, mode))
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
    });
    let on_day_flush_ref = on_day_flush_closure
        .as_mut()
        .map(|f| f as &mut dyn FnMut(RoadPathOutputBuffers) -> Result<(), String>);

    // Marshal one day's worth of encounters to the Python callback (if given)
    // as numpy arrays, in the same column order as the final `encounter_*`
    // return tuple, so callers can share one DataFrame-building helper for
    // both the streamed chunks and the final tail.
    let mut on_encounter_day_flush_closure = on_encounter_day_flush.map(|callback| {
        move |chunk: (Vec<u32>, Vec<u32>, Vec<u32>, Vec<i32>)| -> Result<(), String> {
            let agent = chunk.0.into_pyarray(py);
            let contact = chunk.1.into_pyarray(py);
            let tile = chunk.2.into_pyarray(py);
            let ts = chunk.3.into_pyarray(py);
            callback
                .call1(py, (agent, contact, tile, ts))
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
    });
    let on_encounter_day_flush_ref = on_encounter_day_flush_closure.as_mut().map(|f| {
        f as &mut dyn FnMut((Vec<u32>, Vec<u32>, Vec<u32>, Vec<i32>)) -> Result<(), String>
    });

    // Marshal one day's worth of closed stop rows to the Python callback (if
    // given) as numpy arrays, in the same field order `_build_trip_frame`
    // expects (matching `TripOutputBuffers`), so callers can share one
    // DataFrame-building helper for both the streamed chunks and the final
    // tail.
    let mut on_trip_day_flush_closure = on_trip_day_flush.map(|callback| {
        move |chunk: TripOutputBuffers| -> Result<(), String> {
            let agent = chunk.agents.into_pyarray(py);
            let loc_id = chunk.loc_id.into_pyarray(py);
            let arrival = chunk.arrival.into_pyarray(py);
            let departure = chunk.departure.into_pyarray(py);
            let duration = chunk.duration.into_pyarray(py);
            let stop_id = chunk.stop_id.into_pyarray(py);
            let abstract_loc = chunk.abstract_loc.into_pyarray(py);
            callback
                .call1(
                    py,
                    (
                        agent,
                        loc_id,
                        arrival,
                        departure,
                        duration,
                        stop_id,
                        abstract_loc,
                    ),
                )
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
    });
    let on_trip_day_flush_ref = on_trip_day_flush_closure
        .as_mut()
        .map(|f| f as &mut dyn FnMut(TripOutputBuffers) -> Result<(), String>);

    // Marshal one day's worth of closed micro-activity rows to the Python
    // callback (if given) as numpy arrays, in the same field order as the
    // final `act_*` return tuple.
    let mut on_activity_day_flush_closure = on_activity_day_flush.map(|callback| {
        move |chunk: ActivityOutputBuffers| -> Result<(), String> {
            let agent = chunk.agent.into_pyarray(py);
            let stop_id = chunk.stop_id.into_pyarray(py);
            let seq = chunk.seq.into_pyarray(py);
            let activity = chunk.activity.into_pyarray(py);
            let arrival = chunk.arrival.into_pyarray(py);
            let departure = chunk.departure.into_pyarray(py);
            let block_id = chunk.block_id.into_pyarray(py);
            callback
                .call1(
                    py,
                    (agent, stop_id, seq, activity, arrival, departure, block_id),
                )
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
    });
    let on_activity_day_flush_ref = on_activity_day_flush_closure
        .as_mut()
        .map(|f| f as &mut dyn FnMut(ActivityOutputBuffers) -> Result<(), String>);

    let output = simulate(
        CoreInputs {
            locations: LocationInputs {
                lats,
                lngs,
                relevances: rels,
                distances: dists,
            },
            social_graph: SocialGraphInputs {
                neighbor_starts: &ns,
                neighbors: &nb,
                edge_profile_sim: eps,
            },
            diary: DiaryInputs {
                timestamps: dt_raw,
                abstract_locations: da_raw,
                block_ids: db_raw,
                starts: &ds,
                ends: &de,
            },
            params: SimulationParams {
                rho,
                gamma,
                alpha,
                gravity_deterrence_exponent,
                gravity_origin_exponent,
                gravity_destination_exponent,
                start_ts,
                end_ts,
                indipendency_window_s,
                dt_update_mob_sim_s,
                slot_seconds,
                car_speed_kmh,
                walking_speed_kmh,
                bike_speed_kmh,
                n_agents,
                master_seed,
            },
            initial_locations: InitialLocationInputs {
                starting_locs: sl,
                starting_locs_mode_relevance,
                work_tiles: wt,
            },
            activities: ActivityInputs {
                act_embs: act_embs_s,
                act_dur_mu: act_dur_mu_s,
                act_dur_sigma: act_dur_sigma_s,
                purpose_act_starts: &purpose_act_starts_v,
                purpose_acts: &purpose_acts_v,
                profile_embs: profile_embs_s,
                profile_act_sims: profile_act_sims_s,
                contextual_scores: activity_alignment_scores_s,
                cluster_labels: &activity_cluster_labels_v,
                n_clusters: activity_alignment_clusters,
                n_blocks: activity_alignment_blocks,
                n_previous: activity_alignment_previous,
                history_weight: activity_history_weight,
                emb_dim,
                kappa: act_kappa,
                temperature: act_temp,
                materialize_travel,
            },
            road_network: RoadNetworkInputs {
                edge_from: &road_edge_from_v,
                edge_to: &road_edge_to_v,
                edge_weight_ds: &road_edge_weight_v,
                node_lats: road_node_lats_s,
                node_lngs: road_node_lngs_s,
                location_node: location_road_node_s,
                max_leg_waypoints,
            },
            rail_network: RoadNetworkInputs {
                edge_from: &rail_edge_from_v,
                edge_to: &rail_edge_to_v,
                edge_weight_ds: &rail_edge_weight_v,
                node_lats: rail_node_lats_s,
                node_lngs: rail_node_lngs_s,
                location_node: location_rail_node_s,
                max_leg_waypoints: max_rail_leg_waypoints,
            },
            transport: TransportInputs {
                has_car: has_car_s,
                has_bike: has_bike_s,
                walking_threshold_km: walking_threshold_s,
                bike_threshold_km: bike_threshold_s,
            },
        },
        on_day_flush_ref,
        on_encounter_day_flush_ref,
        on_trip_day_flush_ref,
        on_activity_day_flush_ref,
    )
    .map_err(PyValueError::new_err)?;

    Ok((
        (
            output.agents.into_pyarray(py),
            output.loc_id.into_pyarray(py),
            output.arrival.into_pyarray(py),
            output.departure.into_pyarray(py),
            output.duration.into_pyarray(py),
            output.encounter_agent.into_pyarray(py),
            output.encounter_contact.into_pyarray(py),
            output.encounter_tile.into_pyarray(py),
            output.encounter_ts.into_pyarray(py),
            output.stop_abstract_loc.into_pyarray(py),
        ),
        (
            output.stop_id.into_pyarray(py),
            output.path_agent.into_pyarray(py),
            output.path_stop_id.into_pyarray(py),
            output.path_seq.into_pyarray(py),
            output.path_lat.into_pyarray(py),
            output.path_lng.into_pyarray(py),
            output.path_t.into_pyarray(py),
            output.path_mode.into_pyarray(py),
        ),
        (
            output.act_agent.into_pyarray(py),
            output.act_stop_id.into_pyarray(py),
            output.act_seq.into_pyarray(py),
            output.act_activity.into_pyarray(py),
            output.act_arrival.into_pyarray(py),
            output.act_departure.into_pyarray(py),
            output.act_block_id.into_pyarray(py),
        ),
    ))
}

/// Groups `(day, location, node)` presence rows by `(day, location)` and
/// emits one edge per unique co-presence pair, with per-edge persistence
/// (fraction of `time_steps` the pair was seen together on). Replaces the
/// `itertools.combinations` + `dict[edge, set[day]]` loop in
/// `citybehavex.reports.network_validation._observed_edges_and_persistence`
/// -- see `network_graph.rs` for why (measured 150s there on shanghai's
/// ~65M raw pair-instances). Groups larger than `max_group_size` are
/// skipped (`skipped_groups`/`skipped_rows` report how many/how large).
#[pyfunction]
#[pyo3(name = "build_co_presence_edges")]
pub fn build_co_presence_edges_py<'py>(
    py: Python<'py>,
    day_codes: PyReadonlyArray1<'py, i64>,
    location_codes: PyReadonlyArray1<'py, i64>,
    nodes: PyReadonlyArray1<'py, i64>,
    max_group_size: usize,
    time_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f64>>,
    u64,
    u64,
)> {
    let day = day_codes.as_slice()?;
    let location = location_codes.as_slice()?;
    let node = nodes.as_slice()?;
    if day.len() != location.len() || day.len() != node.len() {
        return Err(PyValueError::new_err(
            "day_codes, location_codes and nodes must have the same length",
        ));
    }
    let (edge_from, edge_to, persistence, skipped_groups, skipped_rows) = py.detach(|| {
        build_co_presence_edges(day, location, node, max_group_size, time_steps)
    });
    Ok((
        edge_from.into_pyarray(py),
        edge_to.into_pyarray(py),
        persistence.into_pyarray(py),
        skipped_groups,
        skipped_rows,
    ))
}

/// Per-node clustering coefficient and per-edge topological overlap
/// (Jaccard similarity of endpoint neighborhoods) for an undirected graph
/// given as an edge list. Replaces the pure-Python `O(sum of degree^2)`
/// nested loops over `set`-based adjacency in
/// `citybehavex.reports.network_validation.clustering_coefficients`/
/// `topological_overlap` (measured: ~51 minutes extrapolated for shanghai's
/// unusually dense observed co-presence graph, ~1,070 average degree).
#[pyfunction]
#[pyo3(name = "graph_metrics")]
pub fn graph_metrics_py<'py>(
    py: Python<'py>,
    node_count: usize,
    edge_from: PyReadonlyArray1<'py, u32>,
    edge_to: PyReadonlyArray1<'py, u32>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let from = edge_from.as_slice()?;
    let to = edge_to.as_slice()?;
    if from.len() != to.len() {
        return Err(PyValueError::new_err(
            "edge_from and edge_to must have the same length",
        ));
    }
    let metrics = py.detach(|| compute_graph_metrics(node_count, from, to));
    Ok((
        metrics.clustering_coefficient.into_pyarray(py),
        metrics.topological_overlap.into_pyarray(py),
    ))
}

/// A road network prepared once (contraction hierarchy) and reused for many
/// point-to-point physical-distance queries. Used by report/comparison code
/// to recompute jump-length / radius-of-gyration metrics over the real road
/// network instead of straight-line Haversine: CH preparation, not the query
/// itself, is the expensive step, so one handle serves every query batch
/// needed by a single comparison run (synthetic + real, every metric, every
/// filter group) instead of re-preparing per call.
#[pyclass]
pub struct RoadNetworkHandle {
    graph: RoadGraph,
}

#[pymethods]
impl RoadNetworkHandle {
    #[new]
    fn new(
        edge_from: PyReadonlyArray1<'_, i64>,
        edge_to: PyReadonlyArray1<'_, i64>,
        edge_weight_ds: PyReadonlyArray1<'_, i64>,
        edge_length_m: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let ef = i64_as_usize_vec(&edge_from)?;
        let et = i64_as_usize_vec(&edge_to)?;
        let ew = i64_as_usize_vec(&edge_weight_ds)?;
        let el = edge_length_m.as_slice()?;
        Ok(Self {
            graph: RoadGraph::build_with_length(&ef, &et, &ew, el),
        })
    }

    /// Batch physical-distance (metres) query for `(from_node, to_node)`
    /// pairs against the prepared contraction hierarchy. Returns
    /// `(distances_m, connected)`, `connected` as `0`/`1` per pair (no
    /// existing precedent in this crate for bool numpy arrays, matching the
    /// plain-numeric-array convention already used elsewhere, e.g.
    /// `abstract_loc: i32`). The Python caller falls back to straight-line
    /// Haversine wherever `connected == 0` (negative/unsnapped node ids or a
    /// disconnected graph component).
    fn batch_distances<'py>(
        &self,
        py: Python<'py>,
        from_nodes: PyReadonlyArray1<'py, i64>,
        to_nodes: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u8>>)> {
        let from_slice = from_nodes.as_slice()?;
        let to_slice = to_nodes.as_slice()?;
        let (dist, conn) =
            py.detach(|| batch_road_distances(&self.graph, from_slice, to_slice));
        let conn_u8: Vec<u8> = conn.into_iter().map(|b| b as u8).collect();
        Ok((dist.into_pyarray(py), conn_u8.into_pyarray(py)))
    }
}
