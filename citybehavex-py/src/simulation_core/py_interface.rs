use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::simulation_core::engine::simulate;
use crate::simulation_core::inputs::{
    ActivityInputs, CoreInputs, DiaryInputs, InitialLocationInputs, LocationInputs,
    RoadNetworkInputs, SimulationParams, SocialGraphInputs,
};
use crate::simulation_core::outputs::{ActivityOutputBuffers, RoadPathOutputBuffers, TripOutputBuffers};
use crate::simulation_core::roads::{batch_road_distances, RoadGraph};

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
    n_agents, master_seed=None, starting_locs=None,
    starting_locs_mode_relevance=false,
    work_tiles=None,
    edge_profile_sim=None,
    act_embs=None, act_dur_mu=None, act_dur_sigma=None,
    purpose_act_starts=None, purpose_acts=None,
    profile_embs=None, emb_dim=0usize,
    act_kappa=1.0f64, act_temp=0.5f64,
    profile_act_sims=None,
    road_edge_from=None, road_edge_to=None, road_edge_weight_ds=None,
    road_node_lats=None, road_node_lngs=None, location_road_node=None,
    max_leg_waypoints=16usize,
    gravity_deterrence_exponent=-2.0f64, gravity_origin_exponent=1.0f64,
    gravity_destination_exponent=1.0f64,
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
    ),
    (
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u16>>,
        Bound<'py, PyArray1<u16>>,
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

    let purpose_act_starts_v = opt_i64_as_usize_vec(&purpose_act_starts)?.unwrap_or_default();
    let purpose_acts_v = opt_i64_as_usize_vec(&purpose_acts)?.unwrap_or_default();

    let road_edge_from_v = opt_i64_as_usize_vec(&road_edge_from)?.unwrap_or_default();
    let road_edge_to_v = opt_i64_as_usize_vec(&road_edge_to)?.unwrap_or_default();
    let road_edge_weight_v = opt_i64_as_usize_vec(&road_edge_weight_ds)?.unwrap_or_default();
    let road_node_lats_s = opt_slice(&road_node_lats)?;
    let road_node_lngs_s = opt_slice(&road_node_lngs)?;
    let location_road_node_s = opt_slice(&location_road_node)?;

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
            callback
                .call1(py, (agent, dest_stop_id, seq, lat, lng, t))
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
                .call1(py, (agent, loc_id, arrival, departure, duration, stop_id, abstract_loc))
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
            callback
                .call1(py, (agent, stop_id, seq, activity, arrival, departure))
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
    });
    let on_activity_day_flush_ref = on_activity_day_flush_closure
        .as_mut()
        .map(|f| f as &mut dyn FnMut(ActivityOutputBuffers) -> Result<(), String>);

    let output = simulate(CoreInputs {
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
            emb_dim,
            kappa: act_kappa,
            temperature: act_temp,
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
    }, on_day_flush_ref, on_encounter_day_flush_ref, on_trip_day_flush_ref, on_activity_day_flush_ref)
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
        ),
        (
            output.act_agent.into_pyarray(py),
            output.act_stop_id.into_pyarray(py),
            output.act_seq.into_pyarray(py),
            output.act_activity.into_pyarray(py),
            output.act_arrival.into_pyarray(py),
            output.act_departure.into_pyarray(py),
        ),
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
        let (dist, conn) =
            batch_road_distances(&self.graph, from_nodes.as_slice()?, to_nodes.as_slice()?);
        let conn_u8: Vec<u8> = conn.into_iter().map(|b| b as u8).collect();
        Ok((dist.into_pyarray(py), conn_u8.into_pyarray(py)))
    }
}
