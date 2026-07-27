use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::CastError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::simulation_core::engine::simulate;
use crate::simulation_core::inputs::{
    ActivityInputs, CoreInputs, DiaryInputs, InitialLocationInputs, LocationInputs,
    RoadNetworkInputs, SimulationParams, SocialGraphInputs, TransportInputs,
};
use crate::simulation_core::outputs::{
    ActivityOutputBuffers, RoadPathOutputBuffers, TripOutputBuffers,
};
use fastmob_core::network::road_graph::{RoadGraph, batch_road_distances};

fn array_attr<'py, T: numpy::Element>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArray1<'py, T>> {
    obj.getattr(name)?
        .extract()
        .map_err(|e: CastError<'_, '_>| PyValueError::new_err(e.to_string()))
}

fn opt_array_attr<'py, T: numpy::Element>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<PyReadonlyArray1<'py, T>>> {
    let value = obj.getattr(name)?;
    if value.is_none() {
        Ok(None)
    } else {
        value
            .extract()
            .map(Some)
            .map_err(|e: CastError<'_, '_>| PyValueError::new_err(e.to_string()))
    }
}

fn bool_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    let value = obj.getattr(name)?;
    value.extract()
}

fn f64_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let value = obj.getattr(name)?;
    value.extract()
}

fn i64_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    let value = obj.getattr(name)?;
    value.extract()
}

fn usize_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    let value = obj.getattr(name)?;
    value.extract()
}

fn opt_u64_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<u64>> {
    let value = obj.getattr(name)?;
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract()?))
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
    locations,
    social_graph,
    diary,
    params,
    initial_locations,
    activities,
    road_network,
    rail_network,
    transport,
    on_day_flush=None,
    on_encounter_day_flush=None,
    on_trip_day_flush=None,
    on_activity_day_flush=None,
    return_social_edges=false
))]
pub fn simulation_core_simulate_agents<'py>(
    py: Python<'py>,
    locations: Bound<'py, PyAny>,
    social_graph: Bound<'py, PyAny>,
    diary: Bound<'py, PyAny>,
    params: Bound<'py, PyAny>,
    initial_locations: Bound<'py, PyAny>,
    activities: Bound<'py, PyAny>,
    road_network: Bound<'py, PyAny>,
    rail_network: Bound<'py, PyAny>,
    transport: Bound<'py, PyAny>,
    on_day_flush: Option<Py<PyAny>>,
    on_encounter_day_flush: Option<Py<PyAny>>,
    on_trip_day_flush: Option<Py<PyAny>>,
    on_activity_day_flush: Option<Py<PyAny>>,
    return_social_edges: bool,
) -> PyResult<Py<PyAny>> {
    let latitudes = array_attr::<f64>(&locations, "latitudes")?;
    let longitudes = array_attr::<f64>(&locations, "longitudes")?;
    let relevances = array_attr::<f64>(&locations, "relevances")?;
    let distances = array_attr::<f64>(&locations, "distances")?;
    let location_semantic_cluster_ids =
        array_attr::<i64>(&locations, "location_semantic_cluster_ids")?;
    let poi_type_alignment_scores = array_attr::<f64>(&locations, "poi_type_alignment_scores")?;

    let neighbor_starts = array_attr::<i64>(&social_graph, "neighbor_starts")?;
    let neighbors = array_attr::<i64>(&social_graph, "neighbors")?;
    let edge_profile_sim = array_attr::<f64>(&social_graph, "edge_profile_sim")?;

    let diary_timestamps = array_attr::<i64>(&diary, "diary_timestamps")?;
    let diary_abs_locs = array_attr::<i32>(&diary, "diary_abs_locs")?;
    let diary_starts = array_attr::<i64>(&diary, "diary_starts")?;
    let diary_ends = array_attr::<i64>(&diary, "diary_ends")?;
    let diary_block_ids = array_attr::<i32>(&diary, "diary_block_ids")?;

    let starting_locs = opt_array_attr::<i64>(&initial_locations, "starting_locs")?;
    let work_tiles = array_attr::<i64>(&initial_locations, "work_tiles")?;

    let act_embs = array_attr::<f64>(&activities, "act_embs")?;
    let act_dur_mu = array_attr::<f64>(&activities, "act_dur_mu")?;
    let act_dur_sigma = array_attr::<f64>(&activities, "act_dur_sigma")?;
    let purpose_act_starts = array_attr::<i64>(&activities, "purpose_act_starts")?;
    let purpose_acts = array_attr::<i64>(&activities, "purpose_acts")?;
    let profile_embs = array_attr::<f64>(&activities, "profile_embs")?;
    let profile_act_sims = array_attr::<f64>(&activities, "profile_act_sims")?;
    let activity_alignment_scores = array_attr::<f64>(&activities, "activity_alignment_scores")?;
    let activity_cluster_labels = array_attr::<i64>(&activities, "activity_cluster_labels")?;
    let poi_semantic_scores = array_attr::<f64>(&activities, "poi_semantic_scores")?;
    let poi_mask_starts = array_attr::<i64>(&activities, "poi_mask_starts")?;
    let poi_mask_activities = array_attr::<i64>(&activities, "poi_mask_activities")?;

    let road_edge_from = array_attr::<i64>(&road_network, "edge_from")?;
    let road_edge_to = array_attr::<i64>(&road_network, "edge_to")?;
    let road_edge_weight_ds = array_attr::<i64>(&road_network, "edge_weight_ds")?;
    let road_node_lats = array_attr::<f64>(&road_network, "node_lats")?;
    let road_node_lngs = array_attr::<f64>(&road_network, "node_lngs")?;
    let location_road_node = array_attr::<i64>(&road_network, "location_node")?;

    let rail_edge_from = array_attr::<i64>(&rail_network, "edge_from")?;
    let rail_edge_to = array_attr::<i64>(&rail_network, "edge_to")?;
    let rail_edge_weight_ds = array_attr::<i64>(&rail_network, "edge_weight_ds")?;
    let rail_node_lats = array_attr::<f64>(&rail_network, "node_lats")?;
    let rail_node_lngs = array_attr::<f64>(&rail_network, "node_lngs")?;
    let location_rail_node = array_attr::<i64>(&rail_network, "location_node")?;

    let has_car = array_attr::<bool>(&transport, "has_car")?;
    let has_bike = array_attr::<bool>(&transport, "has_bike")?;
    let walking_threshold_km = array_attr::<f64>(&transport, "walking_threshold_km")?;
    let bike_threshold_km = array_attr::<f64>(&transport, "bike_threshold_km")?;

    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let rels = relevances.as_slice()?;
    let dists = distances.as_slice()?;
    let dt_raw = diary_timestamps.as_slice()?;
    let da_raw = diary_abs_locs.as_slice()?;
    let db_raw = diary_block_ids.as_slice()?;

    let ns = i64_as_usize_vec(&neighbor_starts)?;
    let nb = i64_as_usize_vec(&neighbors)?;
    let ds = i64_as_usize_vec(&diary_starts)?;
    let de = i64_as_usize_vec(&diary_ends)?;

    let sl_buf = opt_i64_as_usize_vec(&starting_locs)?;
    let sl: Option<&[usize]> = sl_buf.as_deref();
    let wt_buf = i64_as_usize_vec(&work_tiles)?;

    let eps_buf = edge_profile_sim.as_slice()?.to_vec();
    let eps: &[f64] = &eps_buf;

    let activity_cluster_labels_v = i64_as_usize_vec(&activity_cluster_labels)?;
    let location_semantic_cluster_ids_v = i64_as_usize_vec(&location_semantic_cluster_ids)?;
    let poi_mask_starts_v = i64_as_usize_vec(&poi_mask_starts)?;
    let poi_mask_activities_v = i64_as_usize_vec(&poi_mask_activities)?;
    let purpose_act_starts_v = i64_as_usize_vec(&purpose_act_starts)?;
    let purpose_acts_v = i64_as_usize_vec(&purpose_acts)?;

    let road_edge_from_v = i64_as_usize_vec(&road_edge_from)?;
    let road_edge_to_v = i64_as_usize_vec(&road_edge_to)?;
    let road_edge_weight_v = i64_as_usize_vec(&road_edge_weight_ds)?;
    let rail_edge_from_v = i64_as_usize_vec(&rail_edge_from)?;
    let rail_edge_to_v = i64_as_usize_vec(&rail_edge_to)?;
    let rail_edge_weight_v = i64_as_usize_vec(&rail_edge_weight_ds)?;

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
                semantic_cluster_ids: &location_semantic_cluster_ids_v,
                poi_type_scores: poi_type_alignment_scores.as_slice()?,
                poi_type_choice_enabled: bool_attr(&locations, "poi_type_choice_enabled")?,
                poi_type_n_clusters: usize_attr(&locations, "poi_type_alignment_clusters")?,
                poi_type_n_blocks: usize_attr(&locations, "poi_type_alignment_blocks")?,
                poi_type_temperature: f64_attr(&locations, "poi_type_choice_temperature")?,
                poi_type_alpha: f64_attr(&locations, "poi_type_choice_alpha")?,
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
                rho: f64_attr(&params, "rho")?,
                gamma: f64_attr(&params, "gamma")?,
                alpha: f64_attr(&params, "alpha")?,
                gravity_deterrence_exponent: f64_attr(&params, "gravity_deterrence_exponent")?,
                gravity_origin_exponent: f64_attr(&params, "gravity_origin_exponent")?,
                gravity_destination_exponent: f64_attr(&params, "gravity_destination_exponent")?,
                start_ts: i64_attr(&params, "start_ts")?,
                end_ts: i64_attr(&params, "end_ts")?,
                indipendency_window_s: i64_attr(&params, "indipendency_window_s")?,
                dt_update_mob_sim_s: i64_attr(&params, "dt_update_mob_sim_s")?,
                slot_seconds: i64_attr(&params, "slot_seconds")?,
                car_speed_kmh: f64_attr(&params, "car_speed_kmh")?,
                walking_speed_kmh: f64_attr(&params, "walking_speed_kmh")?,
                bike_speed_kmh: f64_attr(&params, "bike_speed_kmh")?,
                n_agents: usize_attr(&params, "n_agents")?,
                master_seed: opt_u64_attr(&params, "master_seed")?,
                dynamic_friendships_enabled: bool_attr(&params, "dynamic_friendships_enabled")?,
                friendship_update_interval_s: i64_attr(&params, "friendship_update_interval_s")?,
                encounter_window_s: i64_attr(&params, "encounter_window_s")?,
                regularity_threshold: f64_attr(&params, "regularity_threshold")?,
                topological_overlap_threshold: f64_attr(&params, "topological_overlap_threshold")?,
                recast_random_baseline_samples: usize_attr(
                    &params,
                    "recast_random_baseline_samples",
                )?,
                recast_random_chance_probability: f64_attr(
                    &params,
                    "recast_random_chance_probability",
                )?,
                strength_initial: f64_attr(&params, "strength_initial")?,
                strength_growth_mu_ln: f64_attr(&params, "strength_growth_mu_ln")?,
                strength_growth_sigma_ln: f64_attr(&params, "strength_growth_sigma_ln")?,
                strength_decay_rate: f64_attr(&params, "strength_decay_rate")?,
                max_dynamic_degree: usize_attr(&params, "max_dynamic_degree")?,
                max_colocation_group_size: usize_attr(&params, "max_colocation_group_size")?,
            },
            initial_locations: InitialLocationInputs {
                starting_locs: sl,
                starting_locs_mode_relevance: bool_attr(
                    &initial_locations,
                    "starting_locs_mode_relevance",
                )?,
                work_tiles: &wt_buf,
            },
            activities: ActivityInputs {
                act_embs: act_embs.as_slice()?,
                act_dur_mu: act_dur_mu.as_slice()?,
                act_dur_sigma: act_dur_sigma.as_slice()?,
                purpose_act_starts: &purpose_act_starts_v,
                purpose_acts: &purpose_acts_v,
                profile_embs: profile_embs.as_slice()?,
                profile_act_sims: profile_act_sims.as_slice()?,
                contextual_scores: activity_alignment_scores.as_slice()?,
                cluster_labels: &activity_cluster_labels_v,
                n_clusters: usize_attr(&activities, "activity_alignment_clusters")?,
                n_blocks: usize_attr(&activities, "activity_alignment_blocks")?,
                n_previous: usize_attr(&activities, "activity_alignment_previous")?,
                poi_semantic_scores: poi_semantic_scores.as_slice()?,
                location_semantic_cluster_ids: &location_semantic_cluster_ids_v,
                poi_mask_starts: &poi_mask_starts_v,
                poi_mask_activities: &poi_mask_activities_v,
                n_poi_semantic_clusters: usize_attr(&activities, "poi_semantic_clusters")?,
                history_weight: f64_attr(&activities, "activity_history_weight")?,
                emb_dim: usize_attr(&activities, "emb_dim")?,
                kappa: f64_attr(&activities, "act_kappa")?,
                temperature: f64_attr(&activities, "act_temp")?,
                materialize_travel: bool_attr(&activities, "materialize_travel")?,
            },
            road_network: RoadNetworkInputs {
                edge_from: &road_edge_from_v,
                edge_to: &road_edge_to_v,
                edge_weight_ds: &road_edge_weight_v,
                node_lats: road_node_lats.as_slice()?,
                node_lngs: road_node_lngs.as_slice()?,
                location_node: location_road_node.as_slice()?,
                max_leg_waypoints: usize_attr(&road_network, "max_leg_waypoints")?,
            },
            rail_network: RoadNetworkInputs {
                edge_from: &rail_edge_from_v,
                edge_to: &rail_edge_to_v,
                edge_weight_ds: &rail_edge_weight_v,
                node_lats: rail_node_lats.as_slice()?,
                node_lngs: rail_node_lngs.as_slice()?,
                location_node: location_rail_node.as_slice()?,
                max_leg_waypoints: usize_attr(&rail_network, "max_leg_waypoints")?,
            },
            transport: TransportInputs {
                has_car: has_car.as_slice()?,
                has_bike: has_bike.as_slice()?,
                walking_threshold_km: walking_threshold_km.as_slice()?,
                bike_threshold_km: bike_threshold_km.as_slice()?,
            },
        },
        on_day_flush_ref,
        on_encounter_day_flush_ref,
        on_trip_day_flush_ref,
        on_activity_day_flush_ref,
    )
    .map_err(PyValueError::new_err)?;

    let trip = (
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
    );
    let paths = (
        output.stop_id.into_pyarray(py),
        output.path_agent.into_pyarray(py),
        output.path_stop_id.into_pyarray(py),
        output.path_seq.into_pyarray(py),
        output.path_lat.into_pyarray(py),
        output.path_lng.into_pyarray(py),
        output.path_t.into_pyarray(py),
        output.path_mode.into_pyarray(py),
    );
    let activities = (
        output.act_agent.into_pyarray(py),
        output.act_stop_id.into_pyarray(py),
        output.act_seq.into_pyarray(py),
        output.act_activity.into_pyarray(py),
        output.act_arrival.into_pyarray(py),
        output.act_departure.into_pyarray(py),
        output.act_block_id.into_pyarray(py),
    );
    if return_social_edges {
        let social = (
            output.social_source.into_pyarray(py),
            output.social_target.into_pyarray(py),
            output.social_weight.into_pyarray(py),
            output.social_kind.into_pyarray(py),
        );
        Ok((trip, paths, activities, social)
            .into_pyobject(py)?
            .unbind()
            .into())
    } else {
        Ok((trip, paths, activities).into_pyobject(py)?.unbind().into())
    }
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
        let (dist, conn) = py.detach(|| batch_road_distances(&self.graph, from_slice, to_slice));
        let conn_u8: Vec<u8> = conn.into_iter().map(|b| b as u8).collect();
        Ok((dist.into_pyarray(py), conn_u8.into_pyarray(py)))
    }
}
