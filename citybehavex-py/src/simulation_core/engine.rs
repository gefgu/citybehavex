use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use fkmob_core::models::od::{CachedGravityOdRows, validate_equal_lengths};
use fkmob_core::models::shared::derive_agent_seed;
use fkmob_core::utils::haversine::haversine_km;
use std::collections::VecDeque;

use crate::simulation_core::activity::{
    COMMUTE_ACTIVITY_IDX, TRAVEL_ACTIVITY_IDX, sample_activity_and_duration,
};
use crate::simulation_core::inputs::{
    ActivityInputs, CoreInputs, DiaryInputs, InitialLocationInputs, LocationInputs,
    RoadNetworkInputs, SimulationParams, SocialGraphInputs,
};
use crate::simulation_core::outputs::{
    ActivityOutputBuffers, RoadPathOutputBuffers, SimulationOutput, TripOutputBuffers,
};
use citybehavex_core::roads::{RoadGraph, subsample_waypoints};
use crate::simulation_core::social::{
    LocationChoiceContext, choose_location_local, pick_starting_loc, update_edge_sim,
};
use crate::simulation_core::types::{AgentParData, AgentState, DiaryState, Scratch, WORK_CODE};

/// Prepared road graph plus a reusable path calculator, borrowed fresh for
/// each `append_trip_record` call (the commit loop is sequential, so a single
/// calculator can be reused across the whole run without synchronization).
struct RoadRuntime<'a> {
    graph: &'a RoadGraph,
    calc: &'a mut fast_paths::PathCalculator,
    inputs: &'a RoadNetworkInputs<'a>,
}

const MACRO_DEPARTURE_BUFFER_S: i64 = 900;
const MODE_CAR: u8 = 1;
const MODE_WALK: u8 = 2;
const MODE_BIKE: u8 = 3;
const MODE_RAIL: u8 = 4;

#[derive(Default)]
struct FallbackCounts {
    unsnapped: i64,
    disconnected: i64,
}

struct TripAppendContext {
    agent_idx: usize,
    next_loc: usize,
    /// Diary abstract-location code that triggered this relocation (0=HOME,
    /// 1=WORK, 2+=OTHER); recorded on the new stop so purpose can be read
    /// directly off it instead of re-derived from arrival timestamps.
    abstract_loc: i32,
    departure: i64,
    max_leg_waypoints: usize,
    mode: u8,
}

/// Resolve when the current dwell/travel unit ends, given the previously
/// sampled micro-activity's end time (`pending_departure`, 0 when activities
/// are disabled or none is pending), clamped to not precede `prev_arrival` nor
/// exceed `ts + slot_seconds`. Falls back to centering on `ts` (the diary's
/// scheduled move time) when there's no pending activity duration to honor.
fn resolve_departure(
    pending_departure: i64,
    prev_arrival: i64,
    ts: i64,
    slot_seconds: i64,
    dur_s: i64,
) -> i64 {
    let departure = if pending_departure > 0 {
        pending_departure.clamp(prev_arrival, ts + slot_seconds)
    } else if dur_s <= slot_seconds {
        ts
    } else {
        ts - dur_s / 2
    };
    departure.max(prev_arrival)
}

fn haversine_fallback_secs(
    cur_loc: usize,
    next_loc: usize,
    lats: &[f64],
    lngs: &[f64],
    car_speed_kmh: f64,
) -> i64 {
    let d_km = haversine_km(lats[cur_loc], lngs[cur_loc], lats[next_loc], lngs[next_loc]);
    let secs = (d_km / car_speed_kmh) * 3600.0;
    if secs.is_finite() && secs > 0.0 {
        secs.round() as i64
    } else {
        0
    }
}

fn haversine_secs_for_speed(
    cur_loc: usize,
    next_loc: usize,
    lats: &[f64],
    lngs: &[f64],
    speed_kmh: f64,
) -> i64 {
    let d_km = haversine_km(lats[cur_loc], lngs[cur_loc], lats[next_loc], lngs[next_loc]);
    let secs = (d_km / speed_kmh) * 3600.0;
    if secs.is_finite() && secs > 0.0 {
        secs.round() as i64
    } else {
        0
    }
}

fn straight_line_leg(
    cur_loc: usize,
    next_loc: usize,
    lats: &[f64],
    lngs: &[f64],
    speed_kmh: f64,
) -> (i64, Vec<f64>, Vec<f64>, Vec<i64>) {
    let dur_s = haversine_secs_for_speed(cur_loc, next_loc, lats, lngs, speed_kmh);
    (
        dur_s,
        vec![lats[cur_loc], lats[next_loc]],
        vec![lngs[cur_loc], lngs[next_loc]],
        vec![0, dur_s * 10],
    )
}

fn try_network_leg(
    cur_loc: usize,
    next_loc: usize,
    network: &mut RoadRuntime<'_>,
    fallback: &mut FallbackCounts,
) -> Option<(i64, Vec<f64>, Vec<f64>, Vec<i64>)> {
    let from_node = network
        .inputs
        .location_node
        .get(cur_loc)
        .copied()
        .unwrap_or(-1);
    let to_node = network
        .inputs
        .location_node
        .get(next_loc)
        .copied()
        .unwrap_or(-1);
    if from_node < 0 || to_node < 0 {
        fallback.unsnapped += 1;
        return None;
    }
    match network
        .graph
        .shortest_path(network.calc, from_node as usize, to_node as usize)
    {
        Some((_weight_ds, nodes)) => {
            let lats: Vec<f64> = nodes.iter().map(|&n| network.inputs.node_lats[n]).collect();
            let lngs: Vec<f64> = nodes.iter().map(|&n| network.inputs.node_lngs[n]).collect();
            let mut cumulative: Vec<i64> = Vec::with_capacity(nodes.len());
            let mut acc: i64 = 0;
            cumulative.push(0);
            for w in nodes.windows(2) {
                acc += network.graph.edge_weight_ds(w[0], w[1]);
                cumulative.push(acc);
            }
            let dur_s = (acc + 5) / 10;
            Some((dur_s.max(0), lats, lngs, cumulative))
        }
        None => {
            fallback.disconnected += 1;
            None
        }
    }
}

/// Route from `cur_loc` to `next_loc` over the road graph, falling back to a
/// straight-line haversine estimate when road routing is disabled or the
/// endpoints are unsnapped/disconnected. Returns (trip duration, waypoint
/// lats, waypoint lngs, cumulative weight at each waypoint in deciseconds).
fn route_leg(
    cur_loc: usize,
    next_loc: usize,
    lats: &[f64],
    lngs: &[f64],
    car_speed_kmh: f64,
    road: Option<&mut RoadRuntime<'_>>,
    fallback: &mut FallbackCounts,
) -> (i64, Vec<f64>, Vec<f64>, Vec<i64>) {
    if let Some(road) = road {
        if let Some(routed) = try_network_leg(cur_loc, next_loc, road, fallback) {
            return routed;
        }
    }
    let dur_s = haversine_fallback_secs(cur_loc, next_loc, lats, lngs, car_speed_kmh);
    (
        dur_s,
        vec![lats[cur_loc], lats[next_loc]],
        vec![lngs[cur_loc], lngs[next_loc]],
        vec![0, dur_s * 10],
    )
}

fn route_mode_leg(
    agent_idx: usize,
    cur_loc: usize,
    next_loc: usize,
    inputs: &CoreInputs<'_>,
    road: Option<&mut RoadRuntime<'_>>,
    rail: Option<&mut RoadRuntime<'_>>,
    road_fallback: &mut FallbackCounts,
    rail_fallback: &mut FallbackCounts,
) -> (u8, i64, Vec<f64>, Vec<f64>, Vec<i64>, usize) {
    let d_km = haversine_km(
        inputs.locations.lats[cur_loc],
        inputs.locations.lngs[cur_loc],
        inputs.locations.lats[next_loc],
        inputs.locations.lngs[next_loc],
    );
    let walk_threshold = inputs
        .transport
        .walking_threshold_km
        .get(agent_idx)
        .copied()
        .unwrap_or(0.0);
    if d_km <= walk_threshold {
        let (dur_s, lats, lngs, cum) = straight_line_leg(
            cur_loc,
            next_loc,
            inputs.locations.lats,
            inputs.locations.lngs,
            inputs.params.walking_speed_kmh,
        );
        return (MODE_WALK, dur_s, lats, lngs, cum, 2);
    }

    let has_car = inputs
        .transport
        .has_car
        .get(agent_idx)
        .copied()
        .unwrap_or(true);
    if has_car {
        let (dur_s, lats, lngs, cum) = route_leg(
            cur_loc,
            next_loc,
            inputs.locations.lats,
            inputs.locations.lngs,
            inputs.params.car_speed_kmh,
            road,
            road_fallback,
        );
        return (
            MODE_CAR,
            dur_s,
            lats,
            lngs,
            cum,
            inputs.road_network.max_leg_waypoints,
        );
    }

    let has_bike = inputs
        .transport
        .has_bike
        .get(agent_idx)
        .copied()
        .unwrap_or(false);
    let bike_threshold = inputs
        .transport
        .bike_threshold_km
        .get(agent_idx)
        .copied()
        .unwrap_or(0.0);
    if has_bike && d_km <= bike_threshold {
        let (dur_s, lats, lngs, cum) = straight_line_leg(
            cur_loc,
            next_loc,
            inputs.locations.lats,
            inputs.locations.lngs,
            inputs.params.bike_speed_kmh,
        );
        return (MODE_BIKE, dur_s, lats, lngs, cum, 2);
    }

    if let Some(rail) = rail {
        if let Some((dur_s, lats, lngs, cum)) =
            try_network_leg(cur_loc, next_loc, rail, rail_fallback)
        {
            return (
                MODE_RAIL,
                dur_s,
                lats,
                lngs,
                cum,
                inputs.rail_network.max_leg_waypoints,
            );
        }
    }

    let (dur_s, lats, lngs, cum) = route_leg(
        cur_loc,
        next_loc,
        inputs.locations.lats,
        inputs.locations.lngs,
        inputs.params.car_speed_kmh,
        road,
        road_fallback,
    );
    (
        MODE_CAR,
        dur_s,
        lats,
        lngs,
        cum,
        inputs.road_network.max_leg_waypoints,
    )
}

/// Resolve the new stop's departure/arrival, mark the agent's relocation
/// (`visit`/`current_location`), and push its stop row. Returns
/// `(departure, arrival, stop_id)`.
fn push_stop_record(
    ctx: &TripAppendContext,
    dur_s: i64,
    locations: &LocationInputs<'_>,
    output: &mut TripOutputBuffers,
    agents: &mut [AgentState],
    next_stop_id: &mut u32,
) -> (i64, i64, u32) {
    let prev_idx = output.last_output_idx[ctx.agent_idx];
    let departure = ctx.departure.max(output.arrival[prev_idx] as i64);
    let arrival = departure + dur_s;
    output.departure[prev_idx] = departure as i32;

    agents[ctx.agent_idx].visit(ctx.next_loc);
    if ctx.abstract_loc > WORK_CODE
        && let Some(&semantic_cluster) = locations.semantic_cluster_ids.get(ctx.next_loc)
    {
        agents[ctx.agent_idx].visit_poi_type(semantic_cluster);
    }
    agents[ctx.agent_idx].current_location = ctx.next_loc;

    let stop_id = *next_stop_id;
    *next_stop_id += 1;
    output.last_output_idx[ctx.agent_idx] = output.agents.len();
    output.agents.push(ctx.agent_idx as u32 + 1);
    output.loc_id.push(ctx.next_loc as u32);
    output.arrival.push(arrival as i32);
    output.departure.push(arrival as i32);
    output.duration.push(dur_s.max(0) as u32);
    output.stop_id.push(stop_id);
    output.abstract_loc.push(ctx.abstract_loc as u8);

    (departure, arrival, stop_id)
}

/// Distribute absolute waypoint timestamps proportionally along the path's
/// cumulative edge weight, so they land exactly on [departure, arrival]
/// regardless of any slot-width clamping applied upstream, then subsample and
/// push the leg.
#[allow(clippy::too_many_arguments)]
fn push_path_waypoints(
    ctx: &TripAppendContext,
    stop_id: u32,
    departure: i64,
    dur_s: i64,
    wp_lats: &[f64],
    wp_lngs: &[f64],
    wp_cum_ds: &[i64],
    paths: &mut RoadPathOutputBuffers,
) {
    let total_ds = *wp_cum_ds.last().unwrap_or(&0);
    let times: Vec<i64> = if total_ds <= 0 {
        wp_cum_ds.iter().map(|_| departure).collect()
    } else {
        wp_cum_ds
            .iter()
            .map(|&c| departure + ((c as f64 / total_ds as f64) * dur_s as f64).round() as i64)
            .collect()
    };
    let (sub_lats, sub_lngs, sub_times) =
        subsample_waypoints(wp_lats, wp_lngs, &times, ctx.max_leg_waypoints);
    let agent_id = ctx.agent_idx as u32 + 1;
    paths.push_leg(
        agent_id, stop_id, &sub_lats, &sub_lngs, &sub_times, ctx.mode,
    );
}

/// Close out the agent's current stop and open a new one at `ctx.next_loc`.
/// Only called for real relocations (`ctx.next_loc != cur_loc`) — same-
/// location abstract-location churn never reaches this function, so it
/// always routes a real trip. Returns `(departure, arrival, stop_id)`:
/// `departure` is when the old stop/activity closed, `arrival` is when the
/// new stop opened (and thus when its first micro-activity, if any, starts).
fn append_trip_record(
    ctx: TripAppendContext,
    dur_s: i64,
    wp_lats: &[f64],
    wp_lngs: &[f64],
    wp_cum_ds: &[i64],
    output: &mut TripOutputBuffers,
    agents: &mut [AgentState],
    locations: &LocationInputs<'_>,
    paths: &mut RoadPathOutputBuffers,
    next_stop_id: &mut u32,
) -> (i64, i64, u32) {
    let (departure, arrival, stop_id) =
        push_stop_record(&ctx, dur_s, locations, output, agents, next_stop_id);
    push_path_waypoints(
        &ctx, stop_id, departure, dur_s, wp_lats, wp_lngs, wp_cum_ds, paths,
    );

    (departure, arrival, stop_id)
}

fn validate_locations(locations: &LocationInputs<'_>) -> Result<usize, String> {
    let n_locations = validate_equal_lengths(&[
        ("latitudes", locations.lats.len()),
        ("longitudes", locations.lngs.len()),
        ("relevances", locations.relevances.len()),
    ])?;
    if n_locations < 2 {
        return Err("need at least 2 locations".to_string());
    }
    if n_locations > u32::MAX as usize {
        return Err(format!(
            "n_locations={} exceeds u32::MAX; location indices are stored as u32",
            n_locations
        ));
    }
    if !locations.distances.is_empty() && locations.distances.len() != n_locations * n_locations {
        return Err(format!(
            "distances must be empty or have length n_locations*n_locations={}, got {}",
            n_locations * n_locations,
            locations.distances.len()
        ));
    }
    if locations.poi_type_choice_enabled {
        if locations.semantic_cluster_ids.len() != n_locations {
            return Err(format!(
                "location_semantic_cluster_ids must have length n_locations={}, got {}",
                n_locations,
                locations.semantic_cluster_ids.len()
            ));
        }
        if locations.poi_type_n_blocks == 0 || locations.poi_type_n_clusters == 0 {
            return Err("POI type alignment dimensions must be positive when enabled".to_string());
        }
        let expected = locations.poi_type_n_blocks * locations.poi_type_n_clusters;
        if locations.poi_type_scores.len() % expected != 0 {
            return Err(format!(
                "poi_type_alignment_scores length must be a multiple of blocks*clusters={}, got {}",
                expected,
                locations.poi_type_scores.len()
            ));
        }
        if !(locations.poi_type_temperature.is_finite() && locations.poi_type_temperature > 0.0) {
            return Err("poi_type_choice_temperature must be positive".to_string());
        }
        if !(locations.poi_type_alpha.is_finite() && locations.poi_type_alpha >= 0.0) {
            return Err("poi_type_choice_alpha must be non-negative".to_string());
        }
    }
    Ok(n_locations)
}

fn validate_social_graph_lengths(
    social_graph: &SocialGraphInputs<'_>,
    n_agents: usize,
) -> Result<(), String> {
    if social_graph.neighbor_starts.len() != n_agents + 1 {
        return Err(format!(
            "neighbor_starts must have length n_agents+1={}, got {}",
            n_agents + 1,
            social_graph.neighbor_starts.len()
        ));
    }
    Ok(())
}

fn validate_diary_lengths(diary: &DiaryInputs<'_>, n_agents: usize) -> Result<(), String> {
    if diary.starts.len() < n_agents || diary.ends.len() < n_agents {
        return Err(format!(
            "diary_starts/diary_ends must have at least {} entries",
            n_agents
        ));
    }
    Ok(())
}

fn validate_params(params: &SimulationParams) -> Result<(), String> {
    if params.slot_seconds <= 0 {
        return Err("slot_seconds must be positive".to_string());
    }
    if params.indipendency_window_s <= 0 {
        return Err("indipendency_window_s must be positive".to_string());
    }
    if params.dt_update_mob_sim_s <= 0 {
        return Err("dt_update_mob_sim_s must be positive".to_string());
    }
    if params.friendship_update_interval_s <= 0 {
        return Err("friendship_update_interval_s must be positive".to_string());
    }
    if params.encounter_window_s <= 0 {
        return Err("encounter_window_s must be positive".to_string());
    }
    if !(params.regularity_threshold.is_finite()
        && (0.0..=1.0).contains(&params.regularity_threshold))
    {
        return Err("regularity_threshold must be in [0, 1]".to_string());
    }
    if !(params.topological_overlap_threshold.is_finite()
        && (0.0..=1.0).contains(&params.topological_overlap_threshold))
    {
        return Err("topological_overlap_threshold must be in [0, 1]".to_string());
    }
    if !(params.recast_random_chance_probability.is_finite()
        && params.recast_random_chance_probability > 0.0
        && params.recast_random_chance_probability <= 1.0)
    {
        return Err("recast_random_chance_probability must be in (0, 1]".to_string());
    }
    if !(params.strength_initial.is_finite() && params.strength_initial > 0.0) {
        return Err("strength_initial must be positive".to_string());
    }
    if !(params.strength_growth_sigma_ln.is_finite() && params.strength_growth_sigma_ln > 0.0) {
        return Err("strength_growth_sigma_ln must be positive".to_string());
    }
    if !(params.strength_decay_rate.is_finite()
        && (0.0..=1.0).contains(&params.strength_decay_rate))
    {
        return Err("strength_decay_rate must be in [0, 1]".to_string());
    }
    if params.max_dynamic_degree == 0 {
        return Err("max_dynamic_degree must be positive".to_string());
    }
    if params.max_colocation_group_size < 2 {
        return Err("max_colocation_group_size must be at least 2".to_string());
    }
    if !(params.car_speed_kmh.is_finite() && params.car_speed_kmh > 0.0) {
        return Err("car_speed_kmh must be positive".to_string());
    }
    if !(params.walking_speed_kmh.is_finite() && params.walking_speed_kmh > 0.0) {
        return Err("walking_speed_kmh must be positive".to_string());
    }
    if !(params.bike_speed_kmh.is_finite() && params.bike_speed_kmh > 0.0) {
        return Err("bike_speed_kmh must be positive".to_string());
    }
    Ok(())
}

fn validate_initial_locations(
    initial_locations: &InitialLocationInputs<'_>,
    n_agents: usize,
) -> Result<(), String> {
    if let Some(starts) = initial_locations.starting_locs
        && starts.len() < n_agents
    {
        return Err(format!(
            "starting_locs must have at least {} entries",
            n_agents
        ));
    }
    if initial_locations.work_tiles.len() < n_agents {
        return Err(format!(
            "work_tiles must have at least {} entries",
            n_agents
        ));
    }
    Ok(())
}

fn validate_transport(inputs: &CoreInputs<'_>, n_agents: usize) -> Result<(), String> {
    let transport = &inputs.transport;
    for (name, len) in [
        ("has_car", transport.has_car.len()),
        ("has_bike", transport.has_bike.len()),
        ("walking_threshold_km", transport.walking_threshold_km.len()),
        ("bike_threshold_km", transport.bike_threshold_km.len()),
    ] {
        if len < n_agents {
            return Err(format!("{name} must have at least {n_agents} entries"));
        }
    }
    Ok(())
}

fn validate_per_agent_ranges(inputs: &CoreInputs<'_>, n_agents: usize) -> Result<(), String> {
    for agent in 0..n_agents {
        if inputs.diary.starts[agent] > inputs.diary.ends[agent]
            || inputs.diary.ends[agent] > inputs.diary.timestamps.len()
        {
            return Err("diary ranges must be ordered and within diary_timestamps".to_string());
        }
        if inputs.diary.ends[agent] > inputs.diary.abstract_locations.len() {
            return Err("diary ranges must be within diary_abs_locs".to_string());
        }
        if inputs.diary.ends[agent] > inputs.diary.block_ids.len() {
            return Err("diary ranges must be within diary_block_ids".to_string());
        }
        if inputs.social_graph.neighbor_starts[agent]
            > inputs.social_graph.neighbor_starts[agent + 1]
            || inputs.social_graph.neighbor_starts[agent + 1] > inputs.social_graph.neighbors.len()
        {
            return Err("neighbor_starts must be ordered and within neighbors".to_string());
        }
    }
    Ok(())
}

fn validate_inputs(inputs: &CoreInputs<'_>) -> Result<usize, String> {
    if inputs.params.n_agents > u32::MAX as usize {
        return Err(format!(
            "n_agents={} exceeds u32::MAX; agent ids are stored as u32",
            inputs.params.n_agents
        ));
    }
    let n_locations = validate_locations(&inputs.locations)?;
    validate_social_graph_lengths(&inputs.social_graph, inputs.params.n_agents)?;
    validate_diary_lengths(&inputs.diary, inputs.params.n_agents)?;
    validate_params(&inputs.params)?;
    validate_initial_locations(&inputs.initial_locations, inputs.params.n_agents)?;
    validate_transport(inputs, inputs.params.n_agents)?;
    validate_per_agent_ranges(inputs, inputs.params.n_agents)?;
    Ok(n_locations)
}

fn build_od_rows<'a>(
    locations: &LocationInputs<'a>,
    params: &SimulationParams,
) -> Option<CachedGravityOdRows<'a>> {
    if locations.distances.is_empty() {
        Some(CachedGravityOdRows::new(
            locations.lats,
            locations.lngs,
            locations.relevances,
            "power_law",
            params.gravity_deterrence_exponent,
            params.gravity_origin_exponent,
            params.gravity_destination_exponent,
        ))
    } else {
        None
    }
}

fn new_agent_states(n_agents: usize) -> Vec<AgentState> {
    (0..n_agents).map(|_| AgentState::new()).collect()
}

/// Each agent's diary slot 0 (index `starts[i]`) is never revisited by
/// `resolve_agent_moves` -- `DiaryState::diary_idx` starts at 1, so slot 0 is
/// only ever consulted here, to seed the agent's true starting
/// abstract-location/block rather than a hardcoded placeholder.
fn initial_diary_state(inputs: &CoreInputs<'_>, agent: usize) -> (i32, i32) {
    let idx = inputs.diary.starts[agent];
    let abstract_loc = inputs
        .diary
        .abstract_locations
        .get(idx)
        .copied()
        .unwrap_or(0);
    let block_id = inputs.diary.block_ids.get(idx).copied().unwrap_or(-1);
    (abstract_loc, block_id)
}

fn new_agent_par_data(inputs: &CoreInputs<'_>, master_seed: u64) -> Vec<AgentParData> {
    let init_ts_edges = vec![inputs.params.start_ts; inputs.social_graph.neighbors.len()];
    (0..inputs.params.n_agents)
        .map(|i| {
            let edge_start = inputs.social_graph.neighbor_starts[i];
            let edge_end = inputs.social_graph.neighbor_starts[i + 1];
            let n_edges = edge_end - edge_start;
            let initial_edge_sim: Vec<f64> = if inputs.social_graph.edge_profile_sim.len()
                == inputs.social_graph.neighbors.len()
            {
                inputs.social_graph.edge_profile_sim[edge_start..edge_end].to_vec()
            } else {
                vec![0.0_f64; n_edges]
            };
            let (initial_abs_loc, initial_block_id) = initial_diary_state(inputs, i);
            AgentParData {
                rng: Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 0)),
                diary: DiaryState {
                    diary_start: inputs.diary.starts[i],
                    diary_end: inputs.diary.ends[i],
                    diary_idx: 1,
                },
                scratch: Scratch::new(),
                moves: Vec::with_capacity(32),
                active_day: 0,
                // Mirrors the real diary slot 0 state that
                // `sample_initial_activities` seeds below, so the first
                // comparison in `resolve_agent_moves` (at diary slot 1)
                // correctly detects whether anything actually changed.
                active_abs_loc: initial_abs_loc,
                active_block_id: initial_block_id,
                neighbor_indices: inputs.social_graph.neighbors[edge_start..edge_end].to_vec(),
                edge_sim: initial_edge_sim,
                edge_upd: init_ts_edges[edge_start..edge_end].to_vec(),
                edge_initial: vec![true; n_edges],
                encounters: Vec::new(),
                processed_encounters: 0,
                activity_counts: vec![0u32; inputs.activities.act_dur_mu.len()],
                last_activity: -1,
                pending_departure: 0,
                activity_seq: 0,
            }
        })
        .collect()
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

fn find_neighbor(row: &[usize], target: usize) -> Option<usize> {
    row.iter().position(|&x| x == target)
}

fn has_neighbor(par_data: &[AgentParData], a: usize, b: usize) -> bool {
    find_neighbor(&par_data[a].neighbor_indices, b).is_some()
}

fn topological_overlap(par_data: &[AgentParData], a: usize, b: usize) -> f64 {
    let left = &par_data[a].neighbor_indices;
    let right = &par_data[b].neighbor_indices;
    if left.is_empty() && right.is_empty() {
        return 0.0;
    }
    let mut set: FxHashSet<usize> = left.iter().copied().filter(|&x| x != b).collect();
    let mut intersection = 0usize;
    let mut union = set.len();
    for &nb in right {
        if nb == a {
            continue;
        }
        if set.remove(&nb) {
            intersection += 1;
        } else {
            union += 1;
        }
    }
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn random_baseline_overlap_threshold<R: Rng>(
    par_data: &[AgentParData],
    samples: usize,
    p_rnd: f64,
    rng: &mut R,
) -> f64 {
    if samples == 0 || par_data.len() < 2 {
        return 0.0;
    }
    let n = par_data.len();
    let mut values = Vec::with_capacity(samples);
    for _ in 0..samples {
        let a = rng.gen_range(0..n);
        let mut b = rng.gen_range(0..n - 1);
        if b >= a {
            b += 1;
        }
        values.push(topological_overlap(par_data, a, b));
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let q = (1.0 - p_rnd).clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * q).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn add_or_update_directed_edge(
    data: &mut AgentParData,
    target: usize,
    strength: f64,
    current_ts: i64,
    initial: bool,
) {
    if let Some(idx) = find_neighbor(&data.neighbor_indices, target) {
        data.edge_sim[idx] = data.edge_sim[idx].max(strength);
        data.edge_upd[idx] = current_ts;
        data.edge_initial[idx] |= initial;
    } else {
        data.neighbor_indices.push(target);
        data.edge_sim.push(strength);
        data.edge_upd.push(current_ts);
        data.edge_initial.push(initial);
    }
}

fn add_symmetric_dynamic_edge(
    par_data: &mut [AgentParData],
    a: usize,
    b: usize,
    strength: f64,
    current_ts: i64,
) {
    let (left, right) = par_data.split_at_mut(b.max(a));
    if a < b {
        add_or_update_directed_edge(&mut left[a], b, strength, current_ts, false);
        add_or_update_directed_edge(&mut right[0], a, strength, current_ts, false);
    } else if b < a {
        add_or_update_directed_edge(&mut right[0], b, strength, current_ts, false);
        add_or_update_directed_edge(&mut left[b], a, strength, current_ts, false);
    }
}

struct DynamicSocialState {
    observations: VecDeque<FxHashSet<(usize, usize)>>,
    window_updates: usize,
    next_update_ts: i64,
    rng: Xoshiro256PlusPlus,
}

impl DynamicSocialState {
    fn new(params: &SimulationParams, master_seed: u64) -> Self {
        let interval = params.friendship_update_interval_s.max(1);
        let window_updates =
            ((params.encounter_window_s + interval - 1) / interval).max(1) as usize;
        Self {
            observations: VecDeque::with_capacity(window_updates),
            window_updates,
            next_update_ts: params.start_ts + interval,
            rng: Xoshiro256PlusPlus::seed_from_u64(master_seed ^ 0x5eed_501a_1f5a_11ce),
        }
    }

    fn due(&self, ts: i64) -> bool {
        ts >= self.next_update_ts
    }

    fn advance(&mut self, params: &SimulationParams) {
        self.next_update_ts += params.friendship_update_interval_s;
    }
}

fn collect_colocation_pairs(
    agents: &[AgentState],
    max_group_size: usize,
) -> FxHashSet<(usize, usize)> {
    let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (agent, state) in agents.iter().enumerate() {
        groups
            .entry(state.current_location)
            .or_default()
            .push(agent);
    }
    let mut pairs = FxHashSet::default();
    for group in groups.values() {
        if group.len() < 2 || group.len() > max_group_size {
            continue;
        }
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                pairs.insert(ordered_pair(group[i], group[j]));
            }
        }
    }
    pairs
}

fn recent_pair_counts(
    observations: &VecDeque<FxHashSet<(usize, usize)>>,
) -> FxHashMap<(usize, usize), usize> {
    let mut counts: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    for window in observations {
        for &pair in window {
            *counts.entry(pair).or_insert(0) += 1;
        }
    }
    counts
}

fn apply_strength_updates(
    par_data: &mut [AgentParData],
    params: &SimulationParams,
    current_ts: i64,
    rng: &mut Xoshiro256PlusPlus,
) {
    let mut encountered: FxHashSet<(usize, usize)> = FxHashSet::default();
    for data in par_data.iter_mut() {
        for e in data.encounters.iter().skip(data.processed_encounters) {
            encountered.insert(ordered_pair(e.agent as usize, e.contact as usize));
        }
        data.processed_encounters = data.encounters.len();
    }

    for agent in 0..par_data.len() {
        for idx in 0..par_data[agent].neighbor_indices.len() {
            let nb = par_data[agent].neighbor_indices[idx];
            let pair = ordered_pair(agent, nb);
            if encountered.contains(&pair) {
                let z = sample_standard_normal(rng);
                let growth =
                    (params.strength_growth_mu_ln + params.strength_growth_sigma_ln * z).exp();
                par_data[agent].edge_sim[idx] += growth;
                par_data[agent].edge_upd[idx] = current_ts;
            } else {
                par_data[agent].edge_sim[idx] *= 1.0 - params.strength_decay_rate;
            }
            if par_data[agent].edge_sim[idx] < 1.0e-9 {
                par_data[agent].edge_sim[idx] = 1.0e-9;
            }
        }
    }
}

fn sample_standard_normal(rng: &mut Xoshiro256PlusPlus) -> f64 {
    let u1 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn update_dynamic_social_graph(
    state: &mut DynamicSocialState,
    par_data: &mut [AgentParData],
    agents: &[AgentState],
    params: &SimulationParams,
    current_ts: i64,
) {
    apply_strength_updates(par_data, params, current_ts, &mut state.rng);

    let colocated = collect_colocation_pairs(agents, params.max_colocation_group_size);
    state.observations.push_back(colocated);
    while state.observations.len() > state.window_updates {
        state.observations.pop_front();
    }
    let counts = recent_pair_counts(&state.observations);
    let denom = state.observations.len().max(1) as f64;
    let baseline = random_baseline_overlap_threshold(
        par_data,
        params.recast_random_baseline_samples,
        params.recast_random_chance_probability,
        &mut state.rng,
    );
    let overlap_threshold = params.topological_overlap_threshold.max(baseline);

    let mut promotions = Vec::new();
    for ((a, b), count) in counts {
        if has_neighbor(par_data, a, b) {
            continue;
        }
        if par_data[a].neighbor_indices.len() >= params.max_dynamic_degree
            || par_data[b].neighbor_indices.len() >= params.max_dynamic_degree
        {
            continue;
        }
        let regularity = count as f64 / denom;
        if regularity < params.regularity_threshold {
            continue;
        }
        let overlap = topological_overlap(par_data, a, b);
        if overlap < overlap_threshold {
            continue;
        }
        promotions.push((a, b));
    }

    for (a, b) in promotions {
        add_symmetric_dynamic_edge(par_data, a, b, params.strength_initial, current_ts);
    }
}

fn flatten_social_edges(par_data: &[AgentParData]) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<u8>) {
    let total: usize = par_data.iter().map(|d| d.neighbor_indices.len()).sum();
    let mut source = Vec::with_capacity(total);
    let mut target = Vec::with_capacity(total);
    let mut weight = Vec::with_capacity(total);
    let mut kind = Vec::with_capacity(total);
    for (agent, data) in par_data.iter().enumerate() {
        for idx in 0..data.neighbor_indices.len() {
            source.push(agent as u32);
            target.push(data.neighbor_indices[idx] as u32);
            weight.push(data.edge_sim[idx]);
            kind.push(if data.edge_initial[idx] { 0 } else { 1 });
        }
    }
    (source, target, weight, kind)
}

/// Assigns each agent's starting/HOME location and fixed WORK location.
/// Both are decided once here and never revisited for the rest of the run —
/// OTHER abstract-location codes are the only ones resolved dynamically
/// (see `resolve_agent_moves`).
fn init_agent_locations(
    agents: &mut [AgentState],
    par_data: &mut [AgentParData],
    inputs: &CoreInputs<'_>,
    n_locations: usize,
) {
    for (i, agent) in agents.iter_mut().enumerate() {
        let loc = if let Some(sl) = inputs.initial_locations.starting_locs {
            sl[i].min(n_locations - 1)
        } else {
            pick_starting_loc(
                inputs.locations.relevances,
                &mut par_data[i].rng,
                inputs.initial_locations.starting_locs_mode_relevance,
            )
        };
        agent.current_location = loc;
        agent.home_location = loc;
        agent.work_location = inputs.initial_locations.work_tiles[i].min(n_locations - 1);
        agent.visit(loc);
    }
}

fn sample_initial_activities(
    inputs: &CoreInputs<'_>,
    par_data: &mut [AgentParData],
    output: &TripOutputBuffers,
    activities: &mut ActivityOutputBuffers,
) {
    for i in 0..par_data.len() {
        let (abstract_loc, block_id) = initial_diary_state(inputs, i);
        start_activity_sample(
            i,
            abstract_loc,
            block_id,
            inputs.params.start_ts,
            output,
            par_data,
            &inputs.activities,
            activities,
        );
    }
}

fn build_road_graph(road_network: &RoadNetworkInputs<'_>) -> Option<RoadGraph> {
    if road_network.enabled() {
        println!("Preparing road-network contraction hierarchy ...");
        Some(RoadGraph::build(
            road_network.edge_from,
            road_network.edge_to,
            road_network.edge_weight_ds,
        ))
    } else {
        None
    }
}

fn build_rail_graph(rail_network: &RoadNetworkInputs<'_>) -> Option<RoadGraph> {
    if rail_network.enabled() {
        println!("Preparing rail-network contraction hierarchy ...");
        Some(RoadGraph::build(
            rail_network.edge_from,
            rail_network.edge_to,
            rail_network.edge_weight_ds,
        ))
    } else {
        None
    }
}

/// Walks one agent's diary up to `window_end`, resolving a physical location
/// for each genuine abstract-location transition: HOME/WORK are direct field
/// reads (fixed at init), everything else goes through the EPR return/
/// exploration decision fresh every time — no per-day memoization.
fn resolve_agent_moves(
    a: usize,
    data: &mut AgentParData,
    agents: &[AgentState],
    inputs: &CoreInputs<'_>,
    n_locations: usize,
    od_rows: Option<&CachedGravityOdRows<'_>>,
    window_end: i64,
) {
    // Tracks this agent's physical location across the whole call, since
    // `agents[a].current_location` only reflects the last *committed* move
    // and several moves can be queued here before the sequential commit
    // phase catches up.
    let mut known_location = agents[a].current_location;
    // Tracked locally rather than through `data.active_block_id`: that field
    // is read by `commit_one_move` to learn the *pre-transition* block id,
    // used to correctly close out `fill_activities_until`'s chain before
    // switching to the new block, and is only ever written there (after that
    // read). Updating it here instead -- before the sequential commit phase
    // has run -- would feed `fill_activities_until` the *new* block id
    // instead of the old one, corrupting every activity sampled up to the
    // transition. `pending_block_id` lets this loop still correctly detect
    // more than one block transition within a single window without
    // touching the field commit_one_move depends on.
    let mut pending_block_id = data.active_block_id;
    while let Some(ts) = data.diary.current_ts(inputs.diary.timestamps) {
        if ts >= window_end {
            break;
        }
        let day = (ts - inputs.params.start_ts).div_euclid(86_400);
        if day != data.active_day {
            data.active_day = day;
            if data.active_abs_loc != 0 {
                data.active_abs_loc = i32::MIN;
            }
        }

        let abstract_loc = data
            .diary
            .current_abstract_location(inputs.diary.abstract_locations);
        let block_id = data.diary.current_block_id(inputs.diary.block_ids);
        let abs_loc_changed = abstract_loc != data.active_abs_loc;
        // A block boundary without an abstract-location change (e.g. two
        // HOME episodes back to back, commonly across a day boundary when
        // the new day's diary also opens at HOME) still needs a fresh move
        // recorded: `active_block_id` drives the contextual alignment
        // lookup in `sample_activity_and_duration`, and without this check
        // it would silently keep scoring the whole new episode against the
        // *previous* episode's block id.
        let block_changed = block_id != pending_block_id;
        if abs_loc_changed || block_changed {
            let loc = if abs_loc_changed {
                match abstract_loc {
                    0 => agents[a].home_location,
                    WORK_CODE => agents[a].work_location,
                    _ => choose_location_local(LocationChoiceContext {
                        agent: a,
                        agents,
                        diary: &data.diary,
                        neighbor_indices: &data.neighbor_indices,
                        edge_sim: &data.edge_sim,
                        rng: &mut data.rng,
                        params: &inputs.params,
                        n_locations,
                        current_ts: ts,
                        locations: &inputs.locations,
                        activities: &inputs.activities,
                        od_rows,
                        diary_abs_locs: inputs.diary.abstract_locations,
                        diary_block_ids: inputs.diary.block_ids,
                        scratch: &mut data.scratch,
                        encounters: &mut data.encounters,
                    }),
                }
            } else {
                // Same abstract purpose, just a new block -- no real travel.
                known_location
            };
            data.active_abs_loc = abstract_loc;
            pending_block_id = block_id;
            data.moves.push((loc, ts, abstract_loc, block_id));
            known_location = loc;
        }
        data.diary
            .advance(inputs.diary.timestamps, inputs.params.end_ts);
    }
}

/// Phase A: resolve every agent's moves for the current window, in parallel.
fn resolve_moves_for_window(
    par_data: &mut [AgentParData],
    agents: &[AgentState],
    inputs: &CoreInputs<'_>,
    n_locations: usize,
    od_rows: Option<&CachedGravityOdRows<'_>>,
    window_end: i64,
) {
    par_data.par_iter_mut().enumerate().for_each(|(a, data)| {
        resolve_agent_moves(a, data, agents, inputs, n_locations, od_rows, window_end);
    });
}

/// Flattens each agent's per-window `moves` into one buffer sorted by
/// `(ts, agent)`, ready for sequential commit.
fn collect_sorted_moves(
    par_data: &mut [AgentParData],
    commit_buf: &mut Vec<(i64, usize, usize, i32, i32)>,
) {
    commit_buf.clear();
    for (a, data) in par_data.iter_mut().enumerate() {
        for &(loc, ts, abs_loc, block_id) in &data.moves {
            commit_buf.push((ts, a, loc, abs_loc, block_id));
        }
        data.moves.clear();
    }
    commit_buf.sort_unstable_by_key(|&(ts, a, _, _, _)| (ts, a));
}

fn current_stop_abstract_loc(a: usize, output: &TripOutputBuffers) -> i32 {
    output.abstract_loc[output.last_output_idx[a]] as i32
}

/// Samples and opens the next micro-activity for the currently-open stop.
#[allow(clippy::too_many_arguments)]
fn start_activity_sample(
    a: usize,
    abstract_loc: i32,
    block_id: i32,
    arrival: i64,
    output: &TripOutputBuffers,
    par_data: &mut [AgentParData],
    activities_in: &ActivityInputs<'_>,
    activities: &mut ActivityOutputBuffers,
) {
    let current_stop_id = output.stop_id[output.last_output_idx[a]];
    let current_location = output.loc_id[output.last_output_idx[a]] as usize;
    let seq = par_data[a].activity_seq;
    par_data[a].activity_seq += 1;
    let AgentParData {
        ref mut activity_counts,
        ref mut last_activity,
        ref mut rng,
        ref mut pending_departure,
        ref mut scratch,
        ..
    } = par_data[a];
    let (act_idx, dur) = sample_activity_and_duration(
        a,
        current_location,
        abstract_loc,
        block_id,
        *last_activity,
        activity_counts,
        rng,
        activities_in,
        scratch,
    );
    *last_activity = act_idx as i32;
    let new_idx = activities.push(a, current_stop_id, seq, arrival, block_id);
    activities.activity[new_idx] = act_idx as u16;
    *pending_departure = arrival + dur;
}

/// Close the currently-open micro-activity at `until`, sampling additional
/// same-stop activities as needed. The last sampled activity is truncated when
/// it would overrun `until`, so macro-schedule deadlines remain authoritative.
fn fill_activities_until(
    a: usize,
    abstract_loc: i32,
    block_id: i32,
    until: i64,
    output: &TripOutputBuffers,
    par_data: &mut [AgentParData],
    activities_in: &ActivityInputs<'_>,
    activities: &mut ActivityOutputBuffers,
) {
    loop {
        let last_idx = activities.last_idx[a];
        let last_arrival = activities.arrival[last_idx] as i64;
        let deadline = until.max(last_arrival);
        let pending_departure = par_data[a].pending_departure;
        if pending_departure <= 0 || pending_departure >= deadline {
            activities.departure[last_idx] = deadline as i32;
            par_data[a].pending_departure = 0;
            break;
        }

        let next_arrival = pending_departure.max(last_arrival);
        activities.departure[last_idx] = next_arrival as i32;
        start_activity_sample(
            a,
            abstract_loc,
            block_id,
            next_arrival,
            output,
            par_data,
            activities_in,
            activities,
        );
    }
}

fn can_materialize_travel_activity(inputs: &ActivityInputs<'_>) -> bool {
    inputs.materialize_travel && inputs.act_dur_mu.len() > TRAVEL_ACTIVITY_IDX
}

fn materialize_travel_activity(
    a: usize,
    stop_id: u32,
    abstract_loc: i32,
    block_id: i32,
    departure: i64,
    arrival: i64,
    par_data: &mut [AgentParData],
    activities: &mut ActivityOutputBuffers,
) {
    if arrival <= departure {
        return;
    }
    let act_idx = if abstract_loc == WORK_CODE {
        COMMUTE_ACTIVITY_IDX
    } else {
        TRAVEL_ACTIVITY_IDX
    };
    let seq = par_data[a].activity_seq;
    par_data[a].activity_seq += 1;
    if act_idx >= par_data[a].activity_counts.len() {
        par_data[a].activity_counts.resize(act_idx + 1, 0);
    }
    par_data[a].activity_counts[act_idx] += 1;
    par_data[a].last_activity = act_idx as i32;
    par_data[a].pending_departure = 0;
    let idx = activities.push(a, stop_id, seq, departure, block_id);
    activities.activity[idx] = act_idx as u16;
    activities.departure[idx] = arrival as i32;
}

/// Commits a single sorted move: either a real relocation (routes a trip,
/// opens a new stop) or a same-location abstract-location boundary (no
/// travel, just a departure/arrival split), then samples the next
/// micro-activity if activities are enabled.
#[allow(clippy::too_many_arguments)]
fn commit_one_move(
    ts: i64,
    a: usize,
    loc: usize,
    abstract_loc: i32,
    block_id: i32,
    agents: &mut [AgentState],
    par_data: &mut [AgentParData],
    output: &mut TripOutputBuffers,
    activities: &mut ActivityOutputBuffers,
    road_graph: Option<&RoadGraph>,
    road_calc: &mut Option<fast_paths::PathCalculator>,
    rail_graph: Option<&RoadGraph>,
    rail_calc: &mut Option<fast_paths::PathCalculator>,
    paths: &mut RoadPathOutputBuffers,
    road_fallback: &mut FallbackCounts,
    rail_fallback: &mut FallbackCounts,
    inputs: &CoreInputs<'_>,
    activities_on: bool,
    next_stop_id: &mut u32,
) {
    let cur_loc = agents[a].current_location;
    let is_new_location = loc != cur_loc;

    // Diary/abstract-location churn that resolves to the agent's current
    // physical tile is not a real move — with activities disabled there's
    // nothing to record for it at all (the stop table only reflects genuine
    // relocations).
    if !is_new_location && !activities_on {
        return;
    }

    // Lower clamp bound for the new departure/arrival: the currently open
    // micro-activity's arrival when activities are on (which may be later
    // than the stop's own arrival, if this stop already had other
    // activities sampled into it), else the stop's arrival.
    let prev_arrival: i64 = if activities_on {
        activities.arrival[activities.last_idx[a]] as i64
    } else {
        output.arrival[output.last_output_idx[a]] as i64
    };

    if is_new_location {
        let mut road_runtime = match (road_graph, road_calc.as_mut()) {
            (Some(g), Some(c)) => Some(RoadRuntime {
                graph: g,
                calc: c,
                inputs: &inputs.road_network,
            }),
            _ => None,
        };
        let mut rail_runtime = match (rail_graph, rail_calc.as_mut()) {
            (Some(g), Some(c)) => Some(RoadRuntime {
                graph: g,
                calc: c,
                inputs: &inputs.rail_network,
            }),
            _ => None,
        };
        let (mode, dur_s, wp_lats, wp_lngs, wp_cum_ds, max_leg_waypoints) = route_mode_leg(
            a,
            cur_loc,
            loc,
            inputs,
            road_runtime.as_mut(),
            rail_runtime.as_mut(),
            road_fallback,
            rail_fallback,
        );
        let departure = if activities_on {
            let current_abs_loc = current_stop_abstract_loc(a, output);
            let latest_departure = (ts - dur_s - MACRO_DEPARTURE_BUFFER_S).max(prev_arrival);
            fill_activities_until(
                a,
                current_abs_loc,
                par_data[a].active_block_id,
                latest_departure,
                output,
                par_data,
                &inputs.activities,
                activities,
            );
            latest_departure
        } else {
            resolve_departure(0, prev_arrival, ts, inputs.params.slot_seconds, dur_s)
        };
        let (_, arrival, stop_id) = append_trip_record(
            TripAppendContext {
                agent_idx: a,
                next_loc: loc,
                abstract_loc,
                departure,
                max_leg_waypoints,
                mode,
            },
            dur_s,
            &wp_lats,
            &wp_lngs,
            &wp_cum_ds,
            output,
            agents,
            &inputs.locations,
            paths,
            next_stop_id,
        );
        par_data[a].activity_seq = 0;
        if activities_on {
            if can_materialize_travel_activity(&inputs.activities) {
                materialize_travel_activity(
                    a,
                    stop_id,
                    abstract_loc,
                    block_id,
                    departure,
                    arrival,
                    par_data,
                    activities,
                );
            }
            par_data[a].active_block_id = block_id;
            start_activity_sample(
                a,
                abstract_loc,
                block_id,
                arrival,
                output,
                par_data,
                &inputs.activities,
                activities,
            );
        }
    } else {
        // Same physical location: the stop stays open, no waypoint leg, no
        // travel. Fill activities under the old abstract purpose up to the
        // macro boundary, then begin the new abstract purpose at exactly `ts`.
        let current_abs_loc = current_stop_abstract_loc(a, output);
        fill_activities_until(
            a,
            current_abs_loc,
            par_data[a].active_block_id,
            ts.max(prev_arrival),
            output,
            par_data,
            &inputs.activities,
            activities,
        );
        par_data[a].active_block_id = block_id;
        start_activity_sample(
            a,
            abstract_loc,
            block_id,
            ts.max(prev_arrival),
            output,
            par_data,
            &inputs.activities,
            activities,
        );
    }
}

/// Phase B: commits every sorted move for the current window, sequentially
/// (routing/output buffers are not safe to update in parallel).
#[allow(clippy::too_many_arguments)]
fn commit_moves(
    commit_buf: &[(i64, usize, usize, i32, i32)],
    agents: &mut [AgentState],
    par_data: &mut [AgentParData],
    output: &mut TripOutputBuffers,
    activities: &mut ActivityOutputBuffers,
    road_graph: Option<&RoadGraph>,
    road_calc: &mut Option<fast_paths::PathCalculator>,
    rail_graph: Option<&RoadGraph>,
    rail_calc: &mut Option<fast_paths::PathCalculator>,
    paths: &mut RoadPathOutputBuffers,
    road_fallback: &mut FallbackCounts,
    rail_fallback: &mut FallbackCounts,
    inputs: &CoreInputs<'_>,
    activities_on: bool,
    next_stop_id: &mut u32,
) {
    for &(ts, a, loc, abstract_loc, block_id) in commit_buf {
        commit_one_move(
            ts,
            a,
            loc,
            abstract_loc,
            block_id,
            agents,
            par_data,
            output,
            activities,
            road_graph,
            road_calc,
            rail_graph,
            rail_calc,
            paths,
            road_fallback,
            rail_fallback,
            inputs,
            activities_on,
            next_stop_id,
        );
    }
}

/// Phase C: update stale edge similarities using the committed agent state.
/// Each agent writes only its own edge_sim/edge_upd; agents is read-only.
fn update_all_edge_sims(
    par_data: &mut [AgentParData],
    agents: &[AgentState],
    window_end: i64,
    dt_update_mob_sim_s: i64,
) {
    par_data.par_iter_mut().enumerate().for_each(|(a, data)| {
        update_edge_sim(
            a,
            agents,
            &data.neighbor_indices,
            &mut data.edge_sim,
            &mut data.edge_upd,
            window_end,
            dt_update_mob_sim_s,
        );
    });
}

fn finalize_open_stops(output: &mut TripOutputBuffers, end_ts: i64) {
    for &idx in &output.last_output_idx {
        output.departure[idx] = end_ts.max(output.arrival[idx] as i64) as i32;
    }
}

fn finalize_open_activities(activities: &mut ActivityOutputBuffers, end_ts: i64) {
    for &idx in &activities.last_idx {
        activities.departure[idx] = end_ts.max(activities.arrival[idx] as i64) as i32;
    }
}

fn report_road_fallbacks(road_network: &RoadNetworkInputs<'_>, fallback: &FallbackCounts) {
    if road_network.enabled() {
        println!(
            "Road routing: {} fallbacks to straight-line (unsnapped={}, disconnected={})",
            fallback.unsnapped + fallback.disconnected,
            fallback.unsnapped,
            fallback.disconnected
        );
    }
}

fn report_rail_fallbacks(rail_network: &RoadNetworkInputs<'_>, fallback: &FallbackCounts) {
    if rail_network.enabled() {
        println!(
            "Rail routing: {} fallbacks to car (unsnapped={}, disconnected={})",
            fallback.unsnapped + fallback.disconnected,
            fallback.unsnapped,
            fallback.disconnected
        );
    }
}

fn flatten_encounters(par_data: &[AgentParData]) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<i32>) {
    let mut enc_agent: Vec<u32> = Vec::new();
    let mut enc_contact: Vec<u32> = Vec::new();
    let mut enc_tile: Vec<u32> = Vec::new();
    let mut enc_ts: Vec<i32> = Vec::new();
    for data in par_data {
        for e in &data.encounters {
            enc_agent.push(e.agent + 1);
            enc_contact.push(e.contact + 1);
            enc_tile.push(e.tile);
            enc_ts.push(e.ts);
        }
    }
    (enc_agent, enc_contact, enc_tile, enc_ts)
}

#[allow(clippy::type_complexity)]
pub(crate) fn simulate(
    inputs: CoreInputs<'_>,
    mut on_day_flush: Option<&mut dyn FnMut(RoadPathOutputBuffers) -> Result<(), String>>,
    mut on_encounter_day_flush: Option<
        &mut dyn FnMut((Vec<u32>, Vec<u32>, Vec<u32>, Vec<i32>)) -> Result<(), String>,
    >,
    mut on_trip_day_flush: Option<&mut dyn FnMut(TripOutputBuffers) -> Result<(), String>>,
    mut on_activity_day_flush: Option<&mut dyn FnMut(ActivityOutputBuffers) -> Result<(), String>>,
) -> Result<SimulationOutput, String> {
    let n_locations = validate_inputs(&inputs)?;
    if inputs.params.n_agents == 0 {
        return Ok(SimulationOutput::empty());
    }
    let activities_on = inputs.activities.enabled();
    let master_seed = inputs.params.master_seed.unwrap_or_else(rand::random);
    let od_rows = build_od_rows(&inputs.locations, &inputs.params);

    let mut agents = new_agent_states(inputs.params.n_agents);
    let mut par_data = new_agent_par_data(&inputs, master_seed);
    init_agent_locations(&mut agents, &mut par_data, &inputs, n_locations);

    // Monotonic stop id counter, decoupled from `TripOutputBuffers`'s array
    // length so day-boundary compaction (which physically removes closed
    // rows) never causes id reuse or collisions. Lives outside the struct
    // (not a field) so it can't be silently reset when the struct is
    // replaced during compaction (`*self = residual` in `take_day_chunk`).
    let mut next_stop_id: u32 = 0;
    let mut output =
        TripOutputBuffers::with_initial_agents(&agents, inputs.params.start_ts, &mut next_stop_id);
    let mut activities = ActivityOutputBuffers::with_capacity(inputs.params.n_agents);
    if activities_on {
        sample_initial_activities(&inputs, &mut par_data, &output, &mut activities);
    }

    let road_graph = build_road_graph(&inputs.road_network);
    let mut road_calc = road_graph.as_ref().map(|g| g.new_calculator());
    let rail_graph = build_rail_graph(&inputs.rail_network);
    let mut rail_calc = rail_graph.as_ref().map(|g| g.new_calculator());
    let mut paths = RoadPathOutputBuffers::default();
    let mut road_fallback = FallbackCounts::default();
    let mut rail_fallback = FallbackCounts::default();

    let mut commit_buf: Vec<(i64, usize, usize, i32, i32)> = Vec::new();
    let mut window_start = inputs.params.start_ts;
    let mut dynamic_social = if inputs.params.dynamic_friendships_enabled {
        Some(DynamicSocialState::new(&inputs.params, master_seed))
    } else {
        None
    };

    // Tracking simulation days and duration
    let mut current_day = 0;
    let mut day_start_instant = std::time::Instant::now();

    println!(
        "Starting simulation from {} to {} with {} agents and {} locations",
        inputs.params.start_ts, inputs.params.end_ts, inputs.params.n_agents, n_locations
    );

    while window_start < inputs.params.end_ts {
        // Calculate the current day based on simulated elapsed seconds (86400s in a day)
        let simulated_day = (window_start - inputs.params.start_ts).div_euclid(86_400);

        // Print and reset timer if we've crossed into a new day
        if simulated_day > current_day {
            println!(
                "Day {} simulated in {:.3?}",
                current_day,
                day_start_instant.elapsed()
            );
            if let Some(flush) = on_day_flush.as_deref_mut() {
                let chunk = paths.take_day_chunk();
                if !chunk.agent.is_empty() {
                    flush(chunk)?;
                }
            }
            if let Some(flush) = on_encounter_day_flush.as_deref_mut() {
                let chunk = flatten_encounters(&par_data);
                if !chunk.0.is_empty() {
                    flush(chunk)?;
                }
                for data in par_data.iter_mut() {
                    data.encounters.clear();
                    data.processed_encounters = 0;
                }
            }
            if let Some(flush) = on_trip_day_flush.as_deref_mut() {
                let chunk = output.take_day_chunk();
                if !chunk.agents.is_empty() {
                    flush(chunk)?;
                }
            }
            if let Some(flush) = on_activity_day_flush.as_deref_mut() {
                let chunk = activities.take_day_chunk();
                if !chunk.agent.is_empty() {
                    flush(chunk)?;
                }
            }
            current_day = simulated_day;
            day_start_instant = std::time::Instant::now();
        }

        let window_end =
            (window_start + inputs.params.indipendency_window_s).min(inputs.params.end_ts);

        resolve_moves_for_window(
            &mut par_data,
            &agents,
            &inputs,
            n_locations,
            od_rows.as_ref(),
            window_end,
        );
        collect_sorted_moves(&mut par_data, &mut commit_buf);
        commit_moves(
            &commit_buf,
            &mut agents,
            &mut par_data,
            &mut output,
            &mut activities,
            road_graph.as_ref(),
            &mut road_calc,
            rail_graph.as_ref(),
            &mut rail_calc,
            &mut paths,
            &mut road_fallback,
            &mut rail_fallback,
            &inputs,
            activities_on,
            &mut next_stop_id,
        );
        if let Some(state) = dynamic_social.as_mut() {
            while state.due(window_end) {
                update_dynamic_social_graph(
                    state,
                    &mut par_data,
                    &agents,
                    &inputs.params,
                    window_end,
                );
                state.advance(&inputs.params);
            }
        } else {
            update_all_edge_sims(
                &mut par_data,
                &agents,
                window_end,
                inputs.params.dt_update_mob_sim_s,
            );
        }

        window_start = window_end;
    }

    // Print duration for the final (potentially partial) day simulated
    println!(
        "Day {} simulated in {:.3?}",
        current_day,
        day_start_instant.elapsed()
    );

    finalize_open_stops(&mut output, inputs.params.end_ts);
    if activities_on {
        finalize_open_activities(&mut activities, inputs.params.end_ts);
    }
    report_road_fallbacks(&inputs.road_network, &road_fallback);
    report_rail_fallbacks(&inputs.rail_network, &rail_fallback);
    let (enc_agent, enc_contact, enc_tile, enc_ts) = flatten_encounters(&par_data);
    let (social_source, social_target, social_weight, social_kind) =
        flatten_social_edges(&par_data);

    Ok(output.into_output(
        enc_agent,
        enc_contact,
        enc_tile,
        enc_ts,
        paths,
        activities,
        social_source,
        social_target,
        social_weight,
        social_kind,
    ))
}
