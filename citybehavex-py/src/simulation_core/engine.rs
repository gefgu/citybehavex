use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use skmob2_core::models::od::{CachedGravityOdRows, validate_equal_lengths};
use skmob2_core::models::shared::derive_agent_seed;
use skmob2_core::utils::haversine::haversine_km;

use crate::simulation_core::activity::sample_activity_and_duration;
use crate::simulation_core::inputs::{
    ActivityInputs, CoreInputs, DiaryInputs, InitialLocationInputs, LocationInputs,
    RoadNetworkInputs, SimulationParams, SocialGraphInputs,
};
use crate::simulation_core::outputs::{
    ActivityOutputBuffers, RoadPathOutputBuffers, SimulationOutput, TripOutputBuffers,
};
use crate::simulation_core::roads::{RoadGraph, subsample_waypoints};
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
    let mut routed = None;
    if let Some(road) = road {
        let from_node = road
            .inputs
            .location_node
            .get(cur_loc)
            .copied()
            .unwrap_or(-1);
        let to_node = road
            .inputs
            .location_node
            .get(next_loc)
            .copied()
            .unwrap_or(-1);
        if from_node >= 0 && to_node >= 0 {
            match road
                .graph
                .shortest_path(road.calc, from_node as usize, to_node as usize)
            {
                Some((_weight_ds, nodes)) => {
                    let lats: Vec<f64> = nodes.iter().map(|&n| road.inputs.node_lats[n]).collect();
                    let lngs: Vec<f64> = nodes.iter().map(|&n| road.inputs.node_lngs[n]).collect();
                    let mut cumulative: Vec<i64> = Vec::with_capacity(nodes.len());
                    let mut acc: i64 = 0;
                    cumulative.push(0);
                    for w in nodes.windows(2) {
                        acc += road.graph.edge_weight_ds(w[0], w[1]);
                        cumulative.push(acc);
                    }
                    let dur_s = (acc + 5) / 10;
                    routed = Some((dur_s.max(0), lats, lngs, cumulative));
                }
                None => fallback.disconnected += 1,
            }
        } else {
            fallback.unsnapped += 1;
        }
    }
    routed.unwrap_or_else(|| {
        let dur_s = haversine_fallback_secs(cur_loc, next_loc, lats, lngs, car_speed_kmh);
        (
            dur_s,
            vec![lats[cur_loc], lats[next_loc]],
            vec![lngs[cur_loc], lngs[next_loc]],
            vec![0, dur_s * 10],
        )
    })
}

/// Resolve the new stop's departure/arrival, mark the agent's relocation
/// (`visit`/`current_location`), and push its stop row. Returns
/// `(departure, arrival, stop_id)`.
fn push_stop_record(
    ctx: &TripAppendContext,
    dur_s: i64,
    output: &mut TripOutputBuffers,
    agents: &mut [AgentState],
    next_stop_id: &mut u32,
) -> (i64, i64, u32) {
    let prev_idx = output.last_output_idx[ctx.agent_idx];
    let departure = ctx.departure.max(output.arrival[prev_idx] as i64);
    let arrival = departure + dur_s;
    output.departure[prev_idx] = departure as i32;

    agents[ctx.agent_idx].visit(ctx.next_loc);
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
    paths.push_leg(agent_id, stop_id, &sub_lats, &sub_lngs, &sub_times);
}

/// Close out the agent's current stop and open a new one at `ctx.next_loc`.
/// Only called for real relocations (`ctx.next_loc != cur_loc`) — same-
/// location abstract-location churn never reaches this function, so it
/// always routes a real trip. Returns `(departure, arrival)`: `departure` is
/// when the old stop/activity closed, `arrival` is when the new stop opened
/// (and thus when its first micro-activity, if any, starts).
fn append_trip_record(
    ctx: TripAppendContext,
    dur_s: i64,
    wp_lats: &[f64],
    wp_lngs: &[f64],
    wp_cum_ds: &[i64],
    output: &mut TripOutputBuffers,
    agents: &mut [AgentState],
    paths: &mut RoadPathOutputBuffers,
    next_stop_id: &mut u32,
) -> (i64, i64) {
    let (departure, arrival, stop_id) = push_stop_record(&ctx, dur_s, output, agents, next_stop_id);
    push_path_waypoints(
        &ctx, stop_id, departure, dur_s, wp_lats, wp_lngs, wp_cum_ds, paths,
    );

    (departure, arrival)
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
    if !(params.car_speed_kmh.is_finite() && params.car_speed_kmh > 0.0) {
        return Err("car_speed_kmh must be positive".to_string());
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
                active_abs_loc: 0,
                neighbor_indices: inputs.social_graph.neighbors[edge_start..edge_end].to_vec(),
                edge_sim: initial_edge_sim,
                edge_upd: init_ts_edges[edge_start..edge_end].to_vec(),
                encounters: Vec::new(),
                activity_counts: vec![0u32; inputs.activities.act_dur_mu.len()],
                pending_departure: 0,
                activity_seq: 0,
            }
        })
        .collect()
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
        start_activity_sample(
            i,
            0,
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
        if abstract_loc != data.active_abs_loc {
            let loc = match abstract_loc {
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
                    od_rows,
                    diary_abs_locs: inputs.diary.abstract_locations,
                    scratch: &mut data.scratch,
                    encounters: &mut data.encounters,
                }),
            };
            data.active_abs_loc = abstract_loc;
            data.moves.push((loc, ts, abstract_loc));
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
    commit_buf: &mut Vec<(i64, usize, usize, i32)>,
) {
    commit_buf.clear();
    for (a, data) in par_data.iter_mut().enumerate() {
        for &(loc, ts, abs_loc) in &data.moves {
            commit_buf.push((ts, a, loc, abs_loc));
        }
        data.moves.clear();
    }
    commit_buf.sort_unstable_by_key(|&(ts, a, _, _)| (ts, a));
}

fn current_stop_abstract_loc(a: usize, output: &TripOutputBuffers) -> i32 {
    output.abstract_loc[output.last_output_idx[a]] as i32
}

/// Samples and opens the next micro-activity for the currently-open stop.
#[allow(clippy::too_many_arguments)]
fn start_activity_sample(
    a: usize,
    abstract_loc: i32,
    arrival: i64,
    output: &TripOutputBuffers,
    par_data: &mut [AgentParData],
    activities_in: &ActivityInputs<'_>,
    activities: &mut ActivityOutputBuffers,
) {
    let current_stop_id = output.stop_id[output.last_output_idx[a]];
    let seq = par_data[a].activity_seq;
    par_data[a].activity_seq += 1;
    let AgentParData {
        ref mut activity_counts,
        ref mut rng,
        ref mut pending_departure,
        ref mut scratch,
        ..
    } = par_data[a];
    let (act_idx, dur) = sample_activity_and_duration(
        a,
        abstract_loc,
        activity_counts,
        rng,
        activities_in,
        scratch,
    );
    let new_idx = activities.push(a, current_stop_id, seq, arrival);
    activities.activity[new_idx] = act_idx as u16;
    *pending_departure = arrival + dur;
}

/// Close the currently-open micro-activity at `until`, sampling additional
/// same-stop activities as needed. The last sampled activity is truncated when
/// it would overrun `until`, so macro-schedule deadlines remain authoritative.
fn fill_activities_until(
    a: usize,
    abstract_loc: i32,
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
            next_arrival,
            output,
            par_data,
            activities_in,
            activities,
        );
    }
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
    agents: &mut [AgentState],
    par_data: &mut [AgentParData],
    output: &mut TripOutputBuffers,
    activities: &mut ActivityOutputBuffers,
    road_graph: Option<&RoadGraph>,
    road_calc: &mut Option<fast_paths::PathCalculator>,
    paths: &mut RoadPathOutputBuffers,
    fallback: &mut FallbackCounts,
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
        let (dur_s, wp_lats, wp_lngs, wp_cum_ds) = route_leg(
            cur_loc,
            loc,
            inputs.locations.lats,
            inputs.locations.lngs,
            inputs.params.car_speed_kmh,
            road_runtime.as_mut(),
            fallback,
        );
        let departure = if activities_on {
            let current_abs_loc = current_stop_abstract_loc(a, output);
            let latest_departure = (ts - dur_s - MACRO_DEPARTURE_BUFFER_S).max(prev_arrival);
            fill_activities_until(
                a,
                current_abs_loc,
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
        let (_, arrival) = append_trip_record(
            TripAppendContext {
                agent_idx: a,
                next_loc: loc,
                abstract_loc,
                departure,
                max_leg_waypoints: inputs.road_network.max_leg_waypoints,
            },
            dur_s,
            &wp_lats,
            &wp_lngs,
            &wp_cum_ds,
            output,
            agents,
            paths,
            next_stop_id,
        );
        par_data[a].activity_seq = 0;
        if activities_on {
            start_activity_sample(
                a,
                abstract_loc,
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
            ts.max(prev_arrival),
            output,
            par_data,
            &inputs.activities,
            activities,
        );
        start_activity_sample(
            a,
            abstract_loc,
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
    commit_buf: &[(i64, usize, usize, i32)],
    agents: &mut [AgentState],
    par_data: &mut [AgentParData],
    output: &mut TripOutputBuffers,
    activities: &mut ActivityOutputBuffers,
    road_graph: Option<&RoadGraph>,
    road_calc: &mut Option<fast_paths::PathCalculator>,
    paths: &mut RoadPathOutputBuffers,
    fallback: &mut FallbackCounts,
    inputs: &CoreInputs<'_>,
    activities_on: bool,
    next_stop_id: &mut u32,
) {
    for &(ts, a, loc, abstract_loc) in commit_buf {
        commit_one_move(
            ts,
            a,
            loc,
            abstract_loc,
            agents,
            par_data,
            output,
            activities,
            road_graph,
            road_calc,
            paths,
            fallback,
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
    let mut paths = RoadPathOutputBuffers::default();
    let mut fallback = FallbackCounts::default();

    let mut commit_buf: Vec<(i64, usize, usize, i32)> = Vec::new();
    let mut window_start = inputs.params.start_ts;

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
            &mut paths,
            &mut fallback,
            &inputs,
            activities_on,
            &mut next_stop_id,
        );
        update_all_edge_sims(
            &mut par_data,
            &agents,
            window_end,
            inputs.params.dt_update_mob_sim_s,
        );

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
    report_road_fallbacks(&inputs.road_network, &fallback);
    let (enc_agent, enc_contact, enc_tile, enc_ts) = flatten_encounters(&par_data);

    Ok(output.into_output(enc_agent, enc_contact, enc_tile, enc_ts, paths, activities))
}
