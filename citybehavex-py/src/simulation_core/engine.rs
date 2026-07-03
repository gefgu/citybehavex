use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use skmob2_core::models::od::{CachedGravityOdRows, validate_equal_lengths};
use skmob2_core::models::shared::derive_agent_seed;
use skmob2_core::utils::haversine::haversine_km;

use crate::simulation_core::activity::sample_activity_and_duration;
use crate::simulation_core::inputs::{CoreInputs, RoadNetworkInputs};
use crate::simulation_core::outputs::{
    ActivityOutputBuffers, RoadPathOutputBuffers, SimulationOutput, TripOutputBuffers,
};
use crate::simulation_core::roads::{RoadGraph, subsample_waypoints};
use crate::simulation_core::social::{
    LocationChoiceContext, choose_location_local, pick_starting_loc, update_edge_sim,
};
use crate::simulation_core::types::{AgentParData, AgentState, DiaryState, Scratch};

/// Prepared road graph plus a reusable path calculator, borrowed fresh for
/// each `append_trip_record` call (the commit loop is sequential, so a single
/// calculator can be reused across the whole run without synchronization).
struct RoadRuntime<'a> {
    graph: &'a RoadGraph,
    calc: &'a mut fast_paths::PathCalculator,
    inputs: &'a RoadNetworkInputs<'a>,
}

#[derive(Default)]
struct FallbackCounts {
    unsnapped: i64,
    disconnected: i64,
}

struct TripAppendContext<'a> {
    agent_idx: usize,
    next_loc: usize,
    ts: i64,
    lats: &'a [f64],
    lngs: &'a [f64],
    car_speed_kmh: f64,
    slot_seconds: i64,
    pending_departure: i64,
    /// Arrival of the most recent event in this agent's timeline — the
    /// currently-open stop's arrival when activities are disabled, or the
    /// currently-open micro-activity's arrival when enabled (which can be
    /// later than the stop's own arrival once a stop has had more than one
    /// activity sampled into it). Used as the lower clamp bound so a new
    /// event can never be resolved to start before the previous one did.
    prev_arrival: i64,
    max_leg_waypoints: usize,
}

/// Resolve when the current dwell/travel unit ends, given the previously
/// sampled micro-activity's end time (`pending_departure`, 0 when activities
/// are disabled or none is pending), clamped to not precede `prev_arrival` nor
/// exceed `ts + slot_seconds`. Falls back to centering on `ts` (the diary's
/// scheduled move time) when there's no pending activity duration to honor.
fn resolve_departure(pending_departure: i64, prev_arrival: i64, ts: i64, slot_seconds: i64, dur_s: i64) -> i64 {
    let departure = if pending_departure > 0 {
        pending_departure.clamp(prev_arrival, ts + slot_seconds)
    } else if dur_s <= slot_seconds {
        ts
    } else {
        ts - dur_s / 2
    };
    departure.max(prev_arrival)
}

fn haversine_fallback_secs(cur_loc: usize, next_loc: usize, lats: &[f64], lngs: &[f64], car_speed_kmh: f64) -> i64 {
    let d_km = haversine_km(lats[cur_loc], lngs[cur_loc], lats[next_loc], lngs[next_loc]);
    let secs = (d_km / car_speed_kmh) * 3600.0;
    if secs.is_finite() && secs > 0.0 {
        secs.round() as i64
    } else {
        0
    }
}

/// Close out the agent's current stop and open a new one at `ctx.next_loc`.
/// Only called for real relocations (`ctx.next_loc != cur_loc`) — same-
/// location abstract-location churn never reaches this function, so it
/// always routes a real trip. Returns `(departure, arrival)`: `departure` is
/// when the old stop/activity closed, `arrival` is when the new stop opened
/// (and thus when its first micro-activity, if any, starts).
fn append_trip_record(
    ctx: TripAppendContext<'_>,
    output: &mut TripOutputBuffers,
    agents: &mut [AgentState],
    road: Option<&mut RoadRuntime<'_>>,
    paths: &mut RoadPathOutputBuffers,
    fallback: &mut FallbackCounts,
) -> (i64, i64) {
    let cur_loc = agents[ctx.agent_idx].current_location;

    // (trip duration, waypoint lats, waypoint lngs, cumulative weight at each waypoint in deciseconds)
    let (dur_s, wp_lats, wp_lngs, wp_cum_ds): (i64, Vec<f64>, Vec<f64>, Vec<i64>) = {
        let mut routed = None;
        if let Some(road) = road {
            let from_node = road.inputs.location_node.get(cur_loc).copied().unwrap_or(-1);
            let to_node = road
                .inputs
                .location_node
                .get(ctx.next_loc)
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
            let dur_s = haversine_fallback_secs(cur_loc, ctx.next_loc, ctx.lats, ctx.lngs, ctx.car_speed_kmh);
            (
                dur_s,
                vec![ctx.lats[cur_loc], ctx.lats[ctx.next_loc]],
                vec![ctx.lngs[cur_loc], ctx.lngs[ctx.next_loc]],
                vec![0, dur_s * 10],
            )
        })
    };

    let prev_idx = output.last_output_idx[ctx.agent_idx];
    let departure = resolve_departure(ctx.pending_departure, ctx.prev_arrival, ctx.ts, ctx.slot_seconds, dur_s);
    let arrival = departure + dur_s;
    output.departure[prev_idx] = departure;

    agents[ctx.agent_idx].visit(ctx.next_loc);
    agents[ctx.agent_idx].current_location = ctx.next_loc;

    let stop_id = output.agents.len() as i64;
    output.last_output_idx[ctx.agent_idx] = output.agents.len();
    output.agents.push(ctx.agent_idx as i64 + 1);
    output.lats.push(ctx.lats[ctx.next_loc]);
    output.lngs.push(ctx.lngs[ctx.next_loc]);
    output.arrival.push(arrival);
    output.departure.push(arrival);
    output.duration.push(dur_s as f64);
    output.stop_id.push(stop_id);

    // Distribute absolute waypoint timestamps proportionally along the path's
    // cumulative edge weight, so they land exactly on [departure, arrival]
    // regardless of any slot-width clamping applied above.
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
        subsample_waypoints(&wp_lats, &wp_lngs, &times, ctx.max_leg_waypoints);
    let agent_id = ctx.agent_idx as i64 + 1;
    paths.push_leg(agent_id, stop_id, &sub_lats, &sub_lngs, &sub_times);

    (departure, arrival)
}

fn validate_inputs(inputs: &CoreInputs<'_>) -> Result<usize, String> {
    let n_locations = validate_equal_lengths(&[
        ("latitudes", inputs.locations.lats.len()),
        ("longitudes", inputs.locations.lngs.len()),
        ("relevances", inputs.locations.relevances.len()),
    ])?;
    if n_locations < 2 {
        return Err("need at least 2 locations".to_string());
    }
    if !inputs.locations.distances.is_empty()
        && inputs.locations.distances.len() != n_locations * n_locations
    {
        return Err(format!(
            "distances must be empty or have length n_locations*n_locations={}, got {}",
            n_locations * n_locations,
            inputs.locations.distances.len()
        ));
    }
    if inputs.social_graph.neighbor_starts.len() != inputs.params.n_agents + 1 {
        return Err(format!(
            "neighbor_starts must have length n_agents+1={}, got {}",
            inputs.params.n_agents + 1,
            inputs.social_graph.neighbor_starts.len()
        ));
    }
    if inputs.diary.starts.len() < inputs.params.n_agents
        || inputs.diary.ends.len() < inputs.params.n_agents
    {
        return Err(format!(
            "diary_starts/diary_ends must have at least {} entries",
            inputs.params.n_agents
        ));
    }
    if inputs.params.slot_seconds <= 0 {
        return Err("slot_seconds must be positive".to_string());
    }
    if inputs.params.indipendency_window_s <= 0 {
        return Err("indipendency_window_s must be positive".to_string());
    }
    if inputs.params.dt_update_mob_sim_s <= 0 {
        return Err("dt_update_mob_sim_s must be positive".to_string());
    }
    if !(inputs.params.car_speed_kmh.is_finite() && inputs.params.car_speed_kmh > 0.0) {
        return Err("car_speed_kmh must be positive".to_string());
    }
    if let Some(starts) = inputs.initial_locations.starting_locs
        && starts.len() < inputs.params.n_agents
    {
        return Err(format!(
            "starting_locs must have at least {} entries",
            inputs.params.n_agents
        ));
    }
    for agent in 0..inputs.params.n_agents {
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

    Ok(n_locations)
}

pub(crate) fn simulate(inputs: CoreInputs<'_>) -> Result<SimulationOutput, String> {
    let n_locations = validate_inputs(&inputs)?;
    if inputs.params.n_agents == 0 {
        return Ok(SimulationOutput::empty());
    }
    let activities_on = inputs.activities.enabled();

    let master_seed = inputs.params.master_seed.unwrap_or_else(rand::random);
    let od_rows = if inputs.locations.distances.is_empty() {
        Some(CachedGravityOdRows::new(
            inputs.locations.lats,
            inputs.locations.lngs,
            inputs.locations.relevances,
            "power_law",
            -2.0,
            1.0,
            1.0,
        ))
    } else {
        None
    };

    let mut agents: Vec<AgentState> = (0..inputs.params.n_agents)
        .map(|_| AgentState::new(n_locations))
        .collect();
    let init_ts_edges = vec![inputs.params.start_ts; inputs.social_graph.neighbors.len()];
    let mut par_data: Vec<AgentParData> = (0..inputs.params.n_agents)
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
                abstract_loc_cache: Vec::with_capacity(16),
                neighbor_indices: inputs.social_graph.neighbors[edge_start..edge_end].to_vec(),
                edge_sim: initial_edge_sim,
                edge_upd: init_ts_edges[edge_start..edge_end].to_vec(),
                encounters: Vec::new(),
                activity_counts: vec![0u32; inputs.activities.act_dur_mu.len()],
                pending_departure: 0,
                activity_seq: 0,
                explore_cache: None,
            }
        })
        .collect();

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
        if i < inputs.initial_locations.work_tiles.len() {
            agent.work_location = Some(inputs.initial_locations.work_tiles[i].min(n_locations - 1));
        }
        agent.visit(loc);
    }

    let mut output = TripOutputBuffers::with_initial_agents(
        &agents,
        inputs.locations.lats,
        inputs.locations.lngs,
        inputs.params.start_ts,
    );
    let mut activities = ActivityOutputBuffers::with_capacity(inputs.params.n_agents);

    if activities_on {
        for (i, data) in par_data
            .iter_mut()
            .enumerate()
            .take(inputs.params.n_agents)
        {
            let AgentParData {
                activity_counts,
                rng,
                pending_departure,
                activity_seq,
                scratch,
                ..
            } = data;
            let (act_idx, dur) =
                sample_activity_and_duration(i, 0, activity_counts, rng, &inputs.activities, scratch);
            let stop_id = output.stop_id[output.last_output_idx[i]];
            let new_idx = activities.push(i, stop_id, 0, inputs.params.start_ts);
            activities.activity[new_idx] = act_idx;
            *pending_departure = inputs.params.start_ts + dur;
            *activity_seq = 1;
        }
    }

    let road_graph = if inputs.road_network.enabled() {
        println!("Preparing road-network contraction hierarchy ...");
        Some(RoadGraph::build(
            inputs.road_network.edge_from,
            inputs.road_network.edge_to,
            inputs.road_network.edge_weight_ds,
        ))
    } else {
        None
    };
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
        inputs.params.start_ts,
        inputs.params.end_ts,
        inputs.params.n_agents,
        n_locations
    );

    while window_start < inputs.params.end_ts {
        // Calculate the current day based on simulated elapsed seconds (86400s in a day)
        let simulated_day = (window_start - inputs.params.start_ts).div_euclid(86_400);

        // Print and reset timer if we've crossed into a new day
        if simulated_day > current_day {
            println!("Day {} simulated in {:.3?}", current_day, day_start_instant.elapsed());
            current_day = simulated_day;
            day_start_instant = std::time::Instant::now();
        }

        let window_end =
            (window_start + inputs.params.indipendency_window_s).min(inputs.params.end_ts);

        par_data.par_iter_mut().enumerate().for_each(|(a, data)| {
            while let Some(ts) = data.diary.current_ts(inputs.diary.timestamps) {
                if ts >= window_end {
                    break;
                }
                let day = (ts - inputs.params.start_ts).div_euclid(86_400);
                if day != data.active_day {
                    data.active_day = day;
                    data.abstract_loc_cache.clear();
                    if data.active_abs_loc != 0 {
                        data.active_abs_loc = i32::MIN;
                    }
                }

                let abstract_loc = data
                    .diary
                    .current_abstract_location(inputs.diary.abstract_locations);
                if abstract_loc != data.active_abs_loc {
                    let loc = if abstract_loc == 0 {
                        agents[a].home_location
                    } else if let Some((_, loc)) = data
                        .abstract_loc_cache
                        .iter()
                        .find(|&&(cached_abs, _)| cached_abs == abstract_loc)
                    {
                        *loc
                    } else {
                        let loc = choose_location_local(LocationChoiceContext {
                            agent: a,
                            agents: &agents,
                            diary: &data.diary,
                            neighbor_indices: &data.neighbor_indices,
                            edge_sim: &data.edge_sim,
                            rng: &mut data.rng,
                            params: &inputs.params,
                            n_locations,
                            current_ts: ts,
                            locations: &inputs.locations,
                            od_rows: od_rows.as_ref(),
                            diary_abs_locs: inputs.diary.abstract_locations,
                            scratch: &mut data.scratch,
                            encounters: &mut data.encounters,
                            explore_cache: &mut data.explore_cache,
                        });
                        data.abstract_loc_cache.push((abstract_loc, loc));
                        loc
                    };
                    data.active_abs_loc = abstract_loc;
                    data.moves.push((loc, ts, abstract_loc));
                }
                data.diary
                    .advance(inputs.diary.timestamps, inputs.params.end_ts);
            }
        });

        commit_buf.clear();
        for (a, data) in par_data.iter_mut().enumerate() {
            for &(loc, ts, abs_loc) in &data.moves {
                commit_buf.push((ts, a, loc, abs_loc));
            }
            data.moves.clear();
        }
        commit_buf.sort_unstable_by_key(|&(ts, a, _, _)| (ts, a));
        for &(ts, a, loc, abstract_loc) in &commit_buf {
            let cur_loc = agents[a].current_location;
            let is_new_location = loc != cur_loc;

            // Diary/abstract-location churn that resolves to the agent's
            // current physical tile is not a real move — with activities
            // disabled there's nothing to record for it at all (the stop
            // table only reflects genuine relocations).
            if !is_new_location && !activities_on {
                continue;
            }

            let pending_departure = if activities_on {
                par_data[a].pending_departure
            } else {
                0
            };
            par_data[a].pending_departure = 0;

            // Lower clamp bound for the new departure/arrival: the currently
            // open micro-activity's arrival when activities are on (which may
            // be later than the stop's own arrival, if this stop already had
            // other activities sampled into it), else the stop's arrival.
            let prev_arrival = if activities_on {
                activities.arrival[activities.last_idx[a]]
            } else {
                output.arrival[output.last_output_idx[a]]
            };

            let (departure, arrival) = if is_new_location {
                let mut road_runtime = match (road_graph.as_ref(), road_calc.as_mut()) {
                    (Some(g), Some(c)) => Some(RoadRuntime {
                        graph: g,
                        calc: c,
                        inputs: &inputs.road_network,
                    }),
                    _ => None,
                };
                let (departure, arrival) = append_trip_record(
                    TripAppendContext {
                        agent_idx: a,
                        next_loc: loc,
                        ts,
                        lats: inputs.locations.lats,
                        lngs: inputs.locations.lngs,
                        car_speed_kmh: inputs.params.car_speed_kmh,
                        slot_seconds: inputs.params.slot_seconds,
                        pending_departure,
                        prev_arrival,
                        max_leg_waypoints: inputs.road_network.max_leg_waypoints,
                    },
                    &mut output,
                    &mut agents,
                    road_runtime.as_mut(),
                    &mut paths,
                    &mut fallback,
                );
                par_data[a].activity_seq = 0;
                (departure, arrival)
            } else {
                // Same physical location: the stop stays open, no waypoint
                // leg, no travel — the "departure" here is purely a boundary
                // between two micro-activities within the same stay.
                let departure = resolve_departure(pending_departure, prev_arrival, ts, inputs.params.slot_seconds, 0);
                (departure, departure)
            };

            if activities_on {
                activities.departure[activities.last_idx[a]] = departure;

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
                    &inputs.activities,
                    scratch,
                );
                let new_idx = activities.push(a, current_stop_id, seq, arrival);
                activities.activity[new_idx] = act_idx;
                *pending_departure = arrival + dur;
            }
        }

        // Phase C: update stale edge similarities using the committed agent state.
        // Each agent writes only its own edge_sim/edge_upd; agents is read-only.
        par_data.par_iter_mut().enumerate().for_each(|(a, data)| {
            update_edge_sim(
                a,
                &agents,
                &data.neighbor_indices,
                &mut data.edge_sim,
                &mut data.edge_upd,
                window_end,
                inputs.params.dt_update_mob_sim_s,
            );
        });

        window_start = window_end;
    }

    // Print duration for the final (potentially partial) day simulated
    println!("Day {} simulated in {:.3?}", current_day, day_start_instant.elapsed());

    for &idx in &output.last_output_idx {
        output.departure[idx] = inputs.params.end_ts.max(output.arrival[idx]);
    }
    if activities_on {
        for &idx in &activities.last_idx {
            activities.departure[idx] = inputs.params.end_ts.max(activities.arrival[idx]);
        }
    }

    if inputs.road_network.enabled() {
        println!(
            "Road routing: {} fallbacks to straight-line (unsnapped={}, disconnected={})",
            fallback.unsnapped + fallback.disconnected,
            fallback.unsnapped,
            fallback.disconnected
        );
    }

    let mut enc_agent: Vec<i64> = Vec::new();
    let mut enc_contact: Vec<i64> = Vec::new();
    let mut enc_tile: Vec<i64> = Vec::new();
    let mut enc_ts: Vec<i64> = Vec::new();
    for data in &par_data {
        for e in &data.encounters {
            enc_agent.push(e.agent as i64 + 1);
            enc_contact.push(e.contact as i64 + 1);
            enc_tile.push(e.tile as i64);
            enc_ts.push(e.ts);
        }
    }

    Ok(output.into_output(enc_agent, enc_contact, enc_tile, enc_ts, paths, activities))
}
