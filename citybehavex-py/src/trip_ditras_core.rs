//! Trip-duration-aware DITRAS.
//!
//! This is a copy of `skmob2_core::models::ditras` whose spatial
//! explore/return/diary-driven location selection is preserved byte-for-byte,
//! but whose timestamp emission is extended: each leg between two locations now
//! has a *trip duration* (car only, `distance / car_speed_kmh`), and the trip is
//! placed around the scheduled slot boundary `T` so that arrival/departure land
//! off the diary grid.
//!
//! Heuristic (slot length = `slot_seconds`):
//! - `dur <= slot_seconds`: depart at `T`, arrive at `T + dur` (within the slot).
//! - `dur >  slot_seconds`: center the trip on `T` -> depart `T - dur/2`,
//!   arrive `T + dur/2`.
//! Departures are clamped to be monotonic per agent (a stay cannot be left
//! before it was entered).
//!
//! Reused, unmodified helpers are imported from `skmob2-core`:
//! gravity OD rows, haversine distance, and the shared RNG choice helpers.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::sync::Arc;

use skmob2_core::models::od::{validate_equal_lengths, CachedGravityOdRows};
use skmob2_core::models::shared::{cdf_choice, derive_agent_seed, weighted_choice_excluding};
use skmob2_core::utils::haversine::haversine_km;

type TripDitrasResult =
    Result<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>, Vec<f64>), String>;
type TripOutputChunks<'a> = (
    &'a mut [i64], // agent ids
    &'a mut [f64], // lats
    &'a mut [f64], // lngs
    &'a mut [i64], // arrival_ts
    &'a mut [i64], // departure_ts
    &'a mut [f64], // trip_duration_seconds
);

const GRAVITY_REJECTION_ATTEMPTS: usize = 16;

struct DitrasThreadState {
    visited_locs: Vec<usize>,
    visit_counts: Vec<u32>,
    total_visits: f64,
    candidates: Vec<usize>,
    cdf: Vec<f64>,
}

impl DitrasThreadState {
    fn new(n: usize) -> Self {
        Self {
            visited_locs: Vec::with_capacity(200),
            visit_counts: vec![0u32; n],
            total_visits: 0.0,
            candidates: Vec::with_capacity(200),
            cdf: Vec::with_capacity(200),
        }
    }
}

fn record_visits(visited: &mut Vec<usize>, counts: &mut [u32], total: &mut f64, loc: usize) {
    if counts[loc] == 0 {
        visited.push(loc);
    }
    counts[loc] += 1;
    *total += 1.0;
}

fn clear_visits(visited: &mut Vec<usize>, counts: &mut [u32], total: &mut f64) {
    for &l in visited.iter() {
        counts[l] = 0;
    }
    visited.clear();
    *total = 0.0;
}

/// Gravity-weighted choice from unvisited locations, excluding `home`.
fn explore_location(
    rng: &mut impl Rng,
    row: &[f64],
    visit_counts: &[u32],
    home: usize,
    candidates: &mut Vec<usize>,
    cdf: &mut Vec<f64>,
) -> Option<usize> {
    if row.is_empty() {
        return None;
    }

    let total = row[row.len() - 1];
    if total.is_finite() && total > 0.0 {
        for _ in 0..GRAVITY_REJECTION_ATTEMPTS {
            let loc = cdf_choice(rng, row);
            if loc != home && visit_counts[loc] == 0 {
                return Some(loc);
            }
        }
    }

    candidates.clear();
    cdf.clear();

    let mut prev = 0.0_f64;
    let mut cumsum = 0.0_f64;
    for (j, &value) in row.iter().enumerate() {
        let weight = value - prev;
        prev = value;
        if j != home && visit_counts[j] == 0 && weight.is_finite() && weight > 0.0 {
            candidates.push(j);
            cumsum += weight;
            cdf.push(cumsum);
        }
    }

    if candidates.is_empty() {
        return None;
    }
    Some(candidates[cdf_choice(rng, cdf)])
}

/// Visit-count-weighted choice from non-home visited locations.
fn return_away(
    rng: &mut impl Rng,
    visited_locs: &[usize],
    visit_counts: &[u32],
    total_visits: f64,
    home: usize,
) -> Option<usize> {
    if !visited_locs.iter().any(|&l| l != home) {
        return None;
    }
    let loc = weighted_choice_excluding(rng, visited_locs, visit_counts, total_visits, home);
    if loc == home {
        None
    } else {
        Some(loc)
    }
}

/// Uniform choice from any unvisited location — last resort before staying put.
fn any_unvisited(
    rng: &mut impl Rng,
    visit_counts: &[u32],
    n: usize,
    candidates: &mut Vec<usize>,
) -> Option<usize> {
    candidates.clear();
    candidates.extend((0..n).filter(|&j| visit_counts[j] == 0));
    if candidates.is_empty() {
        None
    } else {
        Some(candidates[rng.gen_range(0..candidates.len())])
    }
}

fn emitted_len_for_agent(
    diary_timestamps: &[i64],
    diary_start: usize,
    diary_end: usize,
    end_ts: i64,
) -> usize {
    1 + ((diary_start + 1)..diary_end)
        .take_while(|&idx| diary_timestamps[idx] < end_ts)
        .count()
}

fn output_offsets(lengths: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(lengths.len() + 1);
    offsets.push(0);
    let mut total = 0usize;
    for &len in lengths {
        total += len;
        offsets.push(total);
    }
    offsets
}

#[allow(clippy::type_complexity)]
fn split_output_chunks<'a>(
    lengths: &[usize],
    mut out_agents: &'a mut [i64],
    mut out_lats: &'a mut [f64],
    mut out_lngs: &'a mut [f64],
    mut out_arr: &'a mut [i64],
    mut out_dep: &'a mut [i64],
    mut out_dur: &'a mut [f64],
) -> Vec<TripOutputChunks<'a>> {
    let mut chunks = Vec::with_capacity(lengths.len());
    for &len in lengths {
        let (agents_chunk, agents_rest) = out_agents.split_at_mut(len);
        let (lats_chunk, lats_rest) = out_lats.split_at_mut(len);
        let (lngs_chunk, lngs_rest) = out_lngs.split_at_mut(len);
        let (arr_chunk, arr_rest) = out_arr.split_at_mut(len);
        let (dep_chunk, dep_rest) = out_dep.split_at_mut(len);
        let (dur_chunk, dur_rest) = out_dur.split_at_mut(len);
        chunks.push((
            agents_chunk,
            lats_chunk,
            lngs_chunk,
            arr_chunk,
            dep_chunk,
            dur_chunk,
        ));
        out_agents = agents_rest;
        out_lats = lats_rest;
        out_lngs = lngs_rest;
        out_arr = arr_rest;
        out_dep = dep_rest;
        out_dur = dur_rest;
    }
    chunks
}

#[allow(clippy::too_many_arguments)]
fn simulate_one_trip_ditras_agent(
    agent_id: usize,
    home: usize,
    seed: u64,
    lats: &[f64],
    lngs: &[f64],
    od_rows: &CachedGravityOdRows,
    diary_timestamps: &[i64],
    diary_abs_locs: &[i32],
    diary_start: usize,
    diary_end: usize,
    rho: f64,
    gamma: f64,
    start_ts: i64,
    end_ts: i64,
    slot_seconds: i64,
    car_speed_kmh: f64,
    n: usize,
    output: TripOutputChunks<'_>,
    state: &mut DitrasThreadState,
) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut cur_loc = home;
    let (out_agents, out_lats, out_lngs, out_arr, out_dep, out_dur) = output;
    let mut out_idx = 0usize;

    // First record: the agent is home at start_ts.
    out_agents[out_idx] = agent_id as i64;
    out_lats[out_idx] = lats[cur_loc];
    out_lngs[out_idx] = lngs[cur_loc];
    out_arr[out_idx] = start_ts;
    out_dep[out_idx] = start_ts; // backfilled when (and if) the agent leaves
    out_dur[out_idx] = 0.0;
    out_idx += 1;
    record_visits(
        &mut state.visited_locs,
        &mut state.visit_counts,
        &mut state.total_visits,
        cur_loc,
    );

    // Diary index 0 is the start_ts home slot — iterate from index 1 onwards.
    for diary_idx in (diary_start + 1)..diary_end {
        let t = diary_timestamps[diary_idx];
        if t >= end_ts {
            break;
        }

        let abs_loc = diary_abs_locs[diary_idx];
        let next_loc = if abs_loc == 0 {
            home
        } else {
            let s = state.visited_locs.len();
            let explore =
                s == 1 || (s < n && rng.gen_range(0.0_f64..1.0) <= rho * (s as f64).powf(-gamma));

            if explore {
                let row: Arc<[f64]> = od_rows.get(cur_loc);
                explore_location(
                    &mut rng,
                    &row,
                    &state.visit_counts,
                    home,
                    &mut state.candidates,
                    &mut state.cdf,
                )
                .or_else(|| {
                    return_away(
                        &mut rng,
                        &state.visited_locs,
                        &state.visit_counts,
                        state.total_visits,
                        home,
                    )
                })
                .or_else(|| any_unvisited(&mut rng, &state.visit_counts, n, &mut state.candidates))
                .unwrap_or(cur_loc)
            } else {
                return_away(
                    &mut rng,
                    &state.visited_locs,
                    &state.visit_counts,
                    state.total_visits,
                    home,
                )
                .or_else(|| {
                    let row: Arc<[f64]> = od_rows.get(cur_loc);
                    explore_location(
                        &mut rng,
                        &row,
                        &state.visit_counts,
                        home,
                        &mut state.candidates,
                        &mut state.cdf,
                    )
                })
                .or_else(|| any_unvisited(&mut rng, &state.visit_counts, n, &mut state.candidates))
                .unwrap_or(cur_loc)
            }
        };

        // --- trip-duration heuristic -------------------------------------
        // Distance/time of the leg from the previous location to next_loc.
        let dur_s: i64 = if next_loc == cur_loc {
            0
        } else {
            let d_km = haversine_km(lats[cur_loc], lngs[cur_loc], lats[next_loc], lngs[next_loc]);
            let secs = (d_km / car_speed_kmh) * 3600.0;
            if secs.is_finite() && secs > 0.0 {
                secs.round() as i64
            } else {
                0
            }
        };

        // Departure relative to the scheduled slot boundary T.
        let mut departure = if dur_s <= slot_seconds {
            t // short trip: leave at the scheduled boundary, arrive within the slot
        } else {
            t - dur_s / 2 // long trip: center on T
        };
        // Monotonicity: cannot leave the previous stay before having arrived there.
        let prev_arrival = out_arr[out_idx - 1];
        if departure < prev_arrival {
            departure = prev_arrival;
        }
        let arrival = departure + dur_s;

        // Backfill the previous stay's departure (the agent left it now).
        out_dep[out_idx - 1] = departure;

        cur_loc = next_loc;
        out_agents[out_idx] = agent_id as i64;
        out_lats[out_idx] = lats[cur_loc];
        out_lngs[out_idx] = lngs[cur_loc];
        out_arr[out_idx] = arrival;
        out_dep[out_idx] = arrival; // provisional; backfilled by the next leg
        out_dur[out_idx] = dur_s as f64;
        out_idx += 1;
        record_visits(
            &mut state.visited_locs,
            &mut state.visit_counts,
            &mut state.total_visits,
            cur_loc,
        );
    }

    // The final stay lasts until the end of the simulation window.
    let last = out_idx - 1;
    out_dep[last] = end_ts.max(out_arr[last]);
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_trip_ditras_agents_impl(
    lats: &[f64],
    lngs: &[f64],
    relevances: &[f64],
    diary_timestamps: &[i64],
    diary_abs_locs: &[i32],
    diary_starts: &[usize],
    diary_ends: &[usize],
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    rho: f64,
    gamma: f64,
    start_ts: i64,
    end_ts: i64,
    slot_seconds: i64,
    car_speed_kmh: f64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs: Option<&[usize]>,
) -> TripDitrasResult {
    let n = validate_equal_lengths(&[
        ("latitudes", lats.len()),
        ("longitudes", lngs.len()),
        ("relevances", relevances.len()),
    ])?;
    if diary_starts.len() < n_agents || diary_ends.len() < n_agents {
        return Err(format!(
            "diary_starts/diary_ends must have at least {n_agents} entries"
        ));
    }
    if let Some(starts) = starting_locs {
        if starts.len() < n_agents {
            return Err(format!(
                "starting_locs must have at least {n_agents} entries"
            ));
        }
    }
    if slot_seconds <= 0 {
        return Err("slot_seconds must be positive".to_string());
    }
    if !(car_speed_kmh.is_finite() && car_speed_kmh > 0.0) {
        return Err("car_speed_kmh must be positive".to_string());
    }
    for agent in 0..n_agents {
        if diary_starts[agent] > diary_ends[agent] || diary_ends[agent] > diary_timestamps.len() {
            return Err("diary ranges must be ordered and within diary_timestamps".to_string());
        }
        if diary_ends[agent] > diary_abs_locs.len() {
            return Err("diary ranges must be within diary_abs_locs".to_string());
        }
    }
    if n == 0 || n_agents == 0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
    }

    let master_seed = master_seed.unwrap_or_else(rand::random);
    let od_rows = CachedGravityOdRows::new(
        lats,
        lngs,
        relevances,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
    );

    let output_lengths: Vec<usize> = (0..n_agents)
        .map(|agent| {
            emitted_len_for_agent(
                diary_timestamps,
                diary_starts[agent],
                diary_ends[agent],
                end_ts,
            )
        })
        .collect();
    let offsets = output_offsets(&output_lengths);
    let total = offsets[n_agents];
    let mut out_agents = vec![0_i64; total];
    let mut out_lats = vec![0.0_f64; total];
    let mut out_lngs = vec![0.0_f64; total];
    let mut out_arr = vec![0_i64; total];
    let mut out_dep = vec![0_i64; total];
    let mut out_dur = vec![0.0_f64; total];

    let output_chunks = split_output_chunks(
        &output_lengths,
        &mut out_agents,
        &mut out_lats,
        &mut out_lngs,
        &mut out_arr,
        &mut out_dep,
        &mut out_dur,
    );

    output_chunks
        .into_par_iter()
        .enumerate()
        .map_init(
            || DitrasThreadState::new(n),
            |state, (agent, output)| {
                let seed = derive_agent_seed(master_seed, agent, 0);
                let home = match starting_locs {
                    Some(sl) => sl[agent].min(n - 1),
                    None => {
                        let mut rng = Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(
                            master_seed,
                            agent,
                            1,
                        ));
                        rng.gen_range(0..n)
                    }
                };
                simulate_one_trip_ditras_agent(
                    agent + 1,
                    home,
                    seed,
                    lats,
                    lngs,
                    &od_rows,
                    diary_timestamps,
                    diary_abs_locs,
                    diary_starts[agent],
                    diary_ends[agent],
                    rho,
                    gamma,
                    start_ts,
                    end_ts,
                    slot_seconds,
                    car_speed_kmh,
                    n,
                    output,
                    state,
                );
                clear_visits(
                    &mut state.visited_locs,
                    &mut state.visit_counts,
                    &mut state.total_visits,
                );
            },
        )
        .for_each(drop);

    Ok((
        out_agents, out_lats, out_lngs, out_arr, out_dep, out_dur,
    ))
}
