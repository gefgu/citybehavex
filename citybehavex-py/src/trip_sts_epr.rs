//! Trip-duration-aware STS-EPR.
//!
//! This ports `skmob2_core::models::sts_epr` into the citybehavex extension and
//! adds the same trip-duration stay emission used by trip-DITRAS.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use skmob2_core::models::od::{CachedGravityOdRows, validate_equal_lengths};
use skmob2_core::models::shared::{cdf_choice, derive_agent_seed};
use skmob2_core::utils::haversine::haversine_km;

type TripStsEprResult =
    Result<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>, Vec<f64>), String>;

const GRAVITY_REJECTION_ATTEMPTS: usize = 16;

struct DiaryState {
    diary_start: usize,
    diary_end: usize,
    diary_idx: usize,
}

impl DiaryState {
    fn current_ts(&self, diary_timestamps: &[i64]) -> Option<i64> {
        let idx = self.diary_start + self.diary_idx;
        if idx < self.diary_end {
            Some(diary_timestamps[idx])
        } else {
            None
        }
    }

    fn current_abstract_location(&self, diary_abs_locs: &[i32]) -> i32 {
        let idx = self.diary_start + self.diary_idx;
        if idx < self.diary_end {
            diary_abs_locs[idx]
        } else {
            0
        }
    }

    fn advance(&mut self, diary_timestamps: &[i64], end_ts: i64) -> i64 {
        self.diary_idx += 1;
        self.current_ts(diary_timestamps).unwrap_or(end_ts + 3600)
    }
}

struct AgentState {
    current_location: usize,
    home_location: usize,
    visited_locs: Vec<usize>,
    visit_counts: Vec<u32>,
    total_visits: f64,
    s: f64,
}

impl AgentState {
    fn new(n_locations: usize) -> Self {
        Self {
            current_location: 0,
            home_location: 0,
            visited_locs: Vec::with_capacity(200),
            visit_counts: vec![0u32; n_locations],
            total_visits: 0.0,
            s: 0.0,
        }
    }

    fn visit(&mut self, loc: usize) {
        if self.visit_counts[loc] == 0 {
            self.s += 1.0;
            self.visited_locs.push(loc);
        }
        self.visit_counts[loc] += 1;
        self.total_visits += 1.0;
    }
}

struct Scratch {
    candidates: Vec<usize>,
    cdf: Vec<f64>,
    sims: Vec<f64>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            candidates: Vec::with_capacity(200),
            cdf: Vec::with_capacity(200),
            sims: Vec::with_capacity(64),
        }
    }
}

struct AgentParData {
    rng: Xoshiro256PlusPlus,
    diary: DiaryState,
    scratch: Scratch,
    moves: Vec<(usize, i64)>,
    neighbor_indices: Vec<usize>,
    edge_sim: Vec<f64>,
    edge_upd: Vec<i64>,
}

#[derive(Clone, Copy, PartialEq)]
enum SocialMode {
    Exploration,
    Return,
}

fn cosine_similarity_sparse(
    a_locs: &[usize],
    a_counts: &[u32],
    b_locs: &[usize],
    b_counts: &[u32],
) -> f64 {
    let dot: f64 = a_locs
        .iter()
        .map(|&l| (a_counts[l] as f64) * (b_counts[l] as f64))
        .sum();
    let norm_a: f64 = a_locs
        .iter()
        .map(|&l| (a_counts[l] as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b_locs
        .iter()
        .map(|&l| (b_counts[l] as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn cdf_sample(rng: &mut impl Rng, candidates: &[usize], cdf: &[f64]) -> usize {
    let total = cdf[cdf.len() - 1];
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let idx = cdf
        .partition_point(|&v| v <= threshold)
        .min(candidates.len() - 1);
    candidates[idx]
}

fn make_individual_return(
    agent: usize,
    agents: &[AgentState],
    rng: &mut impl Rng,
    scratch: &mut Scratch,
) -> Option<usize> {
    let a = &agents[agent];
    scratch.candidates.clear();
    scratch.cdf.clear();
    let mut cumsum = 0.0_f64;
    for &loc in &a.visited_locs {
        cumsum += a.visit_counts[loc] as f64;
        scratch.candidates.push(loc);
        scratch.cdf.push(cumsum);
    }
    if scratch.candidates.is_empty() || cumsum <= 0.0 {
        None
    } else {
        Some(cdf_sample(rng, &scratch.candidates, &scratch.cdf))
    }
}

fn sts_epr_exploration(
    agent: usize,
    agents: &[AgentState],
    distances: &[f64],
    od_rows: Option<&CachedGravityOdRows>,
    relevances: &[f64],
    n: usize,
    rng: &mut impl Rng,
    scratch: &mut Scratch,
) -> Option<usize> {
    let src = agents[agent].current_location;
    let home = agents[agent].home_location;
    let visit_counts = &agents[agent].visit_counts;

    if let Some(od_rows) = od_rows {
        let row = od_rows.get(src);
        if !row.is_empty() {
            let total = row[row.len() - 1];
            if total.is_finite() && total > 0.0 {
                for _ in 0..GRAVITY_REJECTION_ATTEMPTS {
                    let loc = cdf_choice(rng, &row);
                    if visit_counts[loc] == 0 && loc != src && loc != home {
                        return Some(loc);
                    }
                }
            }

            scratch.candidates.clear();
            scratch.cdf.clear();
            let mut prev = 0.0_f64;
            let mut cumsum = 0.0_f64;
            for (j, &value) in row.iter().enumerate() {
                let weight = value - prev;
                prev = value;
                if visit_counts[j] == 0
                    && j != src
                    && j != home
                    && weight.is_finite()
                    && weight > 0.0
                {
                    scratch.candidates.push(j);
                    cumsum += weight;
                    scratch.cdf.push(cumsum);
                }
            }
            if scratch.candidates.is_empty() {
                return None;
            }
            return Some(cdf_sample(rng, &scratch.candidates, &scratch.cdf));
        }
    }

    let src_rel = relevances[src];
    scratch.candidates.clear();
    scratch.cdf.clear();
    let mut all_zero = true;
    let mut cumsum = 0.0_f64;
    for j in 0..n {
        if visit_counts[j] != 0 || j == src || j == home {
            continue;
        }
        let d = distances[src * n + j].max(0.001);
        let score = (1.0 / (d * d)) * relevances[j] * src_rel;
        scratch.candidates.push(j);
        if score > 0.0 {
            all_zero = false;
        }
        cumsum += score;
        scratch.cdf.push(cumsum);
    }
    if scratch.candidates.is_empty() {
        return None;
    }
    if all_zero {
        return Some(scratch.candidates[rng.gen_range(0..scratch.candidates.len())]);
    }
    Some(cdf_sample(rng, &scratch.candidates, &scratch.cdf))
}

fn update_edge_sim_local(
    agent: usize,
    agents: &[AgentState],
    neighbor_indices: &[usize],
    edge_sim: &mut [f64],
    edge_upd: &mut [i64],
    current_ts: i64,
    dt_update_s: i64,
) {
    for i in 0..neighbor_indices.len() {
        if edge_upd[i] <= current_ts {
            let nb = neighbor_indices[i];
            edge_sim[i] = cosine_similarity_sparse(
                &agents[agent].visited_locs,
                &agents[agent].visit_counts,
                &agents[nb].visited_locs,
                &agents[nb].visit_counts,
            );
            edge_upd[i] = current_ts + dt_update_s;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn make_social_action_local(
    agent: usize,
    agents: &[AgentState],
    neighbor_indices: &[usize],
    edge_sim: &mut [f64],
    edge_upd: &mut [i64],
    mode: SocialMode,
    rng: &mut impl Rng,
    current_ts: i64,
    dt_update_s: i64,
    scratch: &mut Scratch,
) -> Option<usize> {
    if neighbor_indices.is_empty() {
        return None;
    }
    update_edge_sim_local(
        agent,
        agents,
        neighbor_indices,
        edge_sim,
        edge_upd,
        current_ts,
        dt_update_s,
    );

    scratch.sims.clear();
    scratch.sims.extend_from_slice(edge_sim);
    let contact_idx = if scratch.sims.iter().all(|&s| s == 0.0) {
        rng.gen_range(0..scratch.sims.len())
    } else {
        let total: f64 = scratch.sims.iter().sum();
        let threshold = rng.gen_range(0.0_f64..1.0) * total;
        let mut cumsum = 0.0_f64;
        let mut chosen = scratch.sims.len() - 1;
        for (i, &w) in scratch.sims.iter().enumerate() {
            cumsum += w;
            if cumsum > threshold {
                chosen = i;
                break;
            }
        }
        chosen
    };
    let contact = neighbor_indices[contact_idx];

    let agent_counts = &agents[agent].visit_counts;
    let contact_counts = &agents[contact].visit_counts;
    let contact_locs = &agents[contact].visited_locs;
    let agent_locs = &agents[agent].visited_locs;

    scratch.candidates.clear();
    scratch.cdf.clear();
    match mode {
        SocialMode::Exploration => {
            let mut cumsum = 0.0_f64;
            for &loc in contact_locs {
                if agent_counts[loc] == 0 {
                    scratch.candidates.push(loc);
                    cumsum += contact_counts[loc] as f64;
                    scratch.cdf.push(cumsum);
                }
            }
        }
        SocialMode::Return => {
            let mut cumsum = 0.0_f64;
            for &loc in agent_locs {
                let w = contact_counts[loc] as f64;
                if w > 0.0 {
                    scratch.candidates.push(loc);
                    cumsum += w;
                    scratch.cdf.push(cumsum);
                }
            }
        }
    }

    if scratch.candidates.is_empty() || scratch.cdf.last().copied().unwrap_or(0.0) <= 0.0 {
        None
    } else {
        Some(cdf_sample(rng, &scratch.candidates, &scratch.cdf))
    }
}

#[allow(clippy::too_many_arguments)]
fn choose_location_local(
    agent: usize,
    agents: &[AgentState],
    diary: &DiaryState,
    neighbor_indices: &[usize],
    edge_sim: &mut [f64],
    edge_upd: &mut [i64],
    rng: &mut impl Rng,
    rho: f64,
    gamma: f64,
    alpha: f64,
    n_locations: usize,
    current_ts: i64,
    dt_update_s: i64,
    distances: &[f64],
    od_rows: Option<&CachedGravityOdRows>,
    relevances: &[f64],
    diary_abs_locs: &[i32],
    scratch: &mut Scratch,
) -> usize {
    let abstract_location = diary.current_abstract_location(diary_abs_locs);
    if abstract_location == 0 {
        return agents[agent].home_location;
    }

    let s = agents[agent].s.max(1.0);
    let p_explore = rho * s.powf(-gamma);
    let explore = rng.gen_range(0.0_f64..1.0) < p_explore;
    let social = rng.gen_range(0.0_f64..1.0) < alpha;

    let location = if explore {
        if social {
            make_social_action_local(
                agent,
                agents,
                neighbor_indices,
                edge_sim,
                edge_upd,
                SocialMode::Exploration,
                rng,
                current_ts,
                dt_update_s,
                scratch,
            )
            .or_else(|| {
                sts_epr_exploration(
                    agent,
                    agents,
                    distances,
                    od_rows,
                    relevances,
                    n_locations,
                    rng,
                    scratch,
                )
            })
            .or_else(|| make_individual_return(agent, agents, rng, scratch))
        } else {
            sts_epr_exploration(
                agent,
                agents,
                distances,
                od_rows,
                relevances,
                n_locations,
                rng,
                scratch,
            )
            .or_else(|| {
                make_social_action_local(
                    agent,
                    agents,
                    neighbor_indices,
                    edge_sim,
                    edge_upd,
                    SocialMode::Exploration,
                    rng,
                    current_ts,
                    dt_update_s,
                    scratch,
                )
            })
            .or_else(|| make_individual_return(agent, agents, rng, scratch))
        }
    } else if social {
        make_social_action_local(
            agent,
            agents,
            neighbor_indices,
            edge_sim,
            edge_upd,
            SocialMode::Return,
            rng,
            current_ts,
            dt_update_s,
            scratch,
        )
        .or_else(|| make_individual_return(agent, agents, rng, scratch))
        .or_else(|| {
            sts_epr_exploration(
                agent,
                agents,
                distances,
                od_rows,
                relevances,
                n_locations,
                rng,
                scratch,
            )
        })
    } else {
        make_individual_return(agent, agents, rng, scratch)
            .or_else(|| {
                make_social_action_local(
                    agent,
                    agents,
                    neighbor_indices,
                    edge_sim,
                    edge_upd,
                    SocialMode::Return,
                    rng,
                    current_ts,
                    dt_update_s,
                    scratch,
                )
            })
            .or_else(|| {
                sts_epr_exploration(
                    agent,
                    agents,
                    distances,
                    od_rows,
                    relevances,
                    n_locations,
                    rng,
                    scratch,
                )
            })
    };

    location.unwrap_or(agents[agent].current_location)
}

fn pick_starting_loc(relevances: &[f64], rng: &mut impl Rng, mode_relevance: bool) -> usize {
    let n = relevances.len();
    if !mode_relevance {
        return rng.gen_range(0..n);
    }
    let total: f64 = relevances.iter().sum();
    if total <= 0.0 {
        return rng.gen_range(0..n);
    }
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let mut cumsum = 0.0;
    for (i, &r) in relevances.iter().enumerate() {
        cumsum += r;
        if cumsum > threshold {
            return i;
        }
    }
    n - 1
}

#[allow(clippy::too_many_arguments)]
fn append_trip_record(
    agent_idx: usize,
    next_loc: usize,
    t: i64,
    lats: &[f64],
    lngs: &[f64],
    car_speed_kmh: f64,
    slot_seconds: i64,
    out_agents: &mut Vec<i64>,
    out_lats: &mut Vec<f64>,
    out_lngs: &mut Vec<f64>,
    out_arr: &mut Vec<i64>,
    out_dep: &mut Vec<i64>,
    out_dur: &mut Vec<f64>,
    last_output_idx: &mut [usize],
    agents: &mut [AgentState],
) {
    let cur_loc = agents[agent_idx].current_location;
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

    let mut departure = if dur_s <= slot_seconds {
        t
    } else {
        t - dur_s / 2
    };
    let prev_idx = last_output_idx[agent_idx];
    let prev_arrival = out_arr[prev_idx];
    if departure < prev_arrival {
        departure = prev_arrival;
    }
    let arrival = departure + dur_s;
    out_dep[prev_idx] = departure;

    agents[agent_idx].visit(next_loc);
    agents[agent_idx].current_location = next_loc;

    last_output_idx[agent_idx] = out_agents.len();
    out_agents.push(agent_idx as i64 + 1);
    out_lats.push(lats[next_loc]);
    out_lngs.push(lngs[next_loc]);
    out_arr.push(arrival);
    out_dep.push(arrival);
    out_dur.push(dur_s as f64);
}

#[allow(clippy::too_many_arguments)]
fn simulate_trip_sts_epr_impl(
    lats: &[f64],
    lngs: &[f64],
    relevances: &[f64],
    distances: &[f64],
    neighbor_starts: &[usize],
    neighbors: &[usize],
    diary_timestamps: &[i64],
    diary_abs_locs: &[i32],
    diary_starts: &[usize],
    diary_ends: &[usize],
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
    starting_locs: Option<&[usize]>,
    starting_locs_mode_relevance: bool,
) -> TripStsEprResult {
    let n_locations = validate_equal_lengths(&[
        ("latitudes", lats.len()),
        ("longitudes", lngs.len()),
        ("relevances", relevances.len()),
    ])?;
    if n_locations < 2 {
        return Err("need at least 2 locations".to_string());
    }
    if !distances.is_empty() && distances.len() != n_locations * n_locations {
        return Err(format!(
            "distances must be empty or have length n_locations*n_locations={}, got {}",
            n_locations * n_locations,
            distances.len()
        ));
    }
    if neighbor_starts.len() != n_agents + 1 {
        return Err(format!(
            "neighbor_starts must have length n_agents+1={}, got {}",
            n_agents + 1,
            neighbor_starts.len()
        ));
    }
    if diary_starts.len() < n_agents || diary_ends.len() < n_agents {
        return Err(format!(
            "diary_starts/diary_ends must have at least {n_agents} entries"
        ));
    }
    if slot_seconds <= 0 {
        return Err("slot_seconds must be positive".to_string());
    }
    if indipendency_window_s <= 0 {
        return Err("indipendency_window_s must be positive".to_string());
    }
    if dt_update_mob_sim_s <= 0 {
        return Err("dt_update_mob_sim_s must be positive".to_string());
    }
    if !(car_speed_kmh.is_finite() && car_speed_kmh > 0.0) {
        return Err("car_speed_kmh must be positive".to_string());
    }
    if let Some(starts) = starting_locs {
        if starts.len() < n_agents {
            return Err(format!(
                "starting_locs must have at least {n_agents} entries"
            ));
        }
    }
    for agent in 0..n_agents {
        if diary_starts[agent] > diary_ends[agent] || diary_ends[agent] > diary_timestamps.len() {
            return Err("diary ranges must be ordered and within diary_timestamps".to_string());
        }
        if diary_ends[agent] > diary_abs_locs.len() {
            return Err("diary ranges must be within diary_abs_locs".to_string());
        }
        if neighbor_starts[agent] > neighbor_starts[agent + 1]
            || neighbor_starts[agent + 1] > neighbors.len()
        {
            return Err("neighbor_starts must be ordered and within neighbors".to_string());
        }
    }
    if n_agents == 0 {
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
    let od_rows = if distances.is_empty() {
        Some(CachedGravityOdRows::new(
            lats,
            lngs,
            relevances,
            "power_law",
            -2.0,
            1.0,
            1.0,
        ))
    } else {
        None
    };

    let mut agents: Vec<AgentState> = (0..n_agents)
        .map(|_| AgentState::new(n_locations))
        .collect();
    let init_ts_edges = vec![start_ts; neighbors.len()];
    let mut par_data: Vec<AgentParData> = (0..n_agents)
        .map(|i| {
            let edge_start = neighbor_starts[i];
            let edge_end = neighbor_starts[i + 1];
            AgentParData {
                rng: Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 0)),
                diary: DiaryState {
                    diary_start: diary_starts[i],
                    diary_end: diary_ends[i],
                    diary_idx: 1,
                },
                scratch: Scratch::new(),
                moves: Vec::with_capacity(32),
                neighbor_indices: neighbors[edge_start..edge_end].to_vec(),
                edge_sim: vec![0.0_f64; edge_end - edge_start],
                edge_upd: init_ts_edges[edge_start..edge_end].to_vec(),
            }
        })
        .collect();

    for (i, agent) in agents.iter_mut().enumerate() {
        let loc = if let Some(sl) = starting_locs {
            sl[i].min(n_locations - 1)
        } else {
            pick_starting_loc(
                relevances,
                &mut par_data[i].rng,
                starting_locs_mode_relevance,
            )
        };
        agent.current_location = loc;
        agent.home_location = loc;
        agent.visit(loc);
    }

    let mut out_agents: Vec<i64> = Vec::with_capacity(n_agents);
    let mut out_lats: Vec<f64> = Vec::with_capacity(n_agents);
    let mut out_lngs: Vec<f64> = Vec::with_capacity(n_agents);
    let mut out_arr: Vec<i64> = Vec::with_capacity(n_agents);
    let mut out_dep: Vec<i64> = Vec::with_capacity(n_agents);
    let mut out_dur: Vec<f64> = Vec::with_capacity(n_agents);
    let mut last_output_idx: Vec<usize> = Vec::with_capacity(n_agents);
    for (i, agent) in agents.iter().enumerate() {
        last_output_idx.push(out_agents.len());
        out_agents.push(i as i64 + 1);
        out_lats.push(lats[agent.current_location]);
        out_lngs.push(lngs[agent.current_location]);
        out_arr.push(start_ts);
        out_dep.push(start_ts);
        out_dur.push(0.0);
    }

    let mut commit_buf: Vec<(i64, usize, usize)> = Vec::new();
    let mut window_start = start_ts;
    while window_start < end_ts {
        let window_end = (window_start + indipendency_window_s).min(end_ts);

        par_data.par_iter_mut().enumerate().for_each(|(a, data)| {
            while let Some(ts) = data.diary.current_ts(diary_timestamps) {
                if ts >= window_end {
                    break;
                }
                let loc = choose_location_local(
                    a,
                    &agents,
                    &data.diary,
                    &data.neighbor_indices,
                    &mut data.edge_sim,
                    &mut data.edge_upd,
                    &mut data.rng,
                    rho,
                    gamma,
                    alpha,
                    n_locations,
                    ts,
                    dt_update_mob_sim_s,
                    distances,
                    od_rows.as_ref(),
                    relevances,
                    diary_abs_locs,
                    &mut data.scratch,
                );
                data.moves.push((loc, ts));
                data.diary.advance(diary_timestamps, end_ts);
            }
        });

        commit_buf.clear();
        for (a, data) in par_data.iter_mut().enumerate() {
            for &(loc, ts) in &data.moves {
                commit_buf.push((ts, a, loc));
            }
            data.moves.clear();
        }
        commit_buf.sort_unstable_by_key(|&(ts, a, _)| (ts, a));
        for &(ts, a, loc) in &commit_buf {
            append_trip_record(
                a,
                loc,
                ts,
                lats,
                lngs,
                car_speed_kmh,
                slot_seconds,
                &mut out_agents,
                &mut out_lats,
                &mut out_lngs,
                &mut out_arr,
                &mut out_dep,
                &mut out_dur,
                &mut last_output_idx,
                &mut agents,
            );
        }

        window_start = window_end;
    }

    for &idx in &last_output_idx {
        out_dep[idx] = end_ts.max(out_arr[idx]);
    }

    Ok((out_agents, out_lats, out_lngs, out_arr, out_dep, out_dur))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    latitudes, longitudes, relevances, distances,
    neighbor_starts, neighbors,
    diary_timestamps, diary_abs_locs, diary_starts, diary_ends,
    rho, gamma, alpha,
    start_ts, end_ts, indipendency_window_s, dt_update_mob_sim_s,
    slot_seconds, car_speed_kmh,
    n_agents, master_seed=None, starting_locs=None,
    starting_locs_mode_relevance=false
))]
pub fn trip_sts_epr_simulate_agents<'py>(
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
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let rels = relevances.as_slice()?;
    let dists = distances.as_slice()?;
    let ns_raw = neighbor_starts.as_slice()?;
    let nb_raw = neighbors.as_slice()?;
    let dt_raw = diary_timestamps.as_slice()?;
    let da_raw = diary_abs_locs.as_slice()?;
    let ds_raw = diary_starts.as_slice()?;
    let de_raw = diary_ends.as_slice()?;

    let ns: Vec<usize> = ns_raw.iter().map(|&v| v.max(0) as usize).collect();
    let nb: Vec<usize> = nb_raw.iter().map(|&v| v.max(0) as usize).collect();
    let ds: Vec<usize> = ds_raw.iter().map(|&v| v.max(0) as usize).collect();
    let de: Vec<usize> = de_raw.iter().map(|&v| v.max(0) as usize).collect();

    let sl_buf: Vec<usize>;
    let sl: Option<&[usize]> = match &starting_locs {
        Some(v) => {
            sl_buf = v.as_slice()?.iter().map(|&x| x.max(0) as usize).collect();
            Some(&sl_buf)
        }
        None => None,
    };

    let (out_agents, out_lats, out_lngs, out_arr, out_dep, out_dur) = simulate_trip_sts_epr_impl(
        lats,
        lngs,
        rels,
        dists,
        &ns,
        &nb,
        dt_raw,
        da_raw,
        &ds,
        &de,
        rho,
        gamma,
        alpha,
        start_ts,
        end_ts,
        indipendency_window_s,
        dt_update_mob_sim_s,
        slot_seconds,
        car_speed_kmh,
        n_agents,
        master_seed,
        sl,
        starting_locs_mode_relevance,
    )
    .map_err(PyValueError::new_err)?;

    Ok((
        out_agents.into_pyarray(py),
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_arr.into_pyarray(py),
        out_dep.into_pyarray(py),
        out_dur.into_pyarray(py),
    ))
}
