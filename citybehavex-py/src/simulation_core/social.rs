use rand::Rng;
use skmob2_core::models::od::CachedGravityOdRows;
use skmob2_core::models::shared::cdf_choice;
use std::collections::HashMap;

use crate::simulation_core::inputs::{LocationInputs, SimulationParams};
use crate::simulation_core::types::{
    AgentState, DiaryState, Encounter, Scratch, SocialMode, GRAVITY_REJECTION_ATTEMPTS, WORK_CODE,
};

fn count_at(counts: &HashMap<usize, u32>, loc: usize) -> u32 {
    counts.get(&loc).copied().unwrap_or(0)
}

fn cosine_similarity_sparse(
    a_locs: &[usize],
    a_counts: &HashMap<usize, u32>,
    norm_a_sq: f64,
    b_counts: &HashMap<usize, u32>,
    norm_b_sq: f64,
) -> f64 {
    if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
        return 0.0;
    }
    let dot: f64 = a_locs
        .iter()
        .filter_map(|&l| {
            b_counts
                .get(&l)
                .map(|&b| (a_counts[&l] as f64) * (b as f64))
        })
        .sum();
    dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt())
}

fn populate_scratchpad(scratch: &mut Scratch, items: impl Iterator<Item = (usize, f64)>) {
    scratch.candidates.clear();
    scratch.cdf.clear();
    let mut cumsum = 0.0_f64;

    for (loc, weight) in items {
        if weight > 0.0 {
            scratch.candidates.push(loc);
            cumsum += weight;
            scratch.cdf.push(cumsum);
        }
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
    populate_scratchpad(
        scratch,
        a.visited_locs
            .iter()
            .map(|&loc| (loc, count_at(&a.visit_counts, loc) as f64)),
    );
    if scratch.candidates.is_empty() {
        None
    } else {
        Some(cdf_sample(rng, &scratch.candidates, &scratch.cdf))
    }
}

fn social_exploration(
    agent: usize,
    agents: &[AgentState],
    locations: &LocationInputs<'_>,
    od_rows: Option<&CachedGravityOdRows>,
    n_locations: usize,
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
                    if !visit_counts.contains_key(&loc) && loc != src && loc != home {
                        return Some(loc);
                    }
                }
            }

            // Rejection sampling exhausted -- fall back to a full scan of
            // unvisited tiles. Deliberately not cached: for a real city
            // (tens/hundreds of thousands of locations) this candidate list
            // is close to the entire location space for any agent that
            // hasn't visited most of it, so a per-agent cache of it costs
            // O(n_locations) memory per agent regardless of how sparse the
            // agent's actual visit history is -- multiplied across 100k+
            // agents this exhausted available RAM long before the memory
            // used by any output buffer. Recomputing this scan is pure CPU
            // cost with no correctness or memory downside.
            populate_scratchpad(
                scratch,
                row.iter()
                    .enumerate()
                    .scan(0.0_f64, |prev, (j, &value)| {
                        let weight = value - *prev;
                        *prev = value;
                        Some((j, weight))
                    })
                    .filter(|&(j, weight)| {
                        !visit_counts.contains_key(&j)
                            && j != src
                            && j != home
                            && weight.is_finite()
                    }),
            );
            if scratch.candidates.is_empty() {
                return None;
            }
            return Some(cdf_sample(rng, &scratch.candidates, &scratch.cdf));
        }
    }

    let src_rel = locations.relevances[src];
    scratch.candidates.clear();
    scratch.cdf.clear();
    let mut all_zero = true;
    let mut cumsum = 0.0_f64;
    for j in 0..n_locations {
        if visit_counts.contains_key(&j) || j == src || j == home {
            continue;
        }
        let d = locations.distances[src * n_locations + j].max(0.001);
        let score = (1.0 / (d * d)) * locations.relevances[j] * src_rel;
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

pub(crate) fn update_edge_sim(
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
                agents[agent].norm_sq,
                &agents[nb].visit_counts,
                agents[nb].norm_sq,
            );
            edge_upd[i] = current_ts + dt_update_s;
        }
    }
}

struct SocialActionContext<'a, R: Rng> {
    agent: usize,
    agents: &'a [AgentState],
    neighbor_indices: &'a [usize],
    edge_sim: &'a [f64],
    mode: SocialMode,
    rng: &'a mut R,
    scratch: &'a mut Scratch,
}

fn make_social_action_local<R: Rng>(ctx: SocialActionContext<'_, R>) -> Option<(usize, usize)> {
    if ctx.neighbor_indices.is_empty() {
        return None;
    }

    let contact_idx = if ctx.edge_sim.iter().all(|&s| s == 0.0) {
        ctx.rng.gen_range(0..ctx.edge_sim.len())
    } else {
        let total: f64 = ctx.edge_sim.iter().sum();
        let threshold = ctx.rng.gen_range(0.0_f64..1.0) * total;
        let mut cumsum = 0.0_f64;
        let mut chosen = ctx.edge_sim.len() - 1;
        for (i, &w) in ctx.edge_sim.iter().enumerate() {
            cumsum += w;
            if cumsum > threshold {
                chosen = i;
                break;
            }
        }
        chosen
    };
    let contact = ctx.neighbor_indices[contact_idx];

    let agent_counts = &ctx.agents[ctx.agent].visit_counts;
    let contact_counts = &ctx.agents[contact].visit_counts;
    let contact_locs = &ctx.agents[contact].visited_locs;
    let agent_locs = &ctx.agents[ctx.agent].visited_locs;

    match ctx.mode {
        SocialMode::Exploration => populate_scratchpad(
            ctx.scratch,
            contact_locs
                .iter()
                .filter(|&&loc| !agent_counts.contains_key(&loc))
                .map(|&loc| (loc, count_at(contact_counts, loc) as f64)),
        ),
        SocialMode::Return => populate_scratchpad(
            ctx.scratch,
            agent_locs
                .iter()
                .map(|&loc| (loc, count_at(contact_counts, loc) as f64)),
        ),
    }

    if ctx.scratch.candidates.is_empty() {
        None
    } else {
        let loc = cdf_sample(ctx.rng, &ctx.scratch.candidates, &ctx.scratch.cdf);
        Some((loc, contact))
    }
}

pub(crate) struct LocationChoiceContext<'a, R: Rng> {
    pub(crate) agent: usize,
    pub(crate) agents: &'a [AgentState],
    pub(crate) diary: &'a DiaryState,
    pub(crate) neighbor_indices: &'a [usize],
    pub(crate) edge_sim: &'a [f64],
    pub(crate) rng: &'a mut R,
    pub(crate) params: &'a SimulationParams,
    pub(crate) n_locations: usize,
    pub(crate) current_ts: i64,
    pub(crate) locations: &'a LocationInputs<'a>,
    pub(crate) od_rows: Option<&'a CachedGravityOdRows<'a>>,
    pub(crate) diary_abs_locs: &'a [i32],
    pub(crate) scratch: &'a mut Scratch,
    pub(crate) encounters: &'a mut Vec<Encounter>,
}

fn social_action_for_choice<R: Rng>(
    ctx: &mut LocationChoiceContext<'_, R>,
    mode: SocialMode,
) -> Option<usize> {
    make_social_action_local(SocialActionContext {
        agent: ctx.agent,
        agents: ctx.agents,
        neighbor_indices: ctx.neighbor_indices,
        edge_sim: ctx.edge_sim,
        mode,
        rng: ctx.rng,
        scratch: ctx.scratch,
    })
    .map(|(loc, contact)| {
        ctx.encounters.push(Encounter {
            agent: ctx.agent as u32,
            contact: contact as u32,
            tile: loc as u32,
            ts: ctx.current_ts as i32,
        });
        loc
    })
}

fn exploration_for_choice<R: Rng>(ctx: &mut LocationChoiceContext<'_, R>) -> Option<usize> {
    social_exploration(
        ctx.agent,
        ctx.agents,
        ctx.locations,
        ctx.od_rows,
        ctx.n_locations,
        ctx.rng,
        ctx.scratch,
    )
}

fn return_for_choice<R: Rng>(ctx: &mut LocationChoiceContext<'_, R>) -> Option<usize> {
    make_individual_return(ctx.agent, ctx.agents, ctx.rng, ctx.scratch)
}

pub(crate) fn choose_location_local<R: Rng>(mut ctx: LocationChoiceContext<'_, R>) -> usize {
    let abstract_location = ctx.diary.current_abstract_location(ctx.diary_abs_locs);
    if abstract_location == 0 {
        return ctx.agents[ctx.agent].home_location;
    }
    if abstract_location == WORK_CODE {
        return ctx.agents[ctx.agent].work_location;
    }

    let s = ctx.agents[ctx.agent].s.max(1.0);
    let p_explore = ctx.params.rho * s.powf(-ctx.params.gamma);
    let explore = ctx.rng.gen_range(0.0_f64..1.0) < p_explore;
    let social = ctx.rng.gen_range(0.0_f64..1.0) < ctx.params.alpha;

    let location = match (explore, social) {
        (true, true) => social_action_for_choice(&mut ctx, SocialMode::Exploration)
            .or_else(|| exploration_for_choice(&mut ctx))
            .or_else(|| return_for_choice(&mut ctx)),
        (true, false) => exploration_for_choice(&mut ctx)
            .or_else(|| social_action_for_choice(&mut ctx, SocialMode::Exploration))
            .or_else(|| return_for_choice(&mut ctx)),
        (false, true) => social_action_for_choice(&mut ctx, SocialMode::Return)
            .or_else(|| return_for_choice(&mut ctx))
            .or_else(|| exploration_for_choice(&mut ctx)),
        (false, false) => return_for_choice(&mut ctx)
            .or_else(|| social_action_for_choice(&mut ctx, SocialMode::Return))
            .or_else(|| exploration_for_choice(&mut ctx)),
    };

    location.unwrap_or(ctx.agents[ctx.agent].current_location)
}

pub(crate) fn pick_starting_loc(
    relevances: &[f64],
    rng: &mut impl Rng,
    mode_relevance: bool,
) -> usize {
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
