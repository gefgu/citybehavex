use rand::Rng;

use crate::simulation_core::inputs::ActivityInputs;
use crate::simulation_core::types::Scratch;

pub(crate) const COMMUTE_ACTIVITY_IDX: usize = 16;
pub(crate) const TRAVEL_ACTIVITY_IDX: usize = 17;

fn sample_standard_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Activity ids eligible for `abstract_loc`'s purpose bucket, or `None` when
/// the catalog/CSR inputs are missing or the bucket is empty (callers fall
/// back to a flat 1-hour "unknown" activity in that case).
fn eligible_activities_for_purpose<'a>(
    abstract_loc: i32,
    inputs: &ActivityInputs<'a>,
) -> Option<&'a [usize]> {
    if inputs.act_dur_mu.is_empty() || inputs.purpose_act_starts.len() < 2 {
        return None;
    }
    let purpose = abstract_loc.clamp(0, 6) as usize;
    if purpose + 1 >= inputs.purpose_act_starts.len() {
        return None;
    }
    let act_start = inputs.purpose_act_starts[purpose];
    let act_end = inputs.purpose_act_starts[purpose + 1];
    if act_start >= act_end {
        return None;
    }
    Some(&inputs.purpose_acts[act_start..act_end])
}

/// Preference weight for agent `agent` doing activity `act`, blending its
/// visit-count base rate with profile/activity similarity (precomputed sims
/// take priority over embeddings; falls back to the base rate alone when
/// neither is available).
#[allow(clippy::too_many_arguments)]
fn activity_weight(
    agent: usize,
    location: usize,
    abstract_loc: i32,
    block_id: i32,
    previous_activity: i32,
    act: usize,
    count: u32,
    n_acts: usize,
    has_precomputed: bool,
    has_contextual: bool,
    has_embs: bool,
    prof_slice: &[f64],
    inputs: &ActivityInputs<'_>,
) -> f64 {
    let base = if count > 0 {
        count as f64
    } else {
        inputs.kappa
    };
    let poi_sim = if abstract_loc > 1 && has_contextual {
        let cluster = inputs.cluster_labels[agent];
        let semantic_cluster = inputs
            .location_semantic_cluster_ids
            .get(location)
            .copied()
            .unwrap_or(0);
        let idx = ((cluster * inputs.n_poi_semantic_clusters + semantic_cluster) * n_acts) + act;
        if cluster < inputs.n_clusters
            && semantic_cluster < inputs.n_poi_semantic_clusters
            && idx < inputs.poi_semantic_scores.len()
        {
            Some(inputs.poi_semantic_scores[idx].clamp(0.0, 1.0))
        } else {
            None
        }
    } else {
        None
    };
    let contextual_sim = if poi_sim.is_some() {
        poi_sim
    } else if has_contextual && abstract_loc <= 1 && block_id >= 0 {
        let cluster = inputs.cluster_labels[agent];
        let block = block_id as usize;
        let previous_idx = if previous_activity >= 0 {
            (previous_activity as usize + 1).min(inputs.n_previous.saturating_sub(1))
        } else {
            0
        };
        let base_idx = (((cluster * inputs.n_blocks + block) * inputs.n_previous) * n_acts) + act;
        let hist_idx = (((cluster * inputs.n_blocks + block) * inputs.n_previous + previous_idx)
            * n_acts)
            + act;
        if hist_idx < inputs.contextual_scores.len() && base_idx < inputs.contextual_scores.len() {
            let base_score = inputs.contextual_scores[base_idx].clamp(0.0, 1.0);
            let hist_score = inputs.contextual_scores[hist_idx].clamp(0.0, 1.0);
            Some(base_score + inputs.history_weight * (hist_score - base_score))
        } else {
            None
        }
    } else {
        None
    };
    let sim = if let Some(score) = contextual_sim {
        score
    } else if has_precomputed {
        inputs.profile_act_sims[agent * n_acts + act].clamp(-1.0, 1.0)
    } else if has_embs && act * inputs.emb_dim + inputs.emb_dim <= inputs.act_embs.len() {
        let act_emb = &inputs.act_embs[act * inputs.emb_dim..(act + 1) * inputs.emb_dim];
        prof_slice
            .iter()
            .zip(act_emb.iter())
            .map(|(p, q)| p * q)
            .sum::<f64>()
            .clamp(-1.0, 1.0)
    } else {
        f64::NAN
    };
    if sim.is_nan() {
        base
    } else {
        base * (sim / inputs.temperature).exp()
    }
}

fn poi_activities_for_location<'a>(
    location: usize,
    inputs: &ActivityInputs<'a>,
) -> Option<&'a [usize]> {
    if inputs.location_semantic_cluster_ids.len() <= location || inputs.poi_mask_starts.len() < 2 {
        return None;
    }
    let semantic_cluster = inputs.location_semantic_cluster_ids[location];
    if semantic_cluster + 1 >= inputs.poi_mask_starts.len() {
        return None;
    }
    let start = inputs.poi_mask_starts[semantic_cluster];
    let end = inputs.poi_mask_starts[semantic_cluster + 1];
    if start >= end || end > inputs.poi_mask_activities.len() {
        return None;
    }
    Some(&inputs.poi_mask_activities[start..end])
}

/// Log-normal activity duration in seconds, floored at one minute.
fn sample_duration(chosen: usize, inputs: &ActivityInputs<'_>, rng: &mut impl Rng) -> i64 {
    let mu = if chosen < inputs.act_dur_mu.len() {
        inputs.act_dur_mu[chosen]
    } else {
        0.0
    };
    let sigma = if chosen < inputs.act_dur_sigma.len() {
        inputs.act_dur_sigma[chosen]
    } else {
        0.5
    };
    let z = sample_standard_normal(rng);
    let dur_hours = (mu + sigma * z).exp().max(1.0 / 60.0);
    (dur_hours * 3600.0).round() as i64
}

pub(crate) fn sample_activity_and_duration(
    agent: usize,
    location: usize,
    abstract_loc: i32,
    block_id: i32,
    previous_activity: i32,
    activity_counts: &mut Vec<u32>,
    rng: &mut impl Rng,
    inputs: &ActivityInputs<'_>,
    scratch: &mut Scratch,
) -> (i64, i64) {
    let Some(eligible) = eligible_activities_for_purpose(abstract_loc, inputs) else {
        return (0, 3600);
    };
    let n_acts = inputs.act_dur_mu.len();

    let has_precomputed = !inputs.profile_act_sims.is_empty()
        && inputs.profile_act_sims.len() >= (agent + 1) * n_acts;
    let has_contextual = !inputs.contextual_scores.is_empty()
        && !inputs.cluster_labels.is_empty()
        && inputs.cluster_labels.len() > agent
        && inputs.n_clusters > 0
        && inputs.n_blocks > 0
        && inputs.n_previous > 0
        && inputs.cluster_labels[agent] < inputs.n_clusters
        && inputs.contextual_scores.len()
            >= inputs.n_clusters * inputs.n_blocks * inputs.n_previous * n_acts;
    let has_poi_contextual = !inputs.poi_semantic_scores.is_empty()
        && !inputs.cluster_labels.is_empty()
        && inputs.cluster_labels.len() > agent
        && inputs.n_clusters > 0
        && inputs.n_poi_semantic_clusters > 0
        && inputs.cluster_labels[agent] < inputs.n_clusters
        && inputs.poi_semantic_scores.len()
            >= inputs.n_clusters * inputs.n_poi_semantic_clusters * n_acts;
    let has_context_for_current = if abstract_loc > 1 {
        has_poi_contextual
    } else {
        has_contextual
    };
    let has_embs = !has_precomputed
        && !has_context_for_current
        && inputs.emb_dim > 0
        && !inputs.act_embs.is_empty()
        && !inputs.profile_embs.is_empty()
        && (agent + 1) * inputs.emb_dim <= inputs.profile_embs.len();
    let prof_slice: &[f64] = if has_embs {
        &inputs.profile_embs[agent * inputs.emb_dim..(agent + 1) * inputs.emb_dim]
    } else {
        &[]
    };

    scratch.candidates.clear();
    let source_activities = if abstract_loc > 1 {
        poi_activities_for_location(location, inputs).unwrap_or(eligible)
    } else {
        eligible
    };
    for &a in source_activities {
        if a >= n_acts {
            continue;
        }
        if inputs.materialize_travel && (a == COMMUTE_ACTIVITY_IDX || a == TRAVEL_ACTIVITY_IDX) {
            continue;
        }
        scratch.candidates.push(a);
    }
    if scratch.candidates.is_empty() {
        for &a in eligible {
            if a >= n_acts {
                continue;
            }
            if inputs.materialize_travel && (a == COMMUTE_ACTIVITY_IDX || a == TRAVEL_ACTIVITY_IDX)
            {
                continue;
            }
            scratch.candidates.push(a);
        }
    }

    scratch.act_cdf.clear();
    let mut cumsum = 0.0_f64;
    for &a in &scratch.candidates {
        let count = if a < activity_counts.len() {
            activity_counts[a]
        } else {
            0
        };
        let w = activity_weight(
            agent,
            location,
            abstract_loc,
            block_id,
            previous_activity,
            a,
            count,
            n_acts,
            has_precomputed,
            has_contextual || has_poi_contextual,
            has_embs,
            prof_slice,
            inputs,
        );
        cumsum += w;
        scratch.act_cdf.push(cumsum);
    }
    if scratch.act_cdf.is_empty() || cumsum <= 0.0 {
        let fallback = scratch.candidates.first().copied().unwrap_or(eligible[0]);
        return (fallback as i64, sample_duration(fallback, inputs, rng));
    }

    let threshold = rng.gen_range(0.0..1.0) * cumsum;
    let idx = scratch
        .act_cdf
        .partition_point(|&v| v <= threshold)
        .min(scratch.candidates.len() - 1);
    let chosen = scratch.candidates[idx];

    if chosen >= activity_counts.len() {
        activity_counts.resize(chosen + 1, 0);
    }
    activity_counts[chosen] += 1;

    let dur_secs = sample_duration(chosen, inputs, rng);
    (chosen as i64, dur_secs)
}
