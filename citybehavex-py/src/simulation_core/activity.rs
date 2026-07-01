use rand::Rng;

use crate::simulation_core::inputs::ActivityInputs;
use crate::simulation_core::types::Scratch;

fn sample_standard_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub(crate) fn sample_activity_and_duration(
    agent: usize,
    abstract_loc: i32,
    activity_counts: &mut Vec<u32>,
    rng: &mut impl Rng,
    inputs: &ActivityInputs<'_>,
    scratch: &mut Scratch,
) -> (i64, i64) {
    let n_acts = inputs.act_dur_mu.len();
    if n_acts == 0 || inputs.purpose_act_starts.len() < 2 {
        return (0, 3600);
    }
    let purpose = abstract_loc.clamp(0, 6) as usize;
    if purpose + 1 >= inputs.purpose_act_starts.len() {
        return (0, 3600);
    }
    let act_start = inputs.purpose_act_starts[purpose];
    let act_end = inputs.purpose_act_starts[purpose + 1];
    if act_start >= act_end {
        return (0, 3600);
    }
    let eligible = &inputs.purpose_acts[act_start..act_end];

    let has_precomputed = !inputs.profile_act_sims.is_empty()
        && inputs.profile_act_sims.len() >= (agent + 1) * n_acts;

    let has_embs = !has_precomputed
        && inputs.emb_dim > 0
        && !inputs.act_embs.is_empty()
        && !inputs.profile_embs.is_empty()
        && (agent + 1) * inputs.emb_dim <= inputs.profile_embs.len();
    let prof_slice = if has_embs {
        &inputs.profile_embs[agent * inputs.emb_dim..(agent + 1) * inputs.emb_dim]
    } else {
        &[]
    };

    scratch.act_cdf.clear();
    let mut cumsum = 0.0_f64;
    for &a in eligible {
        let count = if a < activity_counts.len() {
            activity_counts[a]
        } else {
            0
        };
        let base = if count > 0 { count as f64 } else { inputs.kappa };
        let sim = if has_precomputed {
            inputs.profile_act_sims[agent * n_acts + a].clamp(-1.0, 1.0)
        } else if has_embs && a * inputs.emb_dim + inputs.emb_dim <= inputs.act_embs.len() {
            let act_emb = &inputs.act_embs[a * inputs.emb_dim..(a + 1) * inputs.emb_dim];
            prof_slice
                .iter()
                .zip(act_emb.iter())
                .map(|(p, q)| p * q)
                .sum::<f64>()
                .clamp(-1.0, 1.0)
        } else {
            f64::NAN
        };
        let w = if sim.is_nan() {
            base
        } else {
            base * (sim / inputs.temperature).exp()
        };
        cumsum += w;
        scratch.act_cdf.push(cumsum);
    }

    let threshold = rng.gen_range(0.0..1.0) * cumsum;
    let idx = scratch
        .act_cdf
        .partition_point(|&v| v <= threshold)
        .min(eligible.len() - 1);
    let chosen = eligible[idx];

    if chosen >= activity_counts.len() {
        activity_counts.resize(chosen + 1, 0);
    }
    activity_counts[chosen] += 1;

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
    let dur_secs = (dur_hours * 3600.0).round() as i64;

    (chosen as i64, dur_secs)
}
