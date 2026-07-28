//! `profiles` chart section: mobility-profile metrics (intermittency,
//! degree-of-return, regularity, diversity, stationarity, entropy) and
//! deterministic Routiner/Regular/Scouter clustering/labeling. Mirrors
//! `legacy Python backend/payload/legacy.py`'s profiles section.

use crate::payload::{
    ComparisonContext, available_filters, empty_chart_payload, prepared_visits_for_filter,
};
use polars::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

const PROFILE_METRICS: &[&str] = &["regularity", "diversity", "stationarity", "entropy"];
const PROFILE_ORDER: &[&str] = &["Scouter", "Regular", "Routiner"];
const MAX_SCATTER_POINTS: usize = 5000;

#[derive(Debug, Clone)]
struct ProfileVisit {
    uid: String,
    start_us: i64,
    end_us: i64,
    purpose: String,
    location_id: String,
}

#[derive(Debug, Clone)]
struct ProfileRow {
    #[allow(dead_code)]
    uid: String,
    intermittency: f64,
    degree_of_return: f64,
    regularity: f64,
    diversity: f64,
    stationarity: f64,
    entropy: f64,
    agent_type: String,
}

fn profile_visits_from_df(df: &DataFrame) -> anyhow::Result<Vec<ProfileVisit>> {
    // `uid` is treated as an opaque, generic key here (matching Python's
    // narwhals-based `compute_profiles`, which never assumes it's numeric) --
    // real observed survey data can use composite string identifiers (e.g.
    // "10_2980"), which a strict Int64 cast would silently null out entirely,
    // as it did before this was a String cast.
    let uid = df.column("uid")?.cast(&DataType::String)?;
    let uid = uid.str()?;
    let start = df
        .column("start_timestamp")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let start = start.datetime()?;
    let end = df
        .column("end_timestamp")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let end = end.datetime()?;
    let purpose = df.column("purpose")?.cast(&DataType::String)?;
    let purpose = purpose.str()?;
    let location = df.column("location_id")?.cast(&DataType::String)?;
    let location = location.str()?;
    let mut rows = Vec::new();
    for i in 0..df.height() {
        // No positive-duration filter here: Python's `compute_profiles` feeds
        // every raw visit row (`df`, no filter) into `regularity`/`diversity`/
        // `entropy`, and `location_token` derivation for intermittency's 5-min
        // expansion. Only `_stationarity` filters `duration_minutes > 0`,
        // scoped to its own dwell/span computation -- see the per-user loop
        // in `compute_profiles_rows` below, not here.
        let (Some(uid), Some(start_us), Some(end_us)) =
            (uid.get(i), start.phys.get(i), end.phys.get(i))
        else {
            continue;
        };
        rows.push(ProfileVisit {
            uid: uid.to_string(),
            start_us,
            end_us,
            purpose: purpose.get(i).unwrap_or("").to_string(),
            location_id: location.get(i).unwrap_or("").to_string(),
        });
    }
    rows.sort_by(|a, b| (&a.uid, a.start_us).cmp(&(&b.uid, b.start_us)));
    Ok(rows)
}

fn distinct_substring_diversity(tokens: &[String]) -> f64 {
    let n = tokens.len();
    if n <= 1 {
        return 0.0;
    }
    let mut seen = HashSet::<Vec<&str>>::new();
    for i in 0..n {
        for j in (i + 1)..=n {
            seen.insert(tokens[i..j].iter().map(String::as_str).collect());
        }
    }
    seen.len() as f64 / ((n * (n + 1) / 2) as f64)
}

fn expand_5min_tokens(visits: &[ProfileVisit]) -> Vec<String> {
    const STEP_US: i64 = 5 * 60 * 1_000_000;
    let mut out = Vec::new();
    let mut seen_ts = HashSet::<i64>::new();
    for visit in visits {
        let mut t = ((visit.start_us + STEP_US - 1) / STEP_US) * STEP_US;
        let end = (visit.end_us / STEP_US) * STEP_US;
        while t <= end {
            if seen_ts.insert(t) {
                out.push(format!("{}_{}", visit.location_id, visit.purpose));
            }
            t += STEP_US;
        }
    }
    if out.is_empty() {
        out.extend(
            visits
                .iter()
                .map(|v| format!("{}_{}", v.location_id, v.purpose)),
        );
    }
    out
}

fn intermittency_and_return(tokens: &[String]) -> Option<(f64, f64)> {
    if tokens.is_empty() {
        return None;
    }
    let mut counts = FxHashMap::<&str, usize>::default();
    for token in tokens {
        *counts.entry(token.as_str()).or_insert(0) += 1;
    }
    let mean_frequency = tokens.len() as f64 / counts.len().max(1) as f64;
    let known_threshold = mean_frequency * 0.8;
    let cold_known: HashSet<&str> = counts
        .iter()
        .filter_map(|(&token, &count)| (count as f64 >= known_threshold).then_some(token))
        .collect();
    let mut first_seen = HashSet::<&str>::new();
    let mut states = Vec::<bool>::new();
    for token in tokens {
        let key = token.as_str();
        let known = first_seen.contains(key) || cold_known.contains(key);
        states.push(known);
        first_seen.insert(key);
    }
    let mut return_blocks = Vec::<usize>::new();
    let mut exploration_blocks = Vec::<usize>::new();
    let mut current = states[0];
    let mut len = 0usize;
    for state in states {
        if state == current {
            len += 1;
        } else {
            if current {
                return_blocks.push(len);
            } else {
                exploration_blocks.push(len);
            }
            current = state;
            len = 1;
        }
    }
    if current {
        return_blocks.push(len);
    } else {
        exploration_blocks.push(len);
    }
    let mean = |xs: &[usize]| {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().sum::<usize>() as f64 / xs.len() as f64
        }
    };
    let mean_return = mean(&return_blocks);
    let mean_exploration = mean(&exploration_blocks);
    Some((
        mean_return + mean_exploration,
        mean_return.atan2(mean_exploration),
    ))
}

fn compute_profiles_rows(visits_df: &DataFrame) -> anyhow::Result<Vec<ProfileRow>> {
    let visits = profile_visits_from_df(visits_df)?;
    let mut by_uid = BTreeMap::<String, Vec<ProfileVisit>>::new();
    for visit in visits {
        by_uid.entry(visit.uid.clone()).or_default().push(visit);
    }
    let mut rows = Vec::new();
    for (uid, rows_for_user) in by_uid {
        let tokens: Vec<String> = rows_for_user
            .iter()
            .map(|v| format!("{}_{}", v.location_id, v.purpose))
            .collect();
        let Some((intermittency, degree_of_return)) =
            intermittency_and_return(&expand_5min_tokens(&rows_for_user))
        else {
            continue;
        };
        if !intermittency.is_finite() || !degree_of_return.is_finite() {
            continue;
        }
        let mut distinct_pairs = HashSet::<(&str, &str)>::new();
        // Mirrors `_stationarity`'s own `duration_minutes > 0` filter --
        // scoped to just this dwell/span computation, not a row exclusion
        // applied to the whole user (regularity/diversity/entropy see every
        // raw visit regardless of duration).
        let mut dwell = 0.0;
        let mut min_start = i64::MAX;
        let mut max_end = i64::MIN;
        for visit in &rows_for_user {
            distinct_pairs.insert((visit.location_id.as_str(), visit.purpose.as_str()));
            if visit.end_us > visit.start_us {
                dwell += (visit.end_us - visit.start_us) as f64 / 60_000_000.0;
                min_start = min_start.min(visit.start_us);
                max_end = max_end.max(visit.end_us);
            }
        }
        let total = rows_for_user.len().max(1) as f64;
        let regularity = 1.0 - distinct_pairs.len() as f64 / total;
        let diversity = distinct_substring_diversity(&tokens);
        let entropy = fastmob_core::measures::individual::entropy::trajectory_entropy_batch(
            tokens,
            vec![(0, rows_for_user.len())],
            true,
        )
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .next()
        .unwrap_or(0.0);
        let span = if max_end > min_start {
            (max_end - min_start) as f64 / 60_000_000.0
        } else {
            0.0
        };
        let stationarity = if span > 0.0 { dwell / span } else { 0.0 };
        rows.push(ProfileRow {
            uid,
            intermittency,
            degree_of_return,
            regularity,
            diversity,
            stationarity,
            entropy,
            agent_type: String::new(),
        });
    }
    if rows.len() < 3 {
        anyhow::bail!(
            "need at least 3 users with finite profiling metrics, got {}",
            rows.len()
        );
    }
    label_profile_clusters(&mut rows);
    Ok(rows)
}

fn label_profile_clusters(rows: &mut [ProfileRow]) {
    let n = rows.len();
    let mean_i = rows.iter().map(|r| r.intermittency).sum::<f64>() / n as f64;
    let mean_d = rows.iter().map(|r| r.degree_of_return).sum::<f64>() / n as f64;
    let std_i = (rows
        .iter()
        .map(|r| (r.intermittency - mean_i).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt()
        .max(1e-12);
    let std_d = (rows
        .iter()
        .map(|r| (r.degree_of_return - mean_d).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt()
        .max(1e-12);
    let points: Vec<[f64; 2]> = rows
        .iter()
        .map(|r| {
            [
                (r.intermittency - mean_i) / std_i,
                (r.degree_of_return - mean_d) / std_d,
            ]
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        rows[a]
            .degree_of_return
            .total_cmp(&rows[b].degree_of_return)
    });
    let mut centers = [
        points[order[n / 6]],
        points[order[n / 2]],
        points[order[(5 * n / 6).min(n - 1)]],
    ];
    let mut assignment = vec![0usize; n];
    for _ in 0..100 {
        let mut changed = false;
        for (idx, point) in points.iter().enumerate() {
            let best = centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (point[0] - a[0]).powi(2) + (point[1] - a[1]).powi(2);
                    let db = (point[0] - b[0]).powi(2) + (point[1] - b[1]).powi(2);
                    da.total_cmp(&db)
                })
                .map(|(cluster, _)| cluster)
                .unwrap_or(0);
            if assignment[idx] != best {
                assignment[idx] = best;
                changed = true;
            }
        }
        let mut sums = [[0.0; 2]; 3];
        let mut counts = [0usize; 3];
        for (cluster, point) in assignment.iter().zip(points.iter()) {
            sums[*cluster][0] += point[0];
            sums[*cluster][1] += point[1];
            counts[*cluster] += 1;
        }
        for cluster in 0..3 {
            if counts[cluster] > 0 {
                centers[cluster] = [
                    sums[cluster][0] / counts[cluster] as f64,
                    sums[cluster][1] / counts[cluster] as f64,
                ];
            }
        }
        if !changed {
            break;
        }
    }
    let mut cluster_mean_return = [(0usize, 0.0f64); 3];
    for cluster in 0..3 {
        let mut count = 0usize;
        let mut sum = 0.0;
        for (idx, row) in rows.iter().enumerate() {
            if assignment[idx] == cluster {
                count += 1;
                sum += row.degree_of_return;
            }
        }
        cluster_mean_return[cluster] = (
            cluster,
            if count > 0 {
                sum / count as f64
            } else {
                f64::NEG_INFINITY
            },
        );
    }
    cluster_mean_return.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut names = ["Scouter"; 3];
    for (rank, (cluster, _)) in cluster_mean_return.iter().enumerate() {
        names[*cluster] = ["Routiner", "Regular", "Scouter"][rank];
    }
    for (idx, row) in rows.iter_mut().enumerate() {
        row.agent_type = names[assignment[idx]].to_string();
    }
}

fn metric_value(row: &ProfileRow, metric: &str) -> f64 {
    match metric {
        "regularity" => row.regularity,
        "diversity" => row.diversity,
        "stationarity" => row.stationarity,
        "entropy" => row.entropy,
        _ => 0.0,
    }
}

fn box_stats(values: &mut [f64]) -> Value {
    values.sort_by(|a, b| a.total_cmp(b));
    let values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return Value::Null;
    }
    let q = |p: f64| {
        let pos = p * (values.len().saturating_sub(1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            values[lo]
        } else {
            values[lo] * (hi as f64 - pos) + values[hi] * (pos - lo as f64)
        }
    };
    json!([
        values[0],
        q(0.25),
        q(0.5),
        q(0.75),
        values[values.len() - 1]
    ])
}

fn profile_scatter(rows: &[ProfileRow], name: &str) -> Value {
    let step = (rows.len() / MAX_SCATTER_POINTS).max(1);
    let points: Vec<Value> = rows
        .iter()
        .step_by(step)
        .take(MAX_SCATTER_POINTS)
        .map(|row| {
            json!({
                "x": row.degree_of_return,
                "y": row.intermittency,
                "profile": row.agent_type,
            })
        })
        .collect();
    json!({"name": name, "points": points})
}

fn build_profiles_block(
    observed_label: &str,
    observed: &[ProfileRow],
    synthetic: &[ProfileRow],
) -> Value {
    let mut box_obj = serde_json::Map::new();
    for metric in PROFILE_METRICS {
        let mut metric_obj = serde_json::Map::new();
        for (name, rows) in [("synthetic", synthetic), (observed_label, observed)] {
            let mut profile_obj = serde_json::Map::new();
            for profile in PROFILE_ORDER {
                let mut values: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.agent_type == *profile)
                    .map(|r| metric_value(r, metric))
                    .collect();
                profile_obj.insert((*profile).to_string(), box_stats(&mut values));
            }
            metric_obj.insert(name.to_string(), Value::Object(profile_obj));
        }
        box_obj.insert((*metric).to_string(), Value::Object(metric_obj));
    }
    json!({
        "scatter": [
            profile_scatter(synthetic, "synthetic"),
            profile_scatter(observed, observed_label),
        ],
        "profile_order": PROFILE_ORDER,
        "metrics": PROFILE_METRICS,
        "datasets": ["synthetic", observed_label],
        "box": box_obj,
    })
}

pub fn profiles_section_payload(ctx: &ComparisonContext) -> anyhow::Result<Value> {
    let mut payload = empty_chart_payload(ctx, Vec::new());
    let all = available_filters(ctx)
        .into_iter()
        .find(|f| f.key == "all")
        .ok_or_else(|| anyhow::anyhow!("missing all filter"))?;
    let visits = prepared_visits_for_filter(ctx, &all)?;
    let (Some(synthetic), Some(observed)) = (visits.synthetic.as_ref(), visits.observed.as_ref())
    else {
        if !visits.warnings.is_empty() {
            payload["warnings"] = json!(visits.warnings);
        }
        return Ok(payload);
    };
    let observed_profiles = compute_profiles_rows(observed)?;
    let synthetic_profiles = compute_profiles_rows(synthetic)?;
    payload["profiles"] =
        build_profiles_block(&ctx.observed_label, &observed_profiles, &synthetic_profiles);
    if !visits.warnings.is_empty() {
        payload["warnings"] = json!(visits.warnings);
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_diversity_counts_distinct_substrings() {
        let tokens = vec![
            "home".to_string(),
            "work".to_string(),
            "home".to_string(),
            "gym".to_string(),
        ];
        assert!((distinct_substring_diversity(&tokens) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn intermittency_and_degree_of_return_are_finite_for_revisits() {
        let tokens = vec![
            "home".to_string(),
            "work".to_string(),
            "home".to_string(),
            "shop".to_string(),
            "home".to_string(),
        ];
        let (intermittency, degree) = intermittency_and_return(&tokens).unwrap();
        assert!(intermittency.is_finite() && intermittency > 0.0);
        assert!(degree.is_finite() && degree >= 0.0);
    }
}
