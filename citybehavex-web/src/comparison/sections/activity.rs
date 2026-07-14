//! `activity` chart section: purpose-share bars, activity-transition-matrix
//! difference/raw blocks, and daily activity profile difference/raw blocks.
//! Mirrors `web/backend/app/payload/legacy.py`'s activity section.

use crate::columns::{ACTIVITY_CANDIDATES, END_TS_CANDIDATES, LOCATION_CANDIDATES};
use crate::comparison::activity::{activity_transition_matrix, daily_activity_distribution};
use crate::comparison::filters::filter_visits;
use crate::comparison::trajectory::load_trajectory;
use crate::comparison::visits::prepare_activity_visits;
use crate::payload::{ComparisonContext, choose_regular_filter, detected_col, empty_chart_payload};
use polars::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;

pub(crate) fn string_column(df: &DataFrame, name: &str) -> anyhow::Result<Vec<String>> {
    Ok(df
        .column(name)?
        .as_materialized_series()
        .cast(&DataType::String)?
        .str()?
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect())
}

fn purpose_distribution(visits: &DataFrame) -> anyhow::Result<(Vec<String>, HashMap<String, f64>)> {
    let purposes = string_column(visits, "purpose")?;
    let mut order = Vec::<String>::new();
    let mut counts = HashMap::<String, i64>::new();
    for purpose in purposes {
        if !counts.contains_key(&purpose) {
            order.push(purpose.clone());
        }
        *counts.entry(purpose).or_insert(0) += 1;
    }
    let total = counts.values().sum::<i64>().max(1) as f64;
    let dist = counts
        .into_iter()
        .map(|(key, count)| {
            (
                key,
                ((count as f64 / total * 100.0) * 100.0).round() / 100.0,
            )
        })
        .collect();
    Ok((order, dist))
}

fn round3(v: f64) -> f64 {
    if v.is_finite() {
        (v * 1000.0).round() / 1000.0
    } else {
        0.0
    }
}

/// Mirrors `legacy.py`'s `"limit": max(float(np.abs(matrix[...]).max()), 1.0)`
/// -- Python's actual runtime computation takes this max from the
/// already-`.round(3)`-ed matrix, not the raw one, so round here to match
/// bit-for-bit (confirmed on real `gparis_simulation` data: without this,
/// `limit` differed from Python in the 4th decimal place, e.g.
/// `50.4454680344345` vs `50.445`).
fn matrix_limit(matrix: &[Vec<f64>]) -> f64 {
    let raw = matrix
        .iter()
        .flatten()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    round3(raw).max(1.0)
}

pub(crate) fn align_square(categories: &[String], matrix: &[Vec<f64>], target: &[String]) -> Vec<Vec<f64>> {
    let index: HashMap<&str, usize> = target
        .iter()
        .enumerate()
        .map(|(i, cat)| (cat.as_str(), i))
        .collect();
    let mut out = vec![vec![0.0; target.len()]; target.len()];
    for (src_i, cat_i) in categories.iter().enumerate() {
        let Some(&dst_i) = index.get(cat_i.as_str()) else {
            continue;
        };
        for (src_j, cat_j) in categories.iter().enumerate() {
            let Some(&dst_j) = index.get(cat_j.as_str()) else {
                continue;
            };
            out[dst_i][dst_j] = matrix
                .get(src_i)
                .and_then(|row| row.get(src_j))
                .copied()
                .unwrap_or(0.0);
        }
    }
    out
}

pub(crate) fn align_daily(categories: &[String], matrix: &[Vec<f64>], target: &[String]) -> Vec<Vec<f64>> {
    let n_bins = matrix.first().map_or(0, Vec::len);
    let index: HashMap<&str, usize> = target
        .iter()
        .enumerate()
        .map(|(i, cat)| (cat.as_str(), i))
        .collect();
    let mut out = vec![vec![0.0; n_bins]; target.len()];
    for (src_i, cat) in categories.iter().enumerate() {
        let Some(&dst_i) = index.get(cat.as_str()) else {
            continue;
        };
        if let Some(row) = matrix.get(src_i) {
            for (bin, value) in row.iter().enumerate() {
                out[dst_i][bin] = if value.is_finite() { *value } else { 0.0 };
            }
        }
    }
    out
}

fn subtract_matrices(lhs: Vec<Vec<f64>>, rhs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    lhs.into_iter()
        .zip(rhs)
        .map(|(lrow, rrow)| {
            lrow.into_iter()
                .zip(rrow)
                .map(|(l, r)| round3(l - r))
                .collect()
        })
        .collect()
}

fn round_matrix(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    matrix
        .into_iter()
        .map(|row| row.into_iter().map(round3).collect())
        .collect()
}

fn datetime_minutes(visits: &DataFrame, name: &str) -> anyhow::Result<Vec<i64>> {
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    const MICROS_PER_MINUTE: i64 = 60_000_000;
    let series = visits
        .column(name)?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let ca = series.datetime()?.clone();
    Ok((0..visits.height())
        .map(|i| ca.phys.get(i).unwrap_or(0).rem_euclid(MICROS_PER_DAY) / MICROS_PER_MINUTE)
        .collect())
}

pub(crate) type DailyTuple = (Vec<String>, Vec<Vec<f64>>);

/// Mirrors fastmob's `daily_activity_distribution(visits)` -- both this
/// chart's `daily_activity_difference` display matrix and the JSD side
/// effect in `sections::metrics` reuse the exact same tuple in
/// `legacy.py::_activity_group`/`_build_activity_block`, computed with no
/// explicit `bin_size_minutes` (the library default of **10 minutes**, i.e.
/// 144 bins -- not the 60-minute/24-bin granularity this port used before
/// being caught by a real-data parity run).
pub(crate) fn daily_tuple(visits: &DataFrame) -> anyhow::Result<DailyTuple> {
    let purpose = string_column(visits, "purpose")?;
    let start_minutes = datetime_minutes(visits, "start_timestamp")?;
    let end_minutes = datetime_minutes(visits, "end_timestamp")?;
    let valid_rows = vec![true; visits.height()];
    daily_activity_distribution(&purpose, &start_minutes, &end_minutes, &valid_rows, 10)
}

pub(crate) fn ordered_union(left: &[String], right: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for cat in left.iter().chain(right.iter()) {
        if !out.contains(cat) {
            out.push(cat.clone());
        }
    }
    out
}

fn build_activity_block(
    ctx: &ComparisonContext,
    synthetic_visits: &DataFrame,
    observed_visits: Option<&DataFrame>,
) -> anyhow::Result<Value> {
    let (syn_order, syn_dist) = purpose_distribution(synthetic_visits)?;
    let (obs_order, obs_dist) = match observed_visits {
        Some(obs) => purpose_distribution(obs)?,
        None => (Vec::new(), HashMap::new()),
    };
    let purpose_categories = ordered_union(&syn_order, &obs_order);
    let mut purpose_series = vec![json!({
        "name": "synthetic",
        "role": "synthetic",
        "values": purpose_categories.iter().map(|c| syn_dist.get(c).copied().unwrap_or(0.0)).collect::<Vec<_>>(),
    })];
    if observed_visits.is_some() {
        purpose_series.push(json!({
            "name": ctx.observed_label,
            "role": "observed",
            "values": purpose_categories.iter().map(|c| obs_dist.get(c).copied().unwrap_or(0.0)).collect::<Vec<_>>(),
        }));
    }

    let (syn_trans_cats, syn_trans_mat) =
        activity_transition_matrix(synthetic_visits, "uid", "purpose")?;
    let (obs_trans_cats, obs_trans_mat) = match observed_visits {
        Some(obs) => activity_transition_matrix(obs, "uid", "purpose")?,
        None => (Vec::new(), Vec::new()),
    };
    let trans_cats = ordered_union(&syn_trans_cats, &obs_trans_cats);
    let syn_aligned = align_square(&syn_trans_cats, &syn_trans_mat, &trans_cats);
    let (transition_matrix, transition_mode, transition_labels) = if observed_visits.is_some() {
        (
            subtract_matrices(
                align_square(&obs_trans_cats, &obs_trans_mat, &trans_cats),
                syn_aligned,
            ),
            "difference",
            vec!["synthetic".to_string(), ctx.observed_label.clone()],
        )
    } else {
        (
            round_matrix(syn_aligned),
            "raw",
            vec!["synthetic".to_string()],
        )
    };

    let (syn_daily_cats, syn_daily_mat) = daily_tuple(synthetic_visits)?;
    let observed_daily = match observed_visits {
        Some(obs) => Some(daily_tuple(obs)?),
        None => None,
    };
    let daily = if let Some((obs_daily_cats, obs_daily_mat)) = observed_daily {
        if syn_daily_mat.first().map_or(0, Vec::len) == obs_daily_mat.first().map_or(0, Vec::len) {
            let cats = ordered_union(&syn_daily_cats, &obs_daily_cats);
            let matrix = subtract_matrices(
                align_daily(&obs_daily_cats, &obs_daily_mat, &cats),
                align_daily(&syn_daily_cats, &syn_daily_mat, &cats),
            );
            Some(json!({
                "categories": cats,
                "n_bins": syn_daily_mat.first().map_or(0, Vec::len),
                "labels": ["synthetic", ctx.observed_label.as_str()],
                "matrix_mode": "difference",
                "matrix": matrix,
                "limit": matrix_limit(&matrix),
            }))
        } else {
            None
        }
    } else {
        let cats = syn_daily_cats;
        let matrix = round_matrix(align_daily(&cats, &syn_daily_mat, &cats));
        Some(json!({
            "categories": cats,
            "n_bins": syn_daily_mat.first().map_or(0, Vec::len),
            "labels": ["synthetic"],
            "matrix_mode": "raw",
            "matrix": matrix,
            "limit": matrix_limit(&matrix),
        }))
    };

    Ok(json!({
        "purpose": {
            "categories": purpose_categories,
            "series": purpose_series,
        },
        "transition_difference": {
            "categories": trans_cats,
            "labels": transition_labels,
            "matrix_mode": transition_mode,
            "matrix": transition_matrix,
            "limit": matrix_limit(&transition_matrix),
        },
        "daily_activity_difference": daily,
    }))
}

pub fn activity_section_payload(ctx: &ComparisonContext, filter_key: &str) -> anyhow::Result<Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let mut warnings = Vec::<String>::new();

    let synthetic_traj = load_trajectory(&ctx.synthetic_path)?;
    let synth_activity_col = detected_col(&synthetic_traj.df, ACTIVITY_CANDIDATES);
    let synth_location_col = detected_col(&synthetic_traj.df, LOCATION_CANDIDATES);
    let Some(synth_result) = prepare_activity_visits(
        &synthetic_traj.df,
        "synthetic",
        Some(&synthetic_traj.uid_col),
        Some(&synthetic_traj.datetime_col),
        synth_activity_col.as_deref(),
        synth_location_col.as_deref(),
        Some(&synthetic_traj.lat_col),
        Some(&synthetic_traj.lng_col),
        ctx.evaluation_h3_resolution,
        None,
    )?
    else {
        return Ok(payload);
    };
    if let Some(warning) = synth_result.warning {
        warnings.push(warning);
    }

    let synthetic_visits = synth_result
        .visits
        .lazy()
        .sort(["uid", "start_timestamp"], SortMultipleOptions::default())
        .collect()?;
    let syn_filtered = filter_visits(Some(&synthetic_visits), &filter)?.unwrap();
    if syn_filtered.height() == 0 {
        warnings.push(format!(
            "{} activity filter has no synthetic visits",
            filter.label
        ));
        payload["warnings"] = json!(warnings);
        return Ok(payload);
    }

    let observed_filtered = if let Some(path) = &ctx.observed_path {
        let observed_traj = load_trajectory(path)?;
        let obs_activity_col = detected_col(&observed_traj.df, ACTIVITY_CANDIDATES);
        let obs_location_col = detected_col(&observed_traj.df, LOCATION_CANDIDATES);
        let obs_end_col = detected_col(&observed_traj.df, END_TS_CANDIDATES);
        match prepare_activity_visits(
            &observed_traj.df,
            &ctx.observed_label,
            Some(&observed_traj.uid_col),
            Some(&observed_traj.datetime_col),
            obs_activity_col.as_deref(),
            obs_location_col.as_deref(),
            Some(&observed_traj.lat_col),
            Some(&observed_traj.lng_col),
            ctx.evaluation_h3_resolution,
            obs_end_col.as_deref(),
        )? {
            Some(result) => {
                if let Some(warning) = result.warning {
                    warnings.push(warning);
                }
                let sorted = result
                    .visits
                    .lazy()
                    .sort(["uid", "start_timestamp"], SortMultipleOptions::default())
                    .collect()?;
                filter_visits(Some(&sorted), &filter)?.filter(|df| df.height() > 0)
            }
            None => None,
        }
    } else {
        None
    };

    let mut group = build_activity_block(ctx, &syn_filtered, observed_filtered.as_ref())?;
    if let Some(obj) = group.as_object_mut() {
        obj.insert("filter_key".to_string(), json!(filter.key));
        obj.insert("filter_label".to_string(), json!(filter.label));
    }
    payload["activity"] = json!({"groups": [group]});
    if !warnings.is_empty() {
        payload["warnings"] = json!(warnings);
    }
    Ok(payload)
}
