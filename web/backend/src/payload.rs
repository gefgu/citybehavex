//! Progressive chart payload assembly for the axum backend.
//!
//! This mirrors the public JSON contract in `web/frontend/src/api.ts` and
//! `legacy Python backend/payload/sections.py`. Expensive section internals are
//! filled in incrementally by `comparison::sections::*`; the base payload is
//! intentionally light so first paint does not wait on every chart.

use crate::columns::{
    ACTIVITY_CANDIDATES, DURATION_CANDIDATES, END_TS_CANDIDATES, LOCATION_CANDIDATES, detect_in,
};
use crate::comparison::filters::{
    FilterMeta, PublicFilter, SpecialDay, filter_visits, filters, special_day_filters, time_filters,
};
use crate::comparison::panel::AdaptationMode;
use crate::comparison::trajectory::load_trajectory;
use crate::comparison::transport::transport_mode_map;
use crate::comparison::visits::prepare_activity_visits;
use crate::config::repo_root;
use crate::experiments::Experiment;
use crate::settings::reports::EvaluationAdaptationMode;
use polars::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;

pub const SECTION_NAMES: &[&str] = &[
    "distributions",
    "metrics",
    "transport-spatial",
    "activity",
    "mobility-laws",
    "micro-activity",
    "time-use",
    "motifs",
    "stvd",
    "profiles",
    "social-network",
];

#[derive(Debug, Clone)]
pub struct ComparisonContext {
    pub synthetic_path: PathBuf,
    pub observed_path: Option<PathBuf>,
    pub observed_label: String,
    pub synthetic_activities_path: Option<PathBuf>,
    pub time_use_path: Option<PathBuf>,
    pub time_use_label: String,
    pub time_use_country: Option<String>,
    pub time_use_survey: Option<i64>,
    pub time_use_weight_col: String,
    pub special_days: Vec<SpecialDay>,
    pub evaluation_mode: AdaptationMode,
    pub evaluation_location_col: Option<String>,
    pub evaluation_h3_resolution: u8,
    pub transport_enabled: bool,
    pub transport_observed_enabled: bool,
    pub transport_synthetic_moving_path: Option<PathBuf>,
    pub transport_uid_col: Option<String>,
    pub transport_datetime_col: Option<String>,
    pub transport_lat_col: Option<String>,
    pub transport_lng_col: Option<String>,
    pub transport_col: Option<String>,
    pub transport_mode_map: HashMap<String, String>,
}

impl ComparisonContext {
    pub fn from_experiment(exp: &Experiment, run: &crate::experiments::Run) -> Self {
        Self {
            synthetic_path: run.path.clone(),
            observed_path: exp.observed_path.as_ref().filter(|p| p.exists()).cloned(),
            observed_label: exp.label.clone(),
            synthetic_activities_path: Some(run.activities_path()).filter(|p| p.exists()),
            time_use_path: exp.time_use_path.as_ref().filter(|p| p.exists()).cloned(),
            time_use_label: exp.time_use_label.clone(),
            time_use_country: exp.time_use_country.clone(),
            time_use_survey: exp.time_use_survey,
            time_use_weight_col: exp.time_use_weight_col.clone(),
            special_days: exp
                .special_days
                .iter()
                .map(|sd| SpecialDay {
                    name: sd.name.clone(),
                    start_date: sd.start_date.clone(),
                    end_date: sd.end_date.clone(),
                })
                .collect(),
            evaluation_mode: match exp.evaluation_adaptation_config.mode {
                EvaluationAdaptationMode::Auto => AdaptationMode::Auto,
                EvaluationAdaptationMode::Force => AdaptationMode::Force,
                EvaluationAdaptationMode::Off => AdaptationMode::Off,
            },
            evaluation_location_col: exp.evaluation_adaptation_config.location_col.clone(),
            evaluation_h3_resolution: exp.evaluation_adaptation_config.h3_resolution as u8,
            transport_enabled: exp.transport_spatial_config.enabled,
            transport_observed_enabled: exp.transport_spatial_config.observed_enabled,
            transport_synthetic_moving_path: exp
                .transport_spatial_config
                .synthetic_moving_path
                .as_deref()
                .map(|p| {
                    let path = PathBuf::from(p);
                    if path.is_absolute() {
                        path
                    } else {
                        repo_root().join(path)
                    }
                }),
            transport_uid_col: exp.transport_spatial_config.uid_col.clone(),
            transport_datetime_col: exp.transport_spatial_config.datetime_col.clone(),
            transport_lat_col: exp.transport_spatial_config.lat_col.clone(),
            transport_lng_col: exp.transport_spatial_config.lng_col.clone(),
            transport_col: exp.transport_spatial_config.transport_col.clone(),
            transport_mode_map: transport_mode_map(&exp.transport_spatial_config.mode_map),
        }
    }

    pub fn mode(&self) -> &'static str {
        if self.observed_path.is_some() {
            "comparison"
        } else {
            "synthetic_only"
        }
    }

    pub fn labels(&self) -> Value {
        match self.observed_path {
            Some(_) => json!({"synthetic": "synthetic", "observed": self.observed_label}),
            None => json!({"synthetic": "synthetic"}),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmptyMetrics {
    pub wasserstein: Vec<Value>,
    pub jsd: Vec<Value>,
    pub cpc: Vec<Value>,
    pub time_use: Vec<Value>,
    pub stvd: Vec<Value>,
}

fn empty_metrics() -> EmptyMetrics {
    EmptyMetrics {
        wasserstein: Vec::new(),
        jsd: Vec::new(),
        cpc: Vec::new(),
        time_use: Vec::new(),
        stvd: Vec::new(),
    }
}

pub fn available_filters(ctx: &ComparisonContext) -> Vec<FilterMeta> {
    let mut out = filters();
    out.extend(special_day_filters(&ctx.special_days));
    out
}

pub fn distribution_filters(ctx: &ComparisonContext) -> Vec<FilterMeta> {
    let mut out = available_filters(ctx);
    out.extend(time_filters());
    out
}

fn public_filters(filters: Vec<FilterMeta>) -> Vec<PublicFilter> {
    filters.into_iter().map(|f| f.public()).collect()
}

pub fn empty_chart_payload(ctx: &ComparisonContext, loaded_filters: Vec<String>) -> Value {
    json!({
        "mode": ctx.mode(),
        "labels": ctx.labels(),
        "available_filters": public_filters(available_filters(ctx)),
        "distribution_filters": public_filters(distribution_filters(ctx)),
        "enabled_sections": SECTION_NAMES,
        "loaded_filters": loaded_filters,
        "metrics": empty_metrics(),
        "ecdf": {"groups": []},
        "transport_spatial": null,
        "mobility_laws": null,
        "activity": null,
        "micro_activity_usage": null,
        "time_use_comparison": null,
        "profiles": null,
        "motifs": null,
        "stvd": null,
        "social_network": null,
        "warnings": [],
    })
}

pub fn chart_base_payload(ctx: &ComparisonContext) -> Value {
    let payload = empty_chart_payload(ctx, Vec::new());
    if ctx.observed_path.is_none() {
        // Python only warns here when an explicit observed path was supplied
        // but is missing. `ComparisonContext` has already resolved "missing"
        // to `None`; the routes add the exact warning when they have access
        // to the configured path.
    }
    payload
}

pub fn metrics_export_payload(ctx: &ComparisonContext, artifact: &Value) -> Value {
    let empty = empty_chart_payload(ctx, Vec::new());
    let time_use_table: Vec<Value> = artifact
        .get("time_use_comparison")
        .and_then(|v| v.get("groups"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|group| {
            let filter_key = group.get("filter_key").cloned().unwrap_or(json!("all"));
            let filter_label = group.get("filter_label").cloned().unwrap_or(json!("All"));
            group
                .get("block")
                .and_then(|v| v.get("rows"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(move |row| {
                    let mut obj = row.as_object().cloned().unwrap_or_default();
                    obj.insert("filter_key".to_string(), filter_key.clone());
                    obj.insert("filter_label".to_string(), filter_label.clone());
                    Value::Object(obj)
                })
        })
        .collect();

    json!({
        "mode": artifact.get("mode").cloned().unwrap_or_else(|| json!(ctx.mode())),
        "labels": artifact.get("labels").cloned().unwrap_or_else(|| ctx.labels()),
        "filters": artifact.get("distribution_filters").cloned().unwrap_or_else(|| empty["distribution_filters"].clone()),
        "metrics": artifact.get("metrics").cloned().unwrap_or_else(|| empty["metrics"].clone()),
        "time_use_table": time_use_table,
        "warnings": artifact.get("warnings").cloned().unwrap_or_else(|| json!([])),
    })
}

pub fn chart_section_payload(
    ctx: &ComparisonContext,
    section: &str,
    filter_key: &str,
) -> anyhow::Result<Value> {
    if !SECTION_NAMES.contains(&section) {
        anyhow::bail!("unknown chart section: {section}");
    }

    match section {
        "distributions" => {
            crate::comparison::sections::distributions::distributions_section_payload(
                ctx, filter_key,
            )
        }
        "metrics" => crate::comparison::sections::metrics::metrics_section_payload(ctx, filter_key),
        "transport-spatial" => {
            crate::comparison::sections::transport_spatial::transport_spatial_section_payload(ctx)
        }
        "activity" => {
            crate::comparison::sections::activity::activity_section_payload(ctx, filter_key)
        }
        "mobility-laws" => {
            crate::comparison::sections::mobility_laws::mobility_laws_section_payload(
                ctx, filter_key,
            )
        }
        "micro-activity" => {
            crate::comparison::sections::micro_activity::micro_activity_section_payload(
                ctx, filter_key,
            )
        }
        "time-use" => {
            crate::comparison::sections::time_use::time_use_section_payload(ctx, filter_key)
        }
        "motifs" => crate::comparison::sections::motifs::motifs_section_payload(ctx, filter_key),
        "stvd" => crate::comparison::sections::stvd::stvd_section_payload(ctx, filter_key),
        "profiles" => crate::comparison::sections::profiles::profiles_section_payload(ctx),
        "social-network" => {
            crate::comparison::sections::social_network::social_network_section_payload(ctx)
        }
        _ => unreachable!("SECTION_NAMES and chart_section_payload match arms are out of sync"),
    }
}

/// Mirrors `legacy Python backend/payload/sections.py::build_metrics_export_payload`,
/// which computes `legacy._build_comparison_payload(..., filter_keys=None,
/// sections=["metrics", "time-use"])` -- Python's single call internally
/// loops over every filter and folds the results into one artifact, whereas
/// this port's `chart_section_payload` only ever computes one filter at a
/// time. Reproduce the same shape here by calling `metrics_section_payload`
/// once per filter and merging the filter-varying arrays.
///
/// Two different filter sets are involved, matching Python's own per-metric
/// scope (confirmed against real `gparis_simulation` data: `/charts/metrics`
/// for a non-regular filter like `morning` already returns populated
/// `wasserstein`/`stvd` rows for it, but empty `jsd`/`cpc` and a `null`
/// `time_use_comparison`): `wasserstein`/`stvd` are computed for every
/// *distribution* filter (`all`/`weekday`/`weekend`/time-of-day buckets/
/// special days), while `jsd`/`cpc`/`time_use`/`time_use_comparison` only
/// make sense for the *regular* filters (`all`/`weekday`/`weekend`/special
/// days -- no time-of-day buckets), since those need full visit/observed
/// data rather than a distribution slice.
pub fn metrics_export_artifact(ctx: &ComparisonContext) -> anyhow::Result<Value> {
    let regular_filters = available_filters(ctx);
    let mut regular_payloads = Vec::with_capacity(regular_filters.len());
    for filter in &regular_filters {
        regular_payloads.push(chart_section_payload(ctx, "metrics", &filter.key)?);
    }

    let mut merged: Option<Value> = None;
    for payload in &regular_payloads {
        match &mut merged {
            None => merged = Some(payload.clone()),
            Some(base) => {
                for key in ["cpc", "time_use"] {
                    let mut rows = base["metrics"][key].as_array().cloned().unwrap_or_default();
                    rows.extend(
                        payload["metrics"][key]
                            .as_array()
                            .cloned()
                            .unwrap_or_default(),
                    );
                    base["metrics"][key] = Value::Array(rows);
                }
                let mut groups = base["time_use_comparison"]["groups"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                groups.extend(
                    payload["time_use_comparison"]["groups"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default(),
                );
                if !groups.is_empty() {
                    base["time_use_comparison"] = json!({"groups": groups});
                }
                let mut warnings = base["warnings"].as_array().cloned().unwrap_or_default();
                warnings.extend(payload["warnings"].as_array().cloned().unwrap_or_default());
                base["warnings"] = Value::Array(warnings);
            }
        }
    }
    if let Some(base) = &mut merged {
        // Mirrors `legacy.py`'s two separate cross-filter loops for `jsd`:
        // the "activity" JSD side effect (Activity distribution/transitions/
        // Daily activity profile) runs to completion across every filter
        // *before* the separate "Daily motifs" JSD loop starts over every
        // filter again -- so the merged order is
        // [every filter's activity-JSD rows][every filter's motifs-JSD row],
        // not filter-major. Split by `metric_name` rather than assuming a
        // fixed position, since either group can be empty for a given
        // filter (e.g. no observed data).
        let mut activity_group = Vec::new();
        let mut motifs_group = Vec::new();
        for payload in &regular_payloads {
            for row in payload["metrics"]["jsd"]
                .as_array()
                .cloned()
                .unwrap_or_default()
            {
                if row.get("metric_name").and_then(Value::as_str) == Some("Daily motifs") {
                    motifs_group.push(row);
                } else {
                    activity_group.push(row);
                }
            }
        }
        activity_group.extend(motifs_group);
        base["metrics"]["jsd"] = Value::Array(activity_group);

        // wasserstein/stvd span every distribution filter, not just the
        // regular ones -- recompute across that wider set (re-fetching the
        // 3 regular filters here too rather than reusing `regular_payloads`
        // keeps this loop's ordering independent of the jsd/cpc/time_use
        // loop above, matching Python's separate cross-filter loops).
        let dist_filters = distribution_filters(ctx);
        let mut wasserstein_rows = Vec::new();
        let mut stvd_rows = Vec::new();
        for filter in &dist_filters {
            let payload = chart_section_payload(ctx, "metrics", &filter.key)?;
            wasserstein_rows.extend(
                payload["metrics"]["wasserstein"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default(),
            );
            stvd_rows.extend(
                payload["metrics"]["stvd"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default(),
            );
        }
        base["metrics"]["wasserstein"] = Value::Array(wasserstein_rows);
        base["metrics"]["stvd"] = Value::Array(stvd_rows);
    }
    Ok(merged.unwrap_or_else(|| chart_base_payload(ctx)))
}

pub(crate) fn choose_filter(
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<FilterMeta> {
    distribution_filters(ctx)
        .into_iter()
        .find(|f| f.key == filter_key)
        .ok_or_else(|| anyhow::anyhow!("unknown filter: {filter_key}"))
}

pub(crate) fn choose_regular_filter(
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<Option<FilterMeta>> {
    choose_filter(ctx, filter_key)?;
    Ok(available_filters(ctx)
        .into_iter()
        .find(|f| f.key == filter_key))
}

pub(crate) fn duration_col(df: &polars::prelude::DataFrame) -> Option<String> {
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    detect_in(&cols, DURATION_CANDIDATES)
}

pub(crate) fn detected_col(df: &DataFrame, candidates: &[&str]) -> Option<String> {
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    detect_in(&cols, candidates)
}

/// Visits prepared for both sides, sorted and filtered down to a single
/// filter -- shared by `metrics_section_payload` (the "Daily motifs" JSD
/// side effect) and `motifs_section_payload`. Mirrors `_build_comparison_payload`'s
/// `synthetic_visits`/`observed_visits` computation via `features.get_activity_visits`
/// (same underlying `prepare_activity_visits` pipeline `activity_section_payload`
/// uses), filtered per-meta the way `_filter_visits` is applied downstream.
pub(crate) struct PreparedVisits {
    pub(crate) synthetic: Option<DataFrame>,
    pub(crate) observed: Option<DataFrame>,
    pub(crate) warnings: Vec<String>,
}

pub(crate) fn prepared_visits_for_filter(
    ctx: &ComparisonContext,
    filter: &FilterMeta,
) -> anyhow::Result<PreparedVisits> {
    let mut warnings = Vec::new();

    let synthetic_traj = load_trajectory(&ctx.synthetic_path)?;
    let synth_activity_col = detected_col(&synthetic_traj.df, ACTIVITY_CANDIDATES);
    let synth_location_col = detected_col(&synthetic_traj.df, LOCATION_CANDIDATES);
    let synthetic = match prepare_activity_visits(
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
    )? {
        Some(result) => {
            if let Some(w) = result.warning {
                warnings.push(w);
            }
            let sorted = result
                .visits
                .lazy()
                .sort(["uid", "start_timestamp"], SortMultipleOptions::default())
                .collect()?;
            filter_visits(Some(&sorted), filter)?.filter(|df| df.height() > 0)
        }
        None => None,
    };

    let observed = if let Some(path) = &ctx.observed_path {
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
                if let Some(w) = result.warning {
                    warnings.push(w);
                }
                let sorted = result
                    .visits
                    .lazy()
                    .sort(["uid", "start_timestamp"], SortMultipleOptions::default())
                    .collect()?;
                filter_visits(Some(&sorted), filter)?.filter(|df| df.height() > 0)
            }
            None => None,
        }
    } else {
        None
    };

    Ok(PreparedVisits {
        synthetic,
        observed,
        warnings,
    })
}

fn distribution_summary(values: &[f64]) -> Value {
    let mut clean: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.is_empty() {
        return json!({"count": 0, "mean": null, "median": null, "std": null, "p10": null, "p90": null});
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let count = clean.len();
    let mean = clean.iter().sum::<f64>() / count as f64;
    let std = (clean.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64).sqrt();
    let q = |p: f64| {
        let pos = p * (count.saturating_sub(1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            clean[lo]
        } else {
            clean[lo] * (1.0 - (pos - lo as f64)) + clean[hi] * (pos - lo as f64)
        }
    };
    json!({
        "count": count,
        "mean": mean,
        "median": q(0.5),
        "std": std,
        "p10": q(0.1),
        "p90": q(0.9),
    })
}

fn wasserstein_1d(left: &[f64], right: &[f64]) -> Option<f64> {
    let mut a: Vec<f64> = left.iter().copied().filter(|v| v.is_finite()).collect();
    let mut b: Vec<f64> = right.iter().copied().filter(|v| v.is_finite()).collect();
    if a.is_empty() || b.is_empty() {
        return None;
    }
    a.sort_by(|x, y| x.total_cmp(y));
    b.sort_by(|x, y| x.total_cmp(y));
    let n = a.len().max(b.len());
    let mut total = 0.0;
    for i in 0..n {
        let qa = a[((i * a.len()) / n).min(a.len() - 1)];
        let qb = b[((i * b.len()) / n).min(b.len() - 1)];
        total += (qa - qb).abs();
    }
    Some(total / n as f64)
}

fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn degree_preserving_random_edges(degrees: &[f64], seed: i64) -> (Vec<u32>, Vec<u32>) {
    let n = degrees.len();
    let total: f64 = degrees.iter().sum();
    if n <= 1 || total <= 0.0 {
        return (Vec::new(), Vec::new());
    }
    let mut state = seed as u64 ^ 0x9e37_79b9_7f4a_7c15;
    let mut from = Vec::new();
    let mut to = Vec::new();
    for i in 0..n.saturating_sub(1) {
        if degrees[i] <= 0.0 {
            continue;
        }
        for j in (i + 1)..n {
            let p = (degrees[i] * degrees[j] / total).clamp(0.0, 1.0);
            if lcg_next(&mut state) < p {
                from.push(i as u32);
                to.push(j as u32);
            }
        }
    }
    (from, to)
}

fn network_block_from_edges(
    node_count: usize,
    edge_from: &[u32],
    edge_to: &[u32],
    source_sidecar: Option<&Value>,
    kind: &str,
) -> Value {
    const MAX_EDGES: usize = 20_000;
    let metrics = fastmob_core::measures::collective::co_presence_network::compute_graph_metrics(
        node_count, edge_from, edge_to,
    );
    // O(node_count + edge_count) degree accumulation, matching
    // `NetworkGraph.degrees()`'s `np.bincount` on the Python side -- the
    // previous per-node `.filter().count()` scan was O(node_count *
    // edge_count), which is fine for the small synthetic graph but
    // catastrophic for the observed co-presence graph (tens of thousands of
    // nodes, tens of millions of edges).
    let mut degrees = vec![0.0f64; node_count];
    for &u in edge_from {
        degrees[u as usize] += 1.0;
    }
    for &v in edge_to {
        degrees[v as usize] += 1.0;
    }
    let max_degree = degrees.iter().copied().fold(0.0f64, f64::max).max(1.0);
    let source_nodes = source_sidecar
        .and_then(|v| v.get("nodes"))
        .and_then(Value::as_array);
    let mut nodes = Vec::new();
    for i in 0..node_count {
        if let Some(row) = source_nodes
            .and_then(|nodes| nodes.get(i))
            .and_then(Value::as_array)
        {
            let mut row = row.clone();
            if row.len() >= 3 {
                row[2] =
                    json!((3.0 + 13.0 * (degrees[i] / max_degree).sqrt() * 10.0).round() / 10.0);
            }
            nodes.push(Value::Array(row));
        } else {
            nodes.push(json!([
                0.0,
                0.0,
                (3.0 + 13.0 * (degrees[i] / max_degree).sqrt() * 10.0).round() / 10.0,
                i + 1
            ]));
        }
    }
    let edge_count = edge_from.len();
    let step = (edge_count / MAX_EDGES).max(1);
    let edges: Vec<Value> = edge_from
        .iter()
        .zip(edge_to.iter())
        .enumerate()
        .filter(|(idx, _)| *idx % step == 0)
        .take(MAX_EDGES)
        .map(|(_, (&u, &v))| json!([u, v, 1.0]))
        .collect();
    json!({
        "kind": kind,
        "node_count": node_count,
        "edge_count": edge_count,
        "layout": source_sidecar.and_then(|v| v.get("layout")).cloned().unwrap_or(json!("source_layout")),
        "directed": false,
        "social_graph_k": source_sidecar.and_then(|v| v.get("social_graph_k")).cloned().unwrap_or(json!(0)),
        "nodes": nodes,
        "edges": edges,
        "edges_sampled": edge_count > MAX_EDGES,
        "degrees": degrees,
        "_metric_cache": {
            "degree": degrees,
            "clustering_coefficient": metrics.clustering_coefficient,
            "edge_persistence": Vec::<f64>::new(),
            "topological_overlap": metrics.topological_overlap,
        }
    })
}

const NETWORK_METRIC_NAMES: [&str; 4] = [
    "degree",
    "clustering_coefficient",
    "edge_persistence",
    "topological_overlap",
];

fn metric_array(metrics: &Value, name: &str) -> Vec<f64> {
    metrics
        .get(name)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_f64)
        .collect()
}

/// Mirrors `network_validation.py::_validation_block`: builds a
/// source-graph-vs-degree-preserving-random-graph comparison block (used for
/// both `synthetic_vs_random` and `observed_vs_random`). Random-baseline
/// generation is skipped above 5000 nodes (an intentional Rust-only
/// performance guard the Python port doesn't have -- see
/// `RUST_BACKEND_MIGRATION.md`; Python's `degree_preserving_random_graph` is
/// impractically slow at observed-graph scale, e.g. Shanghai's ~58k users).
fn validation_block(
    comparison: &str,
    source_label: &str,
    node_count: usize,
    edge_from: &[u32],
    edge_to: &[u32],
    source_sidecar: Option<&Value>,
    source_kind: &str,
    seed: i64,
) -> (Value, Vec<String>, Value) {
    let source =
        network_block_from_edges(node_count, edge_from, edge_to, source_sidecar, source_kind);
    let source_metrics = source
        .get("_metric_cache")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let degrees = metric_array(&source_metrics, "degree");

    let (rand_from, rand_to) = if node_count <= 5000 {
        degree_preserving_random_edges(&degrees, seed)
    } else {
        (Vec::new(), Vec::new())
    };
    let random = network_block_from_edges(
        node_count,
        &rand_from,
        &rand_to,
        source_sidecar,
        "degree_preserving_rnd",
    );
    let random_metrics = random
        .get("_metric_cache")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let mut wasserstein = serde_json::Map::new();
    let mut source_dist = serde_json::Map::new();
    let mut random_dist = serde_json::Map::new();
    for name in NETWORK_METRIC_NAMES {
        let left = metric_array(&source_metrics, name);
        let right = metric_array(&random_metrics, name);
        wasserstein.insert(
            name.to_string(),
            serde_json::to_value(wasserstein_1d(&left, &right)).unwrap(),
        );
        source_dist.insert(name.to_string(), distribution_summary(&left));
        random_dist.insert(name.to_string(), distribution_summary(&right));
    }
    let mut source_public = source;
    let mut random_public = random;
    source_public
        .as_object_mut()
        .map(|o| o.remove("_metric_cache"));
    random_public
        .as_object_mut()
        .map(|o| o.remove("_metric_cache"));

    let mut warnings = Vec::new();
    if node_count > 5000 {
        warnings.push(format!(
            "{comparison}: random baseline skipped for graphs above 5000 nodes"
        ));
    }
    (
        json!({
            "comparison": comparison,
            "random_model": "degree_preserving_rnd",
            "wasserstein": wasserstein,
            "distributions": {source_label: source_dist, "random": random_dist},
            "source_network": source_public,
            "random_network": random_public,
        }),
        warnings,
        source_metrics,
    )
}

/// Mirrors `network_validation.py::_metric_wasserstein_block`: a
/// no-random-baseline diff between two already-computed metric bundles, used
/// for `synthetic_vs_observed`.
fn metric_wasserstein_block(
    comparison: &str,
    left_label: &str,
    left_metrics: &Value,
    right_label: &str,
    right_metrics: &Value,
) -> (Value, Vec<String>) {
    let mut wasserstein = serde_json::Map::new();
    let mut left_dist = serde_json::Map::new();
    let mut right_dist = serde_json::Map::new();
    let mut warnings = Vec::new();
    for name in NETWORK_METRIC_NAMES {
        let left = metric_array(left_metrics, name);
        let right = metric_array(right_metrics, name);
        let w = wasserstein_1d(&left, &right);
        if w.is_none() {
            warnings.push(format!(
                "{comparison}: {name} distribution is empty; Wasserstein unavailable"
            ));
        }
        wasserstein.insert(name.to_string(), serde_json::to_value(w).unwrap());
        left_dist.insert(name.to_string(), distribution_summary(&left));
        right_dist.insert(name.to_string(), distribution_summary(&right));
    }
    (
        json!({
            "comparison": comparison,
            "wasserstein": wasserstein,
            "distributions": {left_label: left_dist, right_label: right_dist},
        }),
        warnings,
    )
}

fn synthetic_social_graph(
    social_path: &std::path::Path,
) -> anyhow::Result<(usize, Vec<u32>, Vec<u32>, Value)> {
    let data: Value = serde_json::from_slice(&std::fs::read(social_path)?)?;
    let node_count = data
        .get("node_count")
        .and_then(Value::as_u64)
        .or_else(|| {
            data.get("nodes")
                .and_then(Value::as_array)
                .map(|v| v.len() as u64)
        })
        .unwrap_or(0) as usize;
    let mut edge_from = Vec::<u32>::new();
    let mut edge_to = Vec::<u32>::new();
    if let Some(edges) = data.get("edges").and_then(Value::as_array) {
        for edge in edges {
            let Some(row) = edge.as_array() else { continue };
            if row.len() < 2 {
                continue;
            }
            let Some(u) = row[0].as_u64() else { continue };
            let Some(v) = row[1].as_u64() else { continue };
            if u < node_count as u64 && v < node_count as u64 && u != v {
                edge_from.push(u.min(v) as u32);
                edge_to.push(u.max(v) as u32);
            }
        }
    }
    Ok((node_count, edge_from, edge_to, data))
}

#[allow(clippy::too_many_arguments)]
pub fn network_validation_payload(
    enabled: bool,
    synthetic_enabled: bool,
    social_path: Option<&std::path::Path>,
    // `None` -> observed validation not requested (config `observed_enabled: false`).
    // `Some(Err(reason))` -> requested but unavailable (no observed data, or no
    // detectable user/datetime columns) -- mirrors `build_network_validation`'s
    // "observed dataframe unavailable" / `_observed_validation_block`'s
    // "requires user and datetime columns" early-outs, neither of which builds
    // an `observed_vs_random` block.
    // `Some(Ok(graph))` -> built (possibly with its own data-quality warnings,
    // e.g. an empty graph -- that case still gets a real, if degenerate, block).
    observed: Option<Result<&crate::comparison::network_validation::ObservedGraph, &str>>,
    seed: i64,
) -> Value {
    if !enabled {
        return json!({"network_validation": null, "warnings": []});
    }
    let mut network_validation = serde_json::Map::new();
    let mut warnings = Vec::new();
    let mut synthetic_metrics: Option<Value> = None;

    if synthetic_enabled {
        match social_path.filter(|p| p.exists()) {
            None => {
                warnings.push("synthetic_vs_random: social network sidecar not found".to_string())
            }
            Some(path) => match synthetic_social_graph(path) {
                Err(err) => warnings.push(format!("synthetic_vs_random: {err}")),
                Ok((node_count, edge_from, edge_to, data)) => {
                    let (block, block_warnings, metrics) = validation_block(
                        "synthetic_vs_random",
                        "synthetic",
                        node_count,
                        &edge_from,
                        &edge_to,
                        Some(&data),
                        "synthetic_social",
                        seed,
                    );
                    network_validation.insert("synthetic_vs_random".to_string(), block);
                    warnings.extend(block_warnings);
                    synthetic_metrics = Some(metrics);
                }
            },
        }
    }

    if let Some(observed) = observed {
        let observed = match observed {
            Err(reason) => {
                warnings.push(format!("observed_vs_random: {reason}"));
                None
            }
            Ok(graph) => Some(graph),
        };
        let Some(observed) = observed else {
            return finish_network_validation_payload(network_validation, warnings);
        };
        warnings.extend(
            observed
                .warnings
                .iter()
                .map(|w| format!("observed_vs_random: {w}")),
        );
        let (block, block_warnings, observed_metrics) = validation_block(
            "observed_vs_random",
            "observed",
            observed.node_count,
            &observed.edge_from,
            &observed.edge_to,
            None,
            "observed_daily_copresence",
            seed,
        );
        network_validation.insert("observed_vs_random".to_string(), block);
        warnings.extend(block_warnings);

        if let Some(synthetic_metrics) = &synthetic_metrics {
            let (diff_block, diff_warnings) = metric_wasserstein_block(
                "synthetic_vs_observed",
                "synthetic",
                synthetic_metrics,
                "observed",
                &observed_metrics,
            );
            network_validation.insert("synthetic_vs_observed".to_string(), diff_block);
            warnings.extend(diff_warnings);
        }
    }

    finish_network_validation_payload(network_validation, warnings)
}

fn finish_network_validation_payload(
    network_validation: serde_json::Map<String, Value>,
    warnings: Vec<String>,
) -> Value {
    let network_validation = if network_validation.is_empty() {
        Value::Null
    } else {
        Value::Object(network_validation)
    };
    json!({
        "network_validation": network_validation,
        "warnings": warnings,
    })
}
