//! Progressive chart payload assembly for the axum backend.
//!
//! This mirrors the public JSON contract in `web/frontend/src/api.ts` and
//! `web/backend/app/payload/sections.py`. Expensive section internals are
//! filled in incrementally by `comparison::sections::*`; the base payload is
//! intentionally light so first paint does not wait on every chart.

use crate::columns::{
    ACTIVITY_CANDIDATES, DURATION_CANDIDATES, END_TS_CANDIDATES, LOCATION_CANDIDATES, detect_in,
};
use crate::comparison::CAR_SPEED_KMH;
use crate::comparison::activity::{activity_transition_matrix, daily_activity_distribution};
use crate::comparison::ecdf::{ecdf_block, transport_ecdf_block};
use crate::comparison::filters::{
    FilterMeta, PublicFilter, SpecialDay, filter_df, filter_visits, filters, special_day_filters,
    time_filters,
};
use crate::comparison::metrics::waiting_times_minutes;
use crate::comparison::micro_activity::micro_activity_daily_usage_data;
use crate::comparison::mobility_laws::{
    daily_location_lognormal_dataset, distance_frequency_dataset, mobility_law_visits,
    truncated_powerlaw_dataset,
};
use crate::comparison::panel::AdaptationMode;
use crate::comparison::panel::{adapt_evaluation_dataframe, collapse_to_stays};
use crate::comparison::sections::metrics::{Side, wasserstein_metric_rows};
use crate::comparison::sections::motifs::build_motifs_block;
use crate::comparison::stvd::compute_stvd_layers;
use crate::comparison::trajectory::{load_trajectory, read_parquet};
use crate::comparison::transport::{
    default_synthetic_moving_path, observed_transport_leg_records, synthetic_transport_leg_records,
    transport_mode_map, transport_spatial_summary,
};
use crate::comparison::visits::prepare_activity_visits;
use crate::config::repo_root;
use crate::datasource::quote_path;
use crate::experiments::Experiment;
use crate::settings::reports::EvaluationAdaptationMode;
use chrono::{Datelike, NaiveDateTime};
use polars::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

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
        "distributions" => distributions_section_payload(ctx, filter_key),
        "metrics" => metrics_section_payload(ctx, filter_key),
        "transport-spatial" => transport_spatial_section_payload(ctx),
        "activity" => activity_section_payload(ctx, filter_key),
        "mobility-laws" => mobility_laws_section_payload(ctx, filter_key),
        "micro-activity" => micro_activity_section_payload(ctx, filter_key),
        "time-use" => time_use_section_payload(ctx, filter_key),
        "motifs" => motifs_section_payload(ctx, filter_key),
        "stvd" => stvd_section_payload(ctx, filter_key),
        "profiles" => profiles_section_payload(ctx),
        "social-network" => social_network_section_payload(ctx),
        _ => unreachable!("SECTION_NAMES and chart_section_payload match arms are out of sync"),
    }
}

fn choose_filter(ctx: &ComparisonContext, filter_key: &str) -> anyhow::Result<FilterMeta> {
    distribution_filters(ctx)
        .into_iter()
        .find(|f| f.key == filter_key)
        .ok_or_else(|| anyhow::anyhow!("unknown filter: {filter_key}"))
}

fn choose_regular_filter(
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<Option<FilterMeta>> {
    choose_filter(ctx, filter_key)?;
    Ok(available_filters(ctx)
        .into_iter()
        .find(|f| f.key == filter_key))
}

fn duration_col(df: &polars::prelude::DataFrame) -> Option<String> {
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    detect_in(&cols, DURATION_CANDIDATES)
}

fn micro_activity_datetime_col(df: &DataFrame) -> Option<String> {
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    detect_in(&cols, &["arrival", "start_timestamp", "datetime"])
}

fn detected_col(df: &DataFrame, candidates: &[&str]) -> Option<String> {
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    detect_in(&cols, candidates)
}

/// Visits prepared for both sides, sorted and filtered down to a single
/// filter -- shared by `metrics_section_payload` (the "Daily motifs" JSD
/// side effect) and `motifs_section_payload`. Mirrors `_build_comparison_payload`'s
/// `synthetic_visits`/`observed_visits` computation via `features.get_activity_visits`
/// (same underlying `prepare_activity_visits` pipeline `activity_section_payload`
/// uses), filtered per-meta the way `_filter_visits` is applied downstream.
struct PreparedVisits {
    synthetic: Option<DataFrame>,
    observed: Option<DataFrame>,
    warnings: Vec<String>,
}

fn prepared_visits_for_filter(
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

fn metrics_section_payload(ctx: &ComparisonContext, filter_key: &str) -> anyhow::Result<Value> {
    let filter = choose_filter(ctx, filter_key)?;
    let synthetic_traj = load_trajectory(&ctx.synthetic_path)?;
    let observed_traj = match &ctx.observed_path {
        Some(path) => Some(load_trajectory(path)?),
        None => None,
    };
    let synthetic_side = Side {
        df: &synthetic_traj.df,
        uid_col: &synthetic_traj.uid_col,
        lat_col: &synthetic_traj.lat_col,
        lng_col: &synthetic_traj.lng_col,
        datetime_col: &synthetic_traj.datetime_col,
        label: "synthetic",
        duration_col: None,
    };
    let observed_duration = observed_traj.as_ref().and_then(|t| duration_col(&t.df));
    let observed_side = observed_traj.as_ref().map(|traj| Side {
        df: &traj.df,
        uid_col: &traj.uid_col,
        lat_col: &traj.lat_col,
        lng_col: &traj.lng_col,
        datetime_col: &traj.datetime_col,
        label: &ctx.observed_label,
        duration_col: observed_duration.as_deref(),
    });
    let rows = wasserstein_metric_rows(
        &synthetic_side,
        observed_side.as_ref(),
        &[filter.clone()],
        ctx.evaluation_mode,
        ctx.evaluation_location_col.as_deref(),
        ctx.evaluation_h3_resolution,
    )?;

    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    payload["metrics"]["wasserstein"] = serde_json::to_value(rows)?;

    // Mirrors Python's `(wants("motifs") or wants("metrics"))` gate in
    // `legacy.py::_build_comparison_payload`: the metrics section
    // internally recomputes the motifs distribution purely for the "Daily
    // motifs" Jensen-Shannon-divergence side effect on `metrics.jsd` -- the
    // motifs block itself is discarded here, matching
    // `payload/sections.py::build_section_metrics`, which only copies
    // `artifact["metrics"]` (never `artifact["motifs"]`) out of the shared
    // artifact. `/charts/motifs` (`motifs_section_payload`) is the route
    // that surfaces the block itself, and mirrors `build_section_motifs`
    // by discarding this same jsd computation in the other direction.
    let visits = prepared_visits_for_filter(ctx, &filter)?;
    let mut jsd = Vec::new();
    if visits.synthetic.is_some() || visits.observed.is_some() {
        build_motifs_block(
            &ctx.observed_label,
            visits.observed.as_ref(),
            visits.synthetic.as_ref(),
            &filter,
            &mut jsd,
        )?;
    }
    payload["metrics"]["jsd"] = serde_json::to_value(jsd)?;
    if !visits.warnings.is_empty() {
        payload["warnings"] = json!(visits.warnings);
    }
    if let Ok(time_use_payload) = time_use_section_payload(ctx, filter_key) {
        if !time_use_payload
            .get("time_use_comparison")
            .is_some_and(Value::is_null)
        {
            payload["time_use_comparison"] = time_use_payload["time_use_comparison"].clone();
            payload["metrics"]["time_use"] =
                time_use_metric_rows(&time_use_payload["time_use_comparison"]);
        }
        if let Some(extra_warnings) = time_use_payload.get("warnings").and_then(Value::as_array) {
            let mut warnings = payload["warnings"].as_array().cloned().unwrap_or_default();
            warnings.extend(extra_warnings.iter().cloned());
            if !warnings.is_empty() {
                payload["warnings"] = Value::Array(warnings);
            }
        }
    }
    Ok(payload)
}

fn motifs_section_payload(ctx: &ComparisonContext, filter_key: &str) -> anyhow::Result<Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let visits = prepared_visits_for_filter(ctx, &filter)?;
    if visits.synthetic.is_none() && visits.observed.is_none() {
        if !visits.warnings.is_empty() {
            payload["warnings"] = json!(visits.warnings);
        }
        return Ok(payload);
    }

    // `build_section_motifs` (payload/sections.py) never propagates the
    // jsd side effect into this route's response -- only `metrics_section_payload`
    // (mirroring `build_section_metrics`) surfaces it.
    let mut unused_jsd = Vec::new();
    let block = build_motifs_block(
        &ctx.observed_label,
        visits.observed.as_ref(),
        visits.synthetic.as_ref(),
        &filter,
        &mut unused_jsd,
    )?;
    payload["motifs"] = json!({"groups": [{
        "filter_key": filter.key,
        "filter_label": filter.label,
        "block": block,
    }]});
    if !visits.warnings.is_empty() {
        payload["warnings"] = json!(visits.warnings);
    }
    Ok(payload)
}

fn values_per_user(df: &DataFrame, uid_col: &str) -> anyhow::Result<Vec<f64>> {
    Ok(df
        .clone()
        .lazy()
        .group_by([col(uid_col)])
        .agg([len().alias("_count")])
        .collect()?
        .column("_count")?
        .cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .flatten()
        .collect())
}

fn numeric_column_values(
    df: &DataFrame,
    name: &str,
    predicate: impl Fn(f64) -> bool,
) -> anyhow::Result<Option<Vec<f64>>> {
    if df.column(name).is_err() {
        return Ok(None);
    }
    Ok(Some(
        df.column(name)?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .flatten()
            .filter(|v| predicate(*v))
            .collect(),
    ))
}

fn distributions_section_payload(
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<Value> {
    let filter = choose_filter(ctx, filter_key)?;
    let synthetic_traj = load_trajectory(&ctx.synthetic_path)?;
    let observed_traj = match &ctx.observed_path {
        Some(path) => Some(load_trajectory(path)?),
        None => None,
    };

    let synth_jumps_rog = crate::comparison::features::jumps_rog_for_filters(
        &synthetic_traj.df,
        &synthetic_traj.uid_col,
        &synthetic_traj.lat_col,
        &synthetic_traj.lng_col,
        &synthetic_traj.datetime_col,
        std::slice::from_ref(&filter),
        "synthetic",
        AdaptationMode::Auto,
        None,
        10,
    )?;
    let observed_jumps_rog = match &observed_traj {
        Some(obs) => Some(crate::comparison::features::jumps_rog_for_filters(
            &obs.df,
            &obs.uid_col,
            &obs.lat_col,
            &obs.lng_col,
            &obs.datetime_col,
            std::slice::from_ref(&filter),
            &ctx.observed_label,
            ctx.evaluation_mode,
            ctx.evaluation_location_col.as_deref(),
            ctx.evaluation_h3_resolution,
        )?),
        None => None,
    };

    let mut warnings = Vec::<String>::new();
    let synth_df = filter_df(
        &synthetic_traj.df,
        Some(&synthetic_traj.datetime_col),
        &filter,
    )?;
    let mut group = json!({
        "filter_key": filter.key,
        "filter_label": filter.label,
        "blocks": {},
    });
    if synth_df.height() == 0 {
        warnings.push(format!(
            "{} distribution filter has no synthetic rows",
            group["filter_label"].as_str().unwrap_or("Selected")
        ));
    } else {
        let real_group_df = match &observed_traj {
            Some(obs) => Some(filter_df(&obs.df, Some(&obs.datetime_col), &filter)?),
            None => None,
        };
        let real_metric_group_df = match (&observed_traj, &real_group_df) {
            (Some(obs), Some(df)) if df.height() > 0 => Some(
                adapt_evaluation_dataframe(
                    df,
                    &ctx.observed_label,
                    &obs.uid_col,
                    &obs.datetime_col,
                    &obs.lat_col,
                    &obs.lng_col,
                    ctx.evaluation_mode,
                    ctx.evaluation_location_col.as_deref(),
                    ctx.evaluation_h3_resolution,
                )?
                .df,
            ),
            _ => None,
        };

        let synth_jr = &synth_jumps_rog[filter_key];
        let observed_jr = observed_jumps_rog
            .as_ref()
            .and_then(|m| m.get(filter_key))
            .filter(|_| {
                real_metric_group_df
                    .as_ref()
                    .is_some_and(|df| df.height() > 0)
            });

        let synth_stays = collapse_to_stays(
            &synth_df,
            &synthetic_traj.uid_col,
            &synthetic_traj.lat_col,
            &synthetic_traj.lng_col,
            &synthetic_traj.datetime_col,
        )?;
        let synth_visits_count = values_per_user(&synth_stays, &synthetic_traj.uid_col)?;
        let real_visits_count = match (&observed_traj, &real_metric_group_df) {
            (Some(obs), Some(real_df)) => {
                let stays = collapse_to_stays(
                    real_df,
                    &obs.uid_col,
                    &obs.lat_col,
                    &obs.lng_col,
                    &obs.datetime_col,
                )?;
                Some(values_per_user(&stays, &obs.uid_col)?)
            }
            _ => None,
        };

        let synth_dwell = numeric_column_values(&synth_df, "dwell_minutes", |v| v >= 0.0)?
            .unwrap_or_else(|| {
                waiting_times_minutes(
                    &synth_df,
                    &synthetic_traj.uid_col,
                    &synthetic_traj.datetime_col,
                )
                .unwrap_or_default()
            });
        let observed_duration = observed_traj.as_ref().and_then(|t| duration_col(&t.df));
        let real_dwell = match (
            &observed_traj,
            &real_metric_group_df,
            observed_duration.as_deref(),
        ) {
            (_, Some(real_df), Some(c)) => {
                Some(numeric_column_values(real_df, c, |_| true)?.unwrap_or_default())
            }
            (Some(obs), Some(real_df), None) => Some(
                waiting_times_minutes(real_df, &obs.uid_col, &obs.datetime_col).unwrap_or_default(),
            ),
            _ => None,
        };

        let (synth_trip, real_trip): (Vec<f64>, Option<Vec<f64>>) = if let Some(trip) =
            numeric_column_values(&synth_df, "trip_duration_minutes", |v| v > 0.0)?
        {
            let real = observed_jr.map(|jr| {
                jr.jumps
                    .iter()
                    .filter(|&&j| j > 0.0)
                    .map(|&j| (j / CAR_SPEED_KMH) * 60.0)
                    .collect::<Vec<_>>()
            });
            (trip, real)
        } else if let (Some(real_df), Some(c)) =
            (real_metric_group_df.as_ref(), observed_duration.as_deref())
        {
            (
                waiting_times_minutes(
                    &synth_df,
                    &synthetic_traj.uid_col,
                    &synthetic_traj.datetime_col,
                )
                .unwrap_or_default(),
                Some(numeric_column_values(real_df, c, |_| true)?.unwrap_or_default()),
            )
        } else {
            (Vec::new(), None)
        };

        let mut blocks = serde_json::Map::new();
        blocks.insert(
            "jump_lengths".to_string(),
            serde_json::to_value(ecdf_block(
                "synthetic",
                &synth_jr.jumps,
                observed_jr.map(|jr| (ctx.observed_label.as_str(), jr.jumps.as_slice())),
                "jump length",
                "km",
            ))?,
        );
        blocks.insert(
            "visits_per_user".to_string(),
            serde_json::to_value(ecdf_block(
                "synthetic",
                &synth_visits_count,
                real_visits_count
                    .as_ref()
                    .map(|v| (ctx.observed_label.as_str(), v.as_slice())),
                "number of visits",
                "",
            ))?,
        );
        blocks.insert(
            "radius_of_gyration".to_string(),
            serde_json::to_value(ecdf_block(
                "synthetic",
                &synth_jr.rog,
                observed_jr.map(|jr| (ctx.observed_label.as_str(), jr.rog.as_slice())),
                "radius of gyration",
                "km",
            ))?,
        );
        blocks.insert(
            "dwell_time".to_string(),
            serde_json::to_value(ecdf_block(
                "synthetic",
                &synth_dwell,
                real_dwell
                    .as_ref()
                    .map(|v| (ctx.observed_label.as_str(), v.as_slice())),
                "dwell time",
                "min",
            ))?,
        );
        if !synth_trip.is_empty() && (ctx.mode() == "synthetic_only" || real_trip.is_some()) {
            blocks.insert(
                "trip_duration".to_string(),
                serde_json::to_value(ecdf_block(
                    "synthetic",
                    &synth_trip,
                    real_trip
                        .as_ref()
                        .map(|v| (ctx.observed_label.as_str(), v.as_slice())),
                    "trip duration",
                    "min",
                ))?,
            );
        }
        group["blocks"] = Value::Object(blocks);
    }

    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    payload["ecdf"] = json!({"groups": [group]});
    if !warnings.is_empty() {
        payload["warnings"] = json!(warnings);
    }
    Ok(payload)
}

fn transport_spatial_section_payload(ctx: &ComparisonContext) -> anyhow::Result<Value> {
    let mut payload = empty_chart_payload(ctx, Vec::new());
    let mut warnings = Vec::<String>::new();
    if !ctx.transport_enabled {
        return Ok(payload);
    }
    let moving_path = ctx
        .transport_synthetic_moving_path
        .clone()
        .unwrap_or_else(|| default_synthetic_moving_path(&ctx.synthetic_path));
    if !moving_path.exists() {
        payload["warnings"] = json!([format!(
            "transport_spatial: moving sidecar not found: {}",
            moving_path.display()
        )]);
        return Ok(payload);
    }

    let mut records = synthetic_transport_leg_records(&moving_path, &ctx.transport_mode_map)?;
    if records.height() == 0 {
        payload["warnings"] = json!(["transport_spatial: no synthetic transport legs"]);
        return Ok(payload);
    }

    if ctx.transport_observed_enabled {
        if let Some(observed_path) = &ctx.observed_path {
            let observed_traj = load_trajectory(observed_path)?;
            let duration = duration_col(&observed_traj.df);
            match observed_transport_leg_records(
                &observed_traj.df,
                ctx.transport_uid_col.as_deref(),
                ctx.transport_datetime_col.as_deref(),
                ctx.transport_lat_col.as_deref(),
                ctx.transport_lng_col.as_deref(),
                ctx.transport_col.as_deref(),
                duration.as_deref(),
                &ctx.transport_mode_map,
            ) {
                Ok(observed_records) if observed_records.height() > 0 => {
                    records.vstack_mut(&observed_records)?;
                }
                Ok(_) => warnings.push("transport_spatial: no observed transport legs".to_string()),
                Err(err) => warnings.push(format!("transport_spatial.observed: {err}")),
            }
        }
    }

    let summary = transport_spatial_summary(&records)?;
    let mut modes = BTreeSet::<String>::new();
    for source in summary.values() {
        for row in &source.modes {
            modes.insert(row.mode.clone());
        }
    }
    let mut mode_order: Vec<String> = modes.into_iter().collect();
    mode_order.sort_by_key(|m| {
        let order = crate::comparison::DEFAULT_MODE_ORDER
            .iter()
            .position(|d| d == m)
            .unwrap_or(99);
        (order, m.clone())
    });

    let mut share_series = Vec::new();
    for (source, label) in [
        ("synthetic", "synthetic"),
        ("observed", ctx.observed_label.as_str()),
    ] {
        let Some(source_summary) = summary.get(source) else {
            continue;
        };
        let by_mode: HashMap<&str, f64> = source_summary
            .modes
            .iter()
            .map(|row| (row.mode.as_str(), row.percent))
            .collect();
        share_series.push(json!({
            "name": label,
            "role": source,
            "values": mode_order.iter().map(|mode| by_mode.get(mode.as_str()).copied().unwrap_or(0.0)).collect::<Vec<_>>(),
        }));
    }

    payload["transport_spatial"] = json!({
        "summary": summary,
        "share": {
            "categories": mode_order,
            "series": share_series,
        },
        "jump_ecdf": transport_ecdf_block(&records, &ctx.observed_label)?,
    });
    if !warnings.is_empty() {
        payload["warnings"] = json!(warnings);
    }
    Ok(payload)
}

fn micro_activity_section_payload(
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let Some(path) = &ctx.synthetic_activities_path else {
        return Ok(payload);
    };
    if !path.exists() {
        return Ok(payload);
    }
    let activities = read_parquet(path)?;
    if activities.height() == 0 {
        return Ok(payload);
    }
    let dt_col = micro_activity_datetime_col(&activities);
    let filtered = filter_df(&activities, dt_col.as_deref(), &filter)?;
    if filtered.height() == 0 {
        return Ok(payload);
    }
    let block = micro_activity_daily_usage_data(&filtered, 10)?;
    payload["micro_activity_usage"] = json!({
        "groups": [{
            "filter_key": filter.key,
            "filter_label": filter.label,
            "block": block,
        }],
    });
    Ok(payload)
}

fn string_column(df: &DataFrame, name: &str) -> anyhow::Result<Vec<String>> {
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

fn matrix_limit(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .flatten()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1.0)
}

fn align_square(categories: &[String], matrix: &[Vec<f64>], target: &[String]) -> Vec<Vec<f64>> {
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

fn align_daily(categories: &[String], matrix: &[Vec<f64>], target: &[String]) -> Vec<Vec<f64>> {
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

type DailyTuple = (Vec<String>, Vec<Vec<f64>>);

fn daily_tuple(visits: &DataFrame) -> anyhow::Result<DailyTuple> {
    let purpose = string_column(visits, "purpose")?;
    let start_minutes = datetime_minutes(visits, "start_timestamp")?;
    let end_minutes = datetime_minutes(visits, "end_timestamp")?;
    let valid_rows = vec![true; visits.height()];
    daily_activity_distribution(&purpose, &start_minutes, &end_minutes, &valid_rows, 60)
}

fn ordered_union(left: &[String], right: &[String]) -> Vec<String> {
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

fn activity_section_payload(ctx: &ComparisonContext, filter_key: &str) -> anyhow::Result<Value> {
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

fn finite_xy(x: &[f64], y: &[f64]) -> Vec<[f64; 2]> {
    x.iter()
        .zip(y.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| [x, y])
        .collect()
}

fn curve_x(datasets: &[Vec<f64>], logarithmic: bool) -> Vec<f64> {
    let values: Vec<f64> = datasets
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if values.is_empty() {
        return Vec::new();
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min >= max {
        return vec![min];
    }
    let n = 100usize;
    if logarithmic {
        let lo = min.log10();
        let hi = max.log10();
        (0..n)
            .map(|i| 10f64.powf(lo + (hi - lo) * (i as f64) / ((n - 1) as f64)))
            .collect()
    } else {
        (0..n)
            .map(|i| min + (max - min) * (i as f64) / ((n - 1) as f64))
            .collect()
    }
}

fn geometric_scale(observed: &[f64], shape: &[f64]) -> f64 {
    let logs: Vec<f64> = observed
        .iter()
        .zip(shape.iter())
        .filter(|(o, s)| o.is_finite() && s.is_finite() && **o > 0.0 && **s > 0.0)
        .map(|(o, s)| (o / s).ln())
        .collect();
    if logs.is_empty() {
        1.0
    } else {
        (logs.iter().sum::<f64>() / logs.len() as f64).exp()
    }
}

fn truncated_powerlaw_series(
    observed_values: Option<&[f64]>,
    synthetic_values: &[f64],
    observed_label: Option<&str>,
    reference: (f64, f64, f64),
) -> anyhow::Result<Value> {
    let syn = truncated_powerlaw_dataset(synthetic_values, "synthetic")?;
    let mut datasets: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, String, &'static str)> =
        vec![(syn.0, syn.1, syn.2, syn.3, "synthetic")];
    if let (Some(values), Some(label)) = (observed_values, observed_label) {
        if let Ok(obs) = truncated_powerlaw_dataset(values, label) {
            datasets.insert(0, (obs.0, obs.1, obs.2, obs.3, "observed"));
        }
    }
    let all_x: Vec<Vec<f64>> = datasets.iter().map(|(_, x, _, _, _)| x.clone()).collect();
    let cx = curve_x(&all_x, true);
    let mut series = Vec::new();
    let mut fits = Vec::new();
    for (params, x, y, label, role) in &datasets {
        let (c, r0, beta, kappa) = (params[0], params[1], params[2], params[3]);
        let fit_y: Vec<f64> = cx
            .iter()
            .map(|x| c * (x + r0).powf(-beta) * (-x / kappa).exp())
            .collect();
        series.push(
            json!({"name": label, "role": role, "type": "scatter", "points": finite_xy(x, y)}),
        );
        series.push(json!({"name": format!("{label} fit"), "role": role, "type": "line", "points": finite_xy(&cx, &fit_y)}));
        fits.push(
            json!({"label": label, "params": {"c": c, "r0": r0, "beta": beta, "kappa": kappa}}),
        );
    }
    let (r0, beta, kappa) = reference;
    let joined_x: Vec<f64> = datasets
        .iter()
        .flat_map(|(_, x, _, _, _)| x.clone())
        .collect();
    let joined_y: Vec<f64> = datasets
        .iter()
        .flat_map(|(_, _, y, _, _)| y.clone())
        .collect();
    let shape: Vec<f64> = joined_x
        .iter()
        .map(|x| (x + r0).powf(-beta) * (-x / kappa).exp())
        .collect();
    let c = geometric_scale(&joined_y, &shape);
    let ref_y: Vec<f64> = cx
        .iter()
        .map(|x| c * (x + r0).powf(-beta) * (-x / kappa).exp())
        .collect();
    series.push(json!({"name": "Gonzalez reference", "role": "reference", "type": "line", "points": finite_xy(&cx, &ref_y)}));
    Ok(json!({
        "x_log": true,
        "formula": "p(x) = c (x + r0)^-beta exp(-x / kappa)",
        "series": series,
        "fits": fits,
    }))
}

fn lognormal_series(
    observed_visits: Option<&DataFrame>,
    synthetic_visits: &DataFrame,
    observed_label: Option<&str>,
) -> anyhow::Result<Value> {
    let syn = daily_location_lognormal_dataset(synthetic_visits, "synthetic")?;
    let mut datasets: Vec<(Vec<f64>, Vec<f64>, f64, f64, String, &'static str)> =
        vec![(syn.0, syn.1, syn.2, syn.3, syn.4, "synthetic")];
    if let (Some(visits), Some(label)) = (observed_visits, observed_label) {
        if let Ok(obs) = daily_location_lognormal_dataset(visits, label) {
            datasets.insert(0, (obs.0, obs.1, obs.2, obs.3, obs.4, "observed"));
        }
    }
    let all_x: Vec<Vec<f64>> = datasets
        .iter()
        .map(|(x, _, _, _, _, _)| x.clone())
        .collect();
    let cx = curve_x(&all_x, false);
    let mut series = Vec::new();
    let mut fits = Vec::new();
    for (x, y, mu, sigma, label, role) in &datasets {
        let fit_y: Vec<f64> = cx
            .iter()
            .map(|x| {
                (-((x.ln() - mu).powi(2)) / (2.0 * sigma.powi(2))).exp()
                    / (x * sigma * (2.0 * std::f64::consts::PI).sqrt())
            })
            .collect();
        series.push(
            json!({"name": label, "role": role, "type": "scatter", "points": finite_xy(x, y)}),
        );
        series.push(json!({"name": format!("{label} fit"), "role": role, "type": "line", "points": finite_xy(&cx, &fit_y)}));
        fits.push(json!({"label": label, "params": {"mu": mu, "sigma": sigma}}));
    }
    let ref_y: Vec<f64> = cx
        .iter()
        .map(|x| {
            (-(x.ln() - 1.0).powi(2) / (2.0 * 0.5f64.powi(2))).exp()
                / (x * 0.5 * (2.0 * std::f64::consts::PI).sqrt())
        })
        .collect();
    series.push(json!({"name": "Log-normal reference", "role": "reference", "type": "line", "points": finite_xy(&cx, &ref_y)}));
    Ok(json!({
        "x_log": false,
        "formula": "f(N) = exp(-(ln N - mu)^2 / (2 sigma^2)) / (N sigma sqrt(2 pi))",
        "series": series,
        "fits": fits,
    }))
}

fn distance_frequency_series(
    observed_visits: Option<&DataFrame>,
    synthetic_visits: &DataFrame,
    observed_label: Option<&str>,
) -> anyhow::Result<Value> {
    let syn = distance_frequency_dataset(synthetic_visits, "synthetic")?;
    let mut datasets: Vec<(Vec<f64>, Vec<f64>, f64, f64, String, &'static str)> =
        vec![(syn.0, syn.1, syn.2, syn.3, syn.4, "synthetic")];
    if let (Some(visits), Some(label)) = (observed_visits, observed_label) {
        if let Ok(obs) = distance_frequency_dataset(visits, label) {
            datasets.insert(0, (obs.0, obs.1, obs.2, obs.3, obs.4, "observed"));
        }
    }
    let all_x: Vec<Vec<f64>> = datasets
        .iter()
        .map(|(x, _, _, _, _, _)| x.clone())
        .collect();
    let cx = curve_x(&all_x, true);
    let mut series = Vec::new();
    let mut fits = Vec::new();
    for (rf, rho, eta, mu, label, role) in &datasets {
        let fit_y: Vec<f64> = cx.iter().map(|x| mu * x.powf(-eta)).collect();
        series.push(
            json!({"name": label, "role": role, "type": "scatter", "points": finite_xy(rf, rho)}),
        );
        series.push(json!({"name": format!("{label} fit"), "role": role, "type": "line", "points": finite_xy(&cx, &fit_y)}));
        fits.push(json!({"label": label, "params": {"eta": eta, "mu": mu}}));
    }
    let joined_x: Vec<f64> = datasets
        .iter()
        .flat_map(|(x, _, _, _, _, _)| x.clone())
        .collect();
    let joined_y: Vec<f64> = datasets
        .iter()
        .flat_map(|(_, y, _, _, _, _)| y.clone())
        .collect();
    let shape: Vec<f64> = joined_x.iter().map(|x| x.powf(-2.0)).collect();
    let scale = geometric_scale(&joined_y, &shape);
    let ref_y: Vec<f64> = cx.iter().map(|x| scale * x.powf(-2.0)).collect();
    series.push(json!({"name": "Schlapfer reference", "role": "reference", "type": "line", "points": finite_xy(&cx, &ref_y)}));
    Ok(json!({
        "x_log": true,
        "formula": "rho(r, f) = mu (r f)^-eta",
        "series": series,
        "fits": fits,
    }))
}

fn law_block_with_meta(mut block: Value, title: &str, x_label: &str, x_unit: &str) -> Value {
    if let Some(obj) = block.as_object_mut() {
        obj.insert("title".to_string(), json!(title));
        obj.insert("x_label".to_string(), json!(x_label));
        obj.insert("x_unit".to_string(), json!(x_unit));
    }
    block
}

fn mobility_laws_section_payload(
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let mut warnings = Vec::new();

    let synthetic_traj = load_trajectory(&ctx.synthetic_path)?;
    let synth_df = filter_df(
        &synthetic_traj.df,
        Some(&synthetic_traj.datetime_col),
        &filter,
    )?;
    if synth_df.height() == 0 {
        return Ok(payload);
    }
    let observed_traj = match &ctx.observed_path {
        Some(path) => Some(load_trajectory(path)?),
        None => None,
    };
    let real_df = match &observed_traj {
        Some(obs) => {
            let filtered = filter_df(&obs.df, Some(&obs.datetime_col), &filter)?;
            if filtered.height() > 0 {
                Some(
                    adapt_evaluation_dataframe(
                        &filtered,
                        &ctx.observed_label,
                        &obs.uid_col,
                        &obs.datetime_col,
                        &obs.lat_col,
                        &obs.lng_col,
                        ctx.evaluation_mode,
                        ctx.evaluation_location_col.as_deref(),
                        ctx.evaluation_h3_resolution,
                    )?
                    .df,
                )
            } else {
                None
            }
        }
        None => None,
    };

    let synth_jr = crate::comparison::features::jumps_rog_for_filters(
        &synthetic_traj.df,
        &synthetic_traj.uid_col,
        &synthetic_traj.lat_col,
        &synthetic_traj.lng_col,
        &synthetic_traj.datetime_col,
        std::slice::from_ref(&filter),
        "synthetic",
        AdaptationMode::Auto,
        None,
        10,
    )?;
    let real_jr = match &observed_traj {
        Some(obs) => Some(crate::comparison::features::jumps_rog_for_filters(
            &obs.df,
            &obs.uid_col,
            &obs.lat_col,
            &obs.lng_col,
            &obs.datetime_col,
            std::slice::from_ref(&filter),
            &ctx.observed_label,
            ctx.evaluation_mode,
            ctx.evaluation_location_col.as_deref(),
            ctx.evaluation_h3_resolution,
        )?),
        None => None,
    };
    let synth_cols: Vec<&str> = synthetic_traj
        .df
        .get_column_names()
        .iter()
        .map(|s| s.as_str())
        .collect();
    let synth_activity_col = detect_in(&synth_cols, ACTIVITY_CANDIDATES);
    let synth_location_col = detect_in(&synth_cols, LOCATION_CANDIDATES);
    let syn_visits = mobility_law_visits(
        &synth_df,
        &synthetic_traj.uid_col,
        &synthetic_traj.datetime_col,
        &synthetic_traj.lat_col,
        &synthetic_traj.lng_col,
        synth_location_col.as_deref(),
        synth_activity_col.as_deref(),
        ctx.evaluation_h3_resolution,
    )?;
    let obs_visits = match (&observed_traj, &real_df) {
        (Some(obs), Some(real_df)) => {
            let cols: Vec<&str> = real_df
                .get_column_names()
                .iter()
                .map(|s| s.as_str())
                .collect();
            let activity_col = detect_in(&cols, ACTIVITY_CANDIDATES);
            let location_col = detect_in(&cols, LOCATION_CANDIDATES);
            Some(mobility_law_visits(
                real_df,
                &obs.uid_col,
                &obs.datetime_col,
                &obs.lat_col,
                &obs.lng_col,
                location_col.as_deref(),
                activity_col.as_deref(),
                ctx.evaluation_h3_resolution,
            )?)
        }
        _ => None,
    };

    let syn_jr = &synth_jr[&filter.key];
    let obs_jr = real_jr.as_ref().and_then(|m| m.get(&filter.key));
    let mut blocks = serde_json::Map::new();
    for (name, value) in [
        (
            "travel_distance",
            truncated_powerlaw_series(
                obs_jr.map(|jr| jr.jumps.as_slice()),
                &syn_jr.jumps,
                obs_jr.map(|_| ctx.observed_label.as_str()),
                (1.5, 1.75, 400.0),
            )
            .map(|v| {
                law_block_with_meta(v, "Travel-distance mobility law", "travel distance", "km")
            }),
        ),
        (
            "radius_of_gyration",
            truncated_powerlaw_series(
                obs_jr.map(|jr| jr.rog.as_slice()),
                &syn_jr.rog,
                obs_jr.map(|_| ctx.observed_label.as_str()),
                (5.8, 1.65, 350.0),
            )
            .map(|v| {
                law_block_with_meta(
                    v,
                    "Radius-of-gyration mobility law",
                    "radius of gyration",
                    "km",
                )
            }),
        ),
        (
            "daily_locations",
            lognormal_series(
                obs_visits.as_ref(),
                &syn_visits,
                obs_visits.as_ref().map(|_| ctx.observed_label.as_str()),
            )
            .map(|v| {
                law_block_with_meta(v, "Daily visited locations", "number of locations (N)", "")
            }),
        ),
        (
            "distance_frequency",
            distance_frequency_series(
                obs_visits.as_ref(),
                &syn_visits,
                obs_visits.as_ref().map(|_| ctx.observed_label.as_str()),
            )
            .map(|v| law_block_with_meta(v, "Distance-frequency visitation law", "r · f", "km")),
        ),
    ] {
        match value {
            Ok(v) => {
                blocks.insert(name.to_string(), v);
            }
            Err(err) => warnings.push(format!("mobility_laws.{}.{}: {err}", filter.key, name)),
        }
    }
    if !blocks.is_empty() {
        payload["mobility_laws"] = json!({"groups": [{
            "filter_key": filter.key,
            "filter_label": filter.label,
            "blocks": blocks,
        }]});
    }
    if !warnings.is_empty() {
        payload["warnings"] = json!(warnings);
    }
    Ok(payload)
}

fn classify_stvd(volume_diff: f64, peak_shift: f64, threshold: f64) -> (usize, usize) {
    let x_bin = if volume_diff < -threshold {
        0
    } else if volume_diff <= threshold {
        1
    } else {
        2
    };
    let y_bin = if peak_shift <= 2.0 {
        0
    } else if peak_shift <= 5.0 {
        1
    } else {
        2
    };
    (x_bin, y_bin)
}

fn stvd_section_payload(ctx: &ComparisonContext, filter_key: &str) -> anyhow::Result<Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let Some(observed_path) = &ctx.observed_path else {
        return Ok(payload);
    };
    let synthetic = load_trajectory(&ctx.synthetic_path)?;
    let observed = load_trajectory(observed_path)?;
    let syn_df = filter_df(&synthetic.df, Some(&synthetic.datetime_col), &filter)?;
    let obs_df = filter_df(&observed.df, Some(&observed.datetime_col), &filter)?;
    if syn_df.height() == 0 || obs_df.height() == 0 {
        return Ok(payload);
    }
    let layers = compute_stvd_layers(
        &syn_df,
        &synthetic.lat_col,
        &synthetic.lng_col,
        &synthetic.datetime_col,
        &obs_df,
        &observed.lat_col,
        &observed.lng_col,
        &observed.datetime_col,
        &[7, 9],
    )?;
    const COLORS: [[&str; 3]; 3] = [
        ["#2c7bb6", "#abd9e9", "#ffffbf"],
        ["#74add1", "#f7f7f7", "#fdae61"],
        ["#ffffbf", "#fdae61", "#d7191c"],
    ];
    let threshold = 25.0;
    let mut out_layers = serde_json::Map::new();
    let mut lngs = Vec::new();
    let mut lats = Vec::new();
    for (res, features) in layers {
        let mut geo_features = Vec::new();
        for feature in features {
            let (x_bin, y_bin) =
                classify_stvd(feature.volume_diff_pct, feature.peak_shift_hours, threshold);
            for [lng, lat] in &feature.ring {
                lngs.push(*lng);
                lats.push(*lat);
            }
            geo_features.push(json!({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [feature.ring]},
                "properties": {
                    "area": feature.cell_hex,
                    "volume_diff_pct": feature.volume_diff_pct,
                    "peak_shift_hours": feature.peak_shift_hours,
                    "color": COLORS[y_bin][x_bin],
                    "class": y_bin * 3 + x_bin,
                },
            }));
        }
        out_layers.insert(
            res.to_string(),
            json!({"type": "FeatureCollection", "features": geo_features}),
        );
    }
    let center = if lngs.is_empty() {
        Value::Null
    } else {
        json!([
            (lngs.iter().copied().fold(f64::INFINITY, f64::min)
                + lngs.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                / 2.0,
            (lats.iter().copied().fold(f64::INFINITY, f64::min)
                + lats.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                / 2.0
        ])
    };
    payload["stvd"] = json!({"groups": [{
        "filter_key": filter.key,
        "filter_label": filter.label,
        "block": {
            "center": center,
            "layers": out_layers,
            "colors": COLORS,
            "threshold": threshold,
        },
    }]});
    Ok(payload)
}

fn social_network_section_payload(ctx: &ComparisonContext) -> anyhow::Result<Value> {
    let mut payload = empty_chart_payload(ctx, Vec::new());
    let path = {
        let stem = ctx
            .synthetic_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        ctx.synthetic_path
            .with_file_name(format!("{stem}_social_network.json"))
    };
    if !path.exists() {
        return Ok(payload);
    }
    let mut data: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let nodes_len = data
        .get("nodes")
        .and_then(Value::as_array)
        .map_or(0usize, Vec::len);
    let edges_len = data
        .get("edges")
        .and_then(Value::as_array)
        .map_or(0usize, Vec::len);
    if data
        .get("node_count")
        .and_then(Value::as_u64)
        .unwrap_or(nodes_len as u64)
        != nodes_len as u64
        || data
            .get("edge_count")
            .and_then(Value::as_u64)
            .unwrap_or(edges_len as u64)
            != edges_len as u64
    {
        anyhow::bail!("social network sidecar count mismatch: {}", path.display());
    }
    const MAX_AGENTS: usize = 5000;
    if nodes_len > MAX_AGENTS {
        if let Some(obj) = data.as_object_mut() {
            let visible = MAX_AGENTS;
            if let Some(nodes) = obj.get_mut("nodes").and_then(Value::as_array_mut) {
                nodes.truncate(visible);
            }
            if let Some(edges) = obj.get_mut("edges").and_then(Value::as_array_mut) {
                edges.retain(|row| {
                    row.as_array().is_some_and(|r| {
                        r.len() >= 2
                            && r[0].as_u64().unwrap_or(u64::MAX) < visible as u64
                            && r[1].as_u64().unwrap_or(u64::MAX) < visible as u64
                    })
                });
            }
            if let Some(degrees) = obj.get_mut("degrees").and_then(Value::as_array_mut) {
                degrees.truncate(visible);
            }
            obj.insert("nodes_sampled".to_string(), json!(true));
            obj.insert("edges_sampled".to_string(), json!(true));
        }
    }
    payload["social_network"] = data;
    Ok(payload)
}

const PROFILE_METRICS: &[&str] = &["regularity", "diversity", "stationarity", "entropy"];
const PROFILE_ORDER: &[&str] = &["Scouter", "Regular", "Routiner"];
const MAX_SCATTER_POINTS: usize = 5000;

#[derive(Debug, Clone)]
struct ProfileVisit {
    uid: i64,
    start_us: i64,
    end_us: i64,
    purpose: String,
    location_id: String,
}

#[derive(Debug, Clone)]
struct ProfileRow {
    uid: i64,
    intermittency: f64,
    degree_of_return: f64,
    regularity: f64,
    diversity: f64,
    stationarity: f64,
    entropy: f64,
    agent_type: String,
}

fn profile_visits_from_df(df: &DataFrame) -> anyhow::Result<Vec<ProfileVisit>> {
    let uid = df.column("uid")?.cast(&DataType::Int64)?;
    let uid = uid.i64()?;
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
        let (Some(uid), Some(start_us), Some(end_us)) =
            (uid.get(i), start.phys.get(i), end.phys.get(i))
        else {
            continue;
        };
        if end_us <= start_us {
            continue;
        }
        rows.push(ProfileVisit {
            uid,
            start_us,
            end_us,
            purpose: purpose.get(i).unwrap_or("").to_string(),
            location_id: location.get(i).unwrap_or("").to_string(),
        });
    }
    rows.sort_by_key(|r| (r.uid, r.start_us));
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
    let mut counts = HashMap::<&str, usize>::new();
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
    let mut by_uid = BTreeMap::<i64, Vec<ProfileVisit>>::new();
    for visit in visits {
        by_uid.entry(visit.uid).or_default().push(visit);
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
        let mut dwell = 0.0;
        let mut min_start = i64::MAX;
        let mut max_end = i64::MIN;
        for visit in &rows_for_user {
            distinct_pairs.insert((visit.location_id.as_str(), visit.purpose.as_str()));
            dwell += (visit.end_us - visit.start_us) as f64 / 60_000_000.0;
            min_start = min_start.min(visit.start_us);
            max_end = max_end.max(visit.end_us);
        }
        let total = rows_for_user.len().max(1) as f64;
        let regularity = 1.0 - distinct_pairs.len() as f64 / total;
        let diversity = distinct_substring_diversity(&tokens);
        let entropy = fkmob_core::measures::individual::entropy::trajectory_entropy_batch(
            tokens,
            vec![(0, rows_for_user.len())],
            true,
        )
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .next()
        .unwrap_or(0.0);
        let span = (max_end - min_start) as f64 / 60_000_000.0;
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

fn profiles_section_payload(ctx: &ComparisonContext) -> anyhow::Result<Value> {
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

const TIME_USE_CATEGORIES: &[&str] = &[
    "sleep",
    "personal care",
    "household",
    "work",
    "study",
    "shopping",
    "leisure",
    "travel",
    "other",
];

fn sql_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn resolve_time_use_path(path: &Path) -> Option<PathBuf> {
    if path.exists()
        && !path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("dta"))
    {
        return Some(path.to_path_buf());
    }
    if path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("dta"))
    {
        let parquet = path.with_extension("parquet");
        if parquet.exists() {
            return Some(parquet);
        }
        let csv = path.with_extension("csv");
        if csv.exists() {
            return Some(csv);
        }
    }
    None
}

fn duckdb_scan_expr(path: &Path) -> anyhow::Result<String> {
    let quoted = quote_path(path);
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("parquet") => Ok(format!("read_parquet('{quoted}')")),
        Some("csv") => Ok(format!("read_csv_auto('{quoted}')")),
        other => anyhow::bail!("unsupported time-use table format: {other:?}"),
    }
}

fn time_use_day_group_from_label(value: &str) -> Option<&'static str> {
    let raw = value.trim().to_ascii_lowercase();
    match raw.as_str() {
        "saturday" | "sat" | "6" | "sunday" | "sun" | "7" => Some("weekend"),
        "monday" | "mon" | "1" | "tuesday" | "tue" | "2" | "wednesday" | "wed" | "3"
        | "thursday" | "thu" | "4" | "friday" | "fri" | "5" => Some("weekday"),
        _ => None,
    }
}

fn observed_time_use_minutes(
    path: &Path,
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<HashMap<String, f64>> {
    let scan = duckdb_scan_expr(path)?;
    let conn = duckdb::Connection::open_in_memory()?;
    let columns: HashSet<String> = conn
        .prepare(&format!("DESCRIBE SELECT * FROM {scan}"))?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<HashSet<_>, _>>()?;
    let mut required: Vec<&str> = TIME_USE_CATEGORIES.to_vec();
    required.push("day");
    required.push(ctx.time_use_weight_col.as_str());
    for col_name in required {
        if !columns.contains(col_name) {
            anyhow::bail!("observed time-use table missing required column {col_name:?}");
        }
    }
    let mut where_clauses = Vec::<String>::new();
    if let Some(country) = &ctx.time_use_country {
        if columns.contains("country") {
            where_clauses.push(format!(
                "CAST(country AS VARCHAR) = {}",
                sql_literal(country)
            ));
        }
    }
    if let Some(survey) = ctx.time_use_survey {
        if columns.contains("survey") {
            where_clauses.push(format!("TRY_CAST(survey AS BIGINT) = {survey}"));
        }
    }
    let where_sql = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };
    let mut select_cols = vec![
        "CAST(day AS VARCHAR) AS day".to_string(),
        format!(
            "TRY_CAST(\"{}\" AS DOUBLE) AS weight",
            ctx.time_use_weight_col.replace('"', "\"\"")
        ),
    ];
    for category in TIME_USE_CATEGORIES {
        select_cols.push(format!(
            "TRY_CAST(\"{}\" AS DOUBLE) AS \"{}\"",
            category.replace('"', "\"\""),
            category.replace('"', "\"\"")
        ));
    }
    let sql = format!("SELECT {} FROM {scan} {where_sql}", select_cols.join(", "));
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut weighted = HashMap::<String, f64>::new();
    let mut total_weight = 0.0;
    while let Some(row) = rows.next()? {
        let day = row.get::<_, Option<String>>(0)?.unwrap_or_default();
        let Some(day_group) = time_use_day_group_from_label(&day) else {
            continue;
        };
        if filter_key != "all" && filter_key != day_group {
            continue;
        }
        let weight = row.get::<_, Option<f64>>(1)?.unwrap_or(0.0);
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        total_weight += weight;
        for (idx, category) in TIME_USE_CATEGORIES.iter().enumerate() {
            let minutes = row.get::<_, Option<f64>>(idx + 2)?.unwrap_or(0.0);
            *weighted.entry((*category).to_string()).or_insert(0.0) += weight * minutes;
        }
    }
    if total_weight <= 0.0 {
        anyhow::bail!("observed time-use table has no positive weights after filters");
    }
    for value in weighted.values_mut() {
        *value /= total_weight;
    }
    Ok(weighted)
}

fn add_synthetic_time_use_segment(
    totals: &mut HashMap<String, f64>,
    agent_days: &mut BTreeSet<(i64, chrono::NaiveDate)>,
    uid: i64,
    category: &str,
    start_us: i64,
    end_us: i64,
    filter_key: &str,
) {
    let (Some(mut start), Some(end)) = (
        NaiveDateTime::from_timestamp_micros(start_us),
        NaiveDateTime::from_timestamp_micros(end_us),
    ) else {
        return;
    };
    while start < end {
        let date = start.date();
        let Some(next_midnight) = date.succ_opt().and_then(|d| d.and_hms_opt(0, 0, 0)) else {
            break;
        };
        let segment_end = end.min(next_midnight);
        let day_group = if date.weekday().number_from_monday() >= 6 {
            "weekend"
        } else {
            "weekday"
        };
        if filter_key == "all" || filter_key == day_group {
            let minutes = (segment_end - start).num_seconds() as f64 / 60.0;
            if minutes > 0.0 {
                *totals.entry(category.to_string()).or_insert(0.0) += minutes;
                agent_days.insert((uid, date));
            }
        }
        start = segment_end;
    }
}

fn time_use_metric_rows(time_use_comparison: &Value) -> Value {
    let rows: Vec<Value> = time_use_comparison
        .get("groups")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|group| {
            let values: Vec<f64> = group
                .get("block")
                .and_then(|v| v.get("rows"))
                .and_then(Value::as_array)?
                .iter()
                .filter_map(|row| {
                    row.get("share_of_day_difference_pct_points")
                        .and_then(Value::as_f64)
                        .map(f64::abs)
                })
                .collect();
            if values.is_empty() {
                return None;
            }
            Some(json!({
                "filter_key": group.get("filter_key").cloned().unwrap_or(json!("all")),
                "filter_label": group.get("filter_label").cloned().unwrap_or(json!("All")),
                "metric": "Mean absolute time-use share difference",
                "value": values.iter().sum::<f64>() / values.len() as f64,
                "unit": "percentage points",
            }))
        })
        .collect();
    Value::Array(rows)
}

fn time_use_section_payload(ctx: &ComparisonContext, filter_key: &str) -> anyhow::Result<Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let (Some(time_use_path), Some(activities_path)) =
        (&ctx.time_use_path, &ctx.synthetic_activities_path)
    else {
        return Ok(payload);
    };
    if !time_use_path.exists() || !activities_path.exists() {
        return Ok(payload);
    }
    let Some(observed_path) = resolve_time_use_path(time_use_path) else {
        payload["warnings"] = json!([format!(
            "time_use_comparison: .dta time-use file has no same-stem CSV/Parquet conversion: {}",
            time_use_path.display()
        )]);
        return Ok(payload);
    };
    let observed_minutes = match observed_time_use_minutes(&observed_path, ctx, &filter.key) {
        Ok(value) => value,
        Err(err) => {
            payload["warnings"] = json!([format!("time_use_comparison: {err}")]);
            return Ok(payload);
        }
    };

    let activities = read_parquet(activities_path)?;
    let required = ["uid", "activity", "arrival", "departure"];
    if required.iter().any(|c| activities.column(c).is_err()) {
        payload["warnings"] =
            json!(["time_use_comparison: activities table missing required columns"]);
        return Ok(payload);
    }
    let uid = activities.column("uid")?.i64()?;
    let activity = activities.column("activity")?.cast(&DataType::Int64)?;
    let activity = activity.i64()?;
    let arrival = activities
        .column("arrival")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let departure = activities
        .column("departure")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let arr = arrival.datetime()?;
    let dep = departure.datetime()?;
    let mut totals: HashMap<String, f64> = HashMap::new();
    let mut agent_days = std::collections::BTreeSet::<(i64, chrono::NaiveDate)>::new();
    for i in 0..activities.height() {
        let (Some(uid), Some(act), Some(a), Some(d)) = (
            uid.get(i),
            activity.get(i),
            arr.phys.get(i),
            dep.phys.get(i),
        ) else {
            continue;
        };
        if d <= a {
            continue;
        }
        let Some(def) = crate::settings::catalog::by_id(act) else {
            continue;
        };
        let category = def.name;
        if !TIME_USE_CATEGORIES.contains(&category) {
            continue;
        }
        add_synthetic_time_use_segment(
            &mut totals,
            &mut agent_days,
            uid,
            category,
            a,
            d,
            &filter.key,
        );
    }
    let denom = (agent_days.len() as f64).max(1.0);
    let rows: Vec<Value> = TIME_USE_CATEGORIES
        .iter()
        .map(|category| {
            let syn = totals.get(*category).copied().unwrap_or(0.0) / denom;
            let obs = observed_minutes.get(*category).copied().unwrap_or(0.0);
            let diff = syn - obs;
            let pct = if obs.abs() > 1e-12 {
                Value::from(diff / obs * 100.0)
            } else {
                Value::Null
            };
            json!({
                "category": category,
                "mtus_minutes": (obs * 1_000_000.0).round() / 1_000_000.0,
                "simulation_minutes": (syn * 1_000_000.0).round() / 1_000_000.0,
                "observed_minutes": (obs * 1_000_000.0).round() / 1_000_000.0,
                "synthetic_minutes": (syn * 1_000_000.0).round() / 1_000_000.0,
                "difference_minutes": (diff * 1_000_000.0).round() / 1_000_000.0,
                "percent_difference": pct,
                "share_of_day_difference_pct_points": ((diff / 1440.0 * 100.0) * 1_000_000.0).round() / 1_000_000.0,
            })
        })
        .collect();
    payload["time_use_comparison"] = json!({"groups": [{
        "filter_key": filter.key,
        "filter_label": filter.label,
        "block": {
            "categories": TIME_USE_CATEGORIES,
            "labels": [ctx.time_use_label, "synthetic"],
            "rows": rows,
        },
    }]});
    Ok(payload)
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
    let metrics =
        citybehavex_core::network_graph::compute_graph_metrics(node_count, edge_from, edge_to);
    let degrees: Vec<f64> = (0..node_count)
        .map(|idx| {
            edge_from.iter().filter(|&&u| u as usize == idx).count()
                + edge_to.iter().filter(|&&v| v as usize == idx).count()
        })
        .map(|v| v as f64)
        .collect();
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

pub fn network_validation_payload(
    enabled: bool,
    social_path: Option<&std::path::Path>,
    seed: i64,
) -> Value {
    if !enabled {
        return json!({"network_validation": null, "warnings": []});
    }
    let Some(path) = social_path.filter(|p| p.exists()) else {
        return json!({"network_validation": null, "warnings": ["synthetic_vs_random: social network sidecar not found"]});
    };
    let data: Value = match std::fs::read(path)
        .map_err(anyhow::Error::from)
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(anyhow::Error::from))
    {
        Ok(data) => data,
        Err(err) => {
            return json!({"network_validation": null, "warnings": [format!("network_validation: {err}")]});
        }
    };
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
    let source = network_block_from_edges(
        node_count,
        &edge_from,
        &edge_to,
        Some(&data),
        "synthetic_social",
    );
    let source_metrics = source
        .get("_metric_cache")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let degrees: Vec<f64> = source_metrics
        .get("degree")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_f64)
        .collect();
    let (rand_from, rand_to) = if node_count <= 5000 {
        degree_preserving_random_edges(&degrees, seed)
    } else {
        (Vec::new(), Vec::new())
    };
    let random = network_block_from_edges(
        node_count,
        &rand_from,
        &rand_to,
        Some(&data),
        "degree_preserving_rnd",
    );
    let random_metrics = random
        .get("_metric_cache")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let metric_names = [
        "degree",
        "clustering_coefficient",
        "edge_persistence",
        "topological_overlap",
    ];
    let mut wasserstein = serde_json::Map::new();
    let mut source_dist = serde_json::Map::new();
    let mut random_dist = serde_json::Map::new();
    for name in metric_names {
        let left: Vec<f64> = source_metrics
            .get(name)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_f64)
            .collect();
        let right: Vec<f64> = random_metrics
            .get(name)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_f64)
            .collect();
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
        warnings.push(
            "synthetic_vs_random: random baseline skipped for graphs above 5000 nodes".to_string(),
        );
    }
    json!({
        "network_validation": {
            "synthetic_vs_random": {
                "comparison": "synthetic_vs_random",
                "random_model": "degree_preserving_rnd",
                "wasserstein": wasserstein,
                "distributions": {"synthetic": source_dist, "random": random_dist},
                "source_network": source_public,
                "random_network": random_public,
            }
        },
        "warnings": warnings,
    })
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

    #[test]
    fn time_use_metric_rows_average_absolute_share_difference() {
        let block = json!({
            "groups": [{
                "filter_key": "all",
                "filter_label": "All",
                "block": {"rows": [
                    {"share_of_day_difference_pct_points": -2.0},
                    {"share_of_day_difference_pct_points": 4.0}
                ]}
            }]
        });
        let rows = time_use_metric_rows(&block);
        assert_eq!(rows[0]["metric"], "Mean absolute time-use share difference");
        assert_eq!(rows[0]["value"], 3.0);
    }
}
