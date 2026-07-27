//! Mirrors the Wasserstein-metric-row assembly inside
//! `payload/legacy.py::_build_comparison_payload`'s `distribution_group`
//! closure (the part of it that appends to the `wasserstein` metrics list --
//! the ECDF-block half of that same closure lives in `sections::distributions`).

use crate::comparison::features::{JumpsRog, jumps_rog_for_filters};
use crate::comparison::filters::{FilterMeta, filter_df};
use crate::comparison::metric_row::{MetricRow, metric_row};
use crate::comparison::metrics::{common_part_of_commuters, wasserstein_distance};
use crate::comparison::panel::{AdaptationMode, adapt_evaluation_dataframe};
use crate::comparison::stvd::stvd_hourly_histogram;
use crate::comparison::util::{canonical_user_ids_vec, to_datetime_expr};
use crate::comparison::{CAR_SPEED_KMH, CPC_H3_RESOLUTIONS, panel::collapse_to_stays};
use polars::prelude::*;
use std::collections::HashMap;

/// Extracts `(lat, lng, uid, timestamp)` parallel arrays for
/// `common_part_of_commuters`. Timestamps only need to sort consistently
/// with real time (not any particular unit), so microseconds work fine here
/// -- matching the codebase's existing `to_datetime_expr` convention rather
/// than converting to true milliseconds.
fn cpc_arrays(
    df: &DataFrame,
    lat_col: &str,
    lng_col: &str,
    uid_col: &str,
    datetime_col: &str,
) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<i64>, Vec<i64>)> {
    let schema = df.schema();
    let dt_expr = to_datetime_expr(&schema, datetime_col);
    let prepared = df
        .clone()
        .lazy()
        .select([
            col(uid_col),
            col(lat_col).cast(DataType::Float64),
            col(lng_col).cast(DataType::Float64),
            dt_expr.alias(datetime_col),
        ])
        .drop_nulls(Some(cols([uid_col, lat_col, lng_col, datetime_col])))
        .collect()?;
    let uid = canonical_user_ids_vec(prepared.column(uid_col)?.as_materialized_series())?;
    let lat: Vec<f64> = prepared
        .column(lat_col)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    let lng: Vec<f64> = prepared
        .column(lng_col)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    let ts: Vec<i64> = prepared
        .column(datetime_col)?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?
        .datetime()?
        .phys
        .into_iter()
        .map(|v| v.unwrap_or(0))
        .collect();
    Ok((lat, lng, uid, ts))
}

/// EPSG:4326 -> EPSG:3857 (Web Mercator), closed-form -- matches
/// `legacy.py`'s `Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)`
/// (confirmed in `RUST_BACKEND_MIGRATION.md`; no `proj` crate dependency
/// needed for this specific, standard reprojection).
fn web_mercator(lat: f64, lng: f64) -> (f64, f64) {
    const R: f64 = 6_378_137.0;
    let x = lng.to_radians() * R;
    let y = ((90.0 + lat).to_radians() / 2.0).tan().ln() * R;
    (x, y)
}

/// Mirrors `legacy.py::_stvd_emd_distribution`: flattens a per-cell hourly
/// histogram into `(x, y, minutes_of_day, volume)` point-cloud arrays for
/// `stvd_emd`, projecting each H3 cell's centroid to Web Mercator and each
/// hour to minutes-of-day (`stvd_emd`'s default `cyclical_period=1440.0` is
/// in minutes).
fn stvd_emd_distribution(
    layer: &crate::comparison::stvd::HourlyLayer,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ts = Vec::new();
    let mut ws = Vec::new();
    for (&cell, hours) in layer {
        let Ok(cell_index) = h3o::CellIndex::try_from(cell) else {
            continue;
        };
        let ll = h3o::LatLng::from(cell_index);
        let (x, y) = web_mercator(ll.lat(), ll.lng());
        for (hour, &count) in hours.iter().enumerate() {
            if count <= 0 {
                continue;
            }
            xs.push(x);
            ys.push(y);
            ts.push(hour as f64 * 60.0);
            ws.push(count as f64);
        }
    }
    (xs, ys, ts, ws)
}

pub struct Side<'a> {
    pub df: &'a DataFrame,
    pub uid_col: &'a str,
    pub lat_col: &'a str,
    pub lng_col: &'a str,
    pub datetime_col: &'a str,
    pub label: &'a str,
    /// Column holding an explicit trip-duration figure for this side, if
    /// one exists (`duration_col` on the observed side in Python; the
    /// synthetic side's `trip_duration_minutes` column is detected by name
    /// directly, matching `legacy.py`).
    pub duration_col: Option<&'a str>,
}

fn value_counts_per_user(df: &DataFrame, uid_col: &str) -> anyhow::Result<Vec<f64>> {
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

fn column_values_filtered(
    df: &DataFrame,
    name: &str,
    predicate: impl Fn(f64) -> bool,
) -> anyhow::Result<Option<Vec<f64>>> {
    if df.column(name).is_err() {
        return Ok(None);
    }
    let values: Vec<f64> = df
        .column(name)?
        .cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .flatten()
        .filter(|v| predicate(*v))
        .collect();
    Ok(Some(values))
}

/// Mirrors `_build_comparison_payload`'s per-filter Wasserstein-row
/// assembly for jump-lengths/visits-per-user/radius-of-gyration/dwell-time/
/// trip-duration. Only runs (and only emits rows) for filters where an
/// observed side is present and non-empty after filtering/adaptation --
/// matching Python's `if real_metric_group_df is not None and real_group_traj
/// is not None:` guard around the whole metrics-row block.
pub fn wasserstein_metric_rows(
    synthetic: &Side,
    observed: Option<&Side>,
    filters: &[FilterMeta],
    observed_mode: AdaptationMode,
    observed_location_col: Option<&str>,
    observed_h3_resolution: u8,
) -> anyhow::Result<Vec<MetricRow>> {
    let synth_jumps_rog: HashMap<String, JumpsRog> = jumps_rog_for_filters(
        synthetic.df,
        synthetic.uid_col,
        synthetic.lat_col,
        synthetic.lng_col,
        synthetic.datetime_col,
        filters,
        synthetic.label,
        AdaptationMode::Auto,
        None,
        10,
    )?;
    let real_jumps_rog: Option<HashMap<String, JumpsRog>> = match observed {
        Some(obs) => Some(jumps_rog_for_filters(
            obs.df,
            obs.uid_col,
            obs.lat_col,
            obs.lng_col,
            obs.datetime_col,
            filters,
            obs.label,
            observed_mode,
            observed_location_col,
            observed_h3_resolution,
        )?),
        None => None,
    };

    let mut rows = Vec::new();
    for meta in filters {
        let synth_df = crate::comparison::filters::filter_df(
            synthetic.df,
            Some(synthetic.datetime_col),
            meta,
        )?;
        if synth_df.height() == 0 {
            continue;
        }
        let Some(observed) = observed else { continue };
        let real_group_df =
            crate::comparison::filters::filter_df(observed.df, Some(observed.datetime_col), meta)?;
        if real_group_df.height() == 0 {
            continue;
        }
        let real_metric_group_df = adapt_evaluation_dataframe(
            &real_group_df,
            observed.label,
            observed.uid_col,
            observed.datetime_col,
            observed.lat_col,
            observed.lng_col,
            observed_mode,
            observed_location_col,
            observed_h3_resolution,
        )?
        .df;
        if real_metric_group_df.height() == 0 {
            continue;
        }

        let synth_jumps = &synth_jumps_rog[&meta.key].jumps;
        let real_jumps = &real_jumps_rog.as_ref().unwrap()[&meta.key].jumps;
        let synth_rog = &synth_jumps_rog[&meta.key].rog;
        let real_rog = &real_jumps_rog.as_ref().unwrap()[&meta.key].rog;

        let synth_stays = collapse_to_stays(
            &synth_df,
            synthetic.uid_col,
            synthetic.lat_col,
            synthetic.lng_col,
            synthetic.datetime_col,
        )?;
        let real_stays = collapse_to_stays(
            &real_metric_group_df,
            observed.uid_col,
            observed.lat_col,
            observed.lng_col,
            observed.datetime_col,
        )?;
        let synth_visits = value_counts_per_user(&synth_stays, synthetic.uid_col)?;
        let real_visits = value_counts_per_user(&real_stays, observed.uid_col)?;

        let synth_dwell = column_values_filtered(&synth_df, "dwell_minutes", |v| v >= 0.0)?
            .unwrap_or_else(|| {
                crate::comparison::metrics::waiting_times_minutes(
                    &synth_df,
                    synthetic.uid_col,
                    synthetic.datetime_col,
                )
                .unwrap_or_default()
            });
        let real_dwell = match observed.duration_col {
            Some(c) => {
                column_values_filtered(&real_metric_group_df, c, |_| true)?.unwrap_or_default()
            }
            None => crate::comparison::metrics::waiting_times_minutes(
                &real_metric_group_df,
                observed.uid_col,
                observed.datetime_col,
            )
            .unwrap_or_default(),
        };

        let (synth_trip, real_trip): (Vec<f64>, Vec<f64>) = if let Some(trip) =
            column_values_filtered(&synth_df, "trip_duration_minutes", |v| v > 0.0)?
        {
            let real_trip: Vec<f64> = real_jumps
                .iter()
                .filter(|&&j| j > 0.0)
                .map(|&j| (j / CAR_SPEED_KMH) * 60.0)
                .collect();
            (trip, real_trip)
        } else if let Some(c) = observed.duration_col {
            let synth_trip = crate::comparison::metrics::waiting_times_minutes(
                &synth_df,
                synthetic.uid_col,
                synthetic.datetime_col,
            )
            .unwrap_or_default();
            let real_trip =
                column_values_filtered(&real_metric_group_df, c, |_| true)?.unwrap_or_default();
            (synth_trip, real_trip)
        } else {
            (Vec::new(), Vec::new())
        };

        if !real_jumps.is_empty() {
            if let Some(row) = metric_row(
                meta,
                "Jump lengths",
                Some(wasserstein_distance(synth_jumps, real_jumps)),
                "km",
            ) {
                rows.push(row);
            }
        }
        if let Some(row) = metric_row(
            meta,
            "Visits per user",
            Some(wasserstein_distance(&synth_visits, &real_visits)),
            "visits",
        ) {
            rows.push(row);
        }
        if !real_rog.is_empty() {
            if let Some(row) = metric_row(
                meta,
                "Radius of gyration",
                Some(wasserstein_distance(synth_rog, real_rog)),
                "km",
            ) {
                rows.push(row);
            }
        }
        if !real_dwell.is_empty() {
            if let Some(row) = metric_row(
                meta,
                "Dwell time",
                Some(wasserstein_distance(&synth_dwell, &real_dwell)),
                "min",
            ) {
                rows.push(row);
            }
        }
        if !synth_trip.is_empty() && !real_trip.is_empty() {
            if let Some(row) = metric_row(
                meta,
                "Trip duration",
                Some(wasserstein_distance(&synth_trip, &real_trip)),
                "min",
            ) {
                rows.push(row);
            }
        }
    }
    Ok(rows)
}

/// The `metrics` chart section itself, mirrors `payload/sections.py::build_section_metrics`.
pub fn metrics_section_payload(
    ctx: &crate::payload::ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<serde_json::Value> {
    use crate::comparison::sections::motifs::build_motifs_block;
    use crate::comparison::trajectory::load_trajectory;
    use crate::payload::{
        choose_filter, duration_col, empty_chart_payload, prepared_visits_for_filter,
    };
    use serde_json::{Value, json};

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

    // Mirrors `legacy.py`'s CPC loop: `common_part_of_commuters` at
    // [7, 8, 9] between the current filter's synthetic/observed rows, only
    // when both sides have data (`mode == "comparison"`).
    let mut cpc_metrics: Vec<Value> = Vec::new();
    if let Some(obs_traj) = &observed_traj {
        let synth_filtered = filter_df(
            &synthetic_traj.df,
            Some(&synthetic_traj.datetime_col),
            &filter,
        )?;
        let real_filtered = filter_df(&obs_traj.df, Some(&obs_traj.datetime_col), &filter)?;
        if synth_filtered.height() > 0 && real_filtered.height() > 0 {
            let (lat_a, lng_a, uid_a, ts_a) = cpc_arrays(
                &synth_filtered,
                &synthetic_traj.lat_col,
                &synthetic_traj.lng_col,
                &synthetic_traj.uid_col,
                &synthetic_traj.datetime_col,
            )?;
            let (lat_b, lng_b, uid_b, ts_b) = cpc_arrays(
                &real_filtered,
                &obs_traj.lat_col,
                &obs_traj.lng_col,
                &obs_traj.uid_col,
                &obs_traj.datetime_col,
            )?;
            let cpc = common_part_of_commuters(
                &lat_a,
                &lng_a,
                &uid_a,
                &ts_a,
                &lat_b,
                &lng_b,
                &uid_b,
                &ts_b,
                &CPC_H3_RESOLUTIONS,
            )?;
            for (resolution, value) in cpc {
                cpc_metrics.push(json!({
                    "filter_key": filter.key,
                    "filter_label": filter.label,
                    "resolution": resolution,
                    "value": value,
                }));
            }
        }
    }
    payload["metrics"]["cpc"] = serde_json::to_value(cpc_metrics)?;

    // Mirrors `legacy.py::_stvd_metric_rows`: STVD-EMD (spatio-temporal
    // sliced-Wasserstein) between the current filter's synthetic/observed
    // hourly H3 histograms, one row per resolution in [7, 8, 9], only when
    // both sides have lat/lng.
    let mut stvd_metrics: Vec<Value> = Vec::new();
    if let Some(obs_traj) = &observed_traj {
        let synth_filtered = filter_df(
            &synthetic_traj.df,
            Some(&synthetic_traj.datetime_col),
            &filter,
        )?;
        let real_filtered = filter_df(&obs_traj.df, Some(&obs_traj.datetime_col), &filter)?;
        if synth_filtered.height() > 0 && real_filtered.height() > 0 {
            let synth_hourly = stvd_hourly_histogram(
                &synth_filtered,
                &synthetic_traj.lat_col,
                &synthetic_traj.lng_col,
                &synthetic_traj.datetime_col,
                &CPC_H3_RESOLUTIONS,
            )?;
            let real_hourly = stvd_hourly_histogram(
                &real_filtered,
                &obs_traj.lat_col,
                &obs_traj.lng_col,
                &obs_traj.datetime_col,
                &CPC_H3_RESOLUTIONS,
            )?;
            for &resolution in &CPC_H3_RESOLUTIONS {
                let (Some(synth_layer), Some(real_layer)) =
                    (synth_hourly.get(&resolution), real_hourly.get(&resolution))
                else {
                    continue;
                };
                let (xs_a, ys_a, ts_a, ws_a) = stvd_emd_distribution(synth_layer);
                let (xs_b, ys_b, ts_b, ws_b) = stvd_emd_distribution(real_layer);
                if xs_a.is_empty() || xs_b.is_empty() {
                    continue;
                }
                let value = fastmob_core::measures::evaluation::stvd_emd::stvd_emd_impl(
                    &xs_a, &ys_a, &ts_a, &ws_a, &xs_b, &ys_b, &ts_b, &ws_b, 10.0, 1440.0, 50,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
                if let Some(row) = metric_row(&filter, "STVD-EMD", Some(value), "m") {
                    let mut row = serde_json::to_value(row)?;
                    row["resolution"] = json!(resolution);
                    stvd_metrics.push(row);
                }
            }
        }
    }
    payload["metrics"]["stvd"] = serde_json::to_value(stvd_metrics)?;

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

    // Mirrors `legacy.py`'s activity-JSD side effect (the same
    // `activity_transition_matrix`/`daily_activity_distribution` computation
    // `sections::activity` does for the `activity` chart, re-run here purely
    // for these 3 JSD values): only when both sides have data, matching
    // Python's `if obs_v is not None and not obs_v.is_empty()` guard.
    if let (Some(syn_v), Some(obs_v)) = (visits.synthetic.as_ref(), visits.observed.as_ref()) {
        use crate::comparison::activity::activity_transition_matrix;
        use crate::comparison::sections::activity::{
            align_daily, align_square, daily_tuple, ordered_union, string_column,
        };

        let syn_purposes = string_column(syn_v, "purpose")?;
        let obs_purposes = string_column(obs_v, "purpose")?;
        let mut counts_syn = HashMap::<String, f64>::new();
        let mut counts_obs = HashMap::<String, f64>::new();
        for p in &syn_purposes {
            *counts_syn.entry(p.clone()).or_insert(0.0) += 1.0;
        }
        for p in &obs_purposes {
            *counts_obs.entry(p.clone()).or_insert(0.0) += 1.0;
        }
        let mut labels: Vec<String> = counts_syn
            .keys()
            .chain(counts_obs.keys())
            .cloned()
            .collect();
        labels.sort();
        labels.dedup();
        let v1: Vec<f64> = labels
            .iter()
            .map(|l| counts_syn.get(l).copied().unwrap_or(0.0))
            .collect();
        let v2: Vec<f64> = labels
            .iter()
            .map(|l| counts_obs.get(l).copied().unwrap_or(0.0))
            .collect();
        if let Some(row) = crate::comparison::metrics::jensen_shannon_divergence(&v1, &v2)
            .ok()
            .and_then(|value| metric_row(&filter, "Activity distribution", Some(value), ""))
        {
            jsd.push(row);
        }

        let (syn_trans_cats, syn_trans_mat) = activity_transition_matrix(syn_v, "uid", "purpose")?;
        let (obs_trans_cats, obs_trans_mat) = activity_transition_matrix(obs_v, "uid", "purpose")?;
        let trans_union = ordered_union(&syn_trans_cats, &obs_trans_cats);
        let syn_trans_aligned = align_square(&syn_trans_cats, &syn_trans_mat, &trans_union);
        let obs_trans_aligned = align_square(&obs_trans_cats, &obs_trans_mat, &trans_union);
        let flat_syn: Vec<f64> = syn_trans_aligned.into_iter().flatten().collect();
        let flat_obs: Vec<f64> = obs_trans_aligned.into_iter().flatten().collect();
        if let Some(row) =
            crate::comparison::metrics::jensen_shannon_divergence(&flat_syn, &flat_obs)
                .ok()
                .and_then(|value| metric_row(&filter, "Activity transitions", Some(value), ""))
        {
            jsd.push(row);
        }

        let (syn_daily_cats, syn_daily_mat) = daily_tuple(syn_v)?;
        let (obs_daily_cats, obs_daily_mat) = daily_tuple(obs_v)?;
        let daily_union = ordered_union(&syn_daily_cats, &obs_daily_cats);
        let syn_daily_aligned = align_daily(&syn_daily_cats, &syn_daily_mat, &daily_union);
        let obs_daily_aligned = align_daily(&obs_daily_cats, &obs_daily_mat, &daily_union);
        if let Some(row) =
            crate::comparison::metrics::time_bin_matrix_jsd(&syn_daily_aligned, &obs_daily_aligned)
                .ok()
                .and_then(|value| metric_row(&filter, "Daily activity profile", Some(value), ""))
        {
            jsd.push(row);
        }
    }

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
    if let Ok(time_use_payload) =
        crate::comparison::sections::time_use::time_use_section_payload(ctx, filter_key)
    {
        if !time_use_payload
            .get("time_use_comparison")
            .is_some_and(Value::is_null)
        {
            payload["time_use_comparison"] = time_use_payload["time_use_comparison"].clone();
            payload["metrics"]["time_use"] =
                crate::comparison::sections::time_use::time_use_metric_rows(
                    &time_use_payload["time_use_comparison"],
                );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::filters::filters;

    fn traj_df() -> DataFrame {
        df![
            "uid" => [1i64, 1, 1, 2, 2],
            "lat" => [48.85, 48.86, 48.87, 48.90, 48.91],
            "lng" => [2.35, 2.36, 2.37, 2.40, 2.41],
            "dt" => [
                "2026-01-05T08:00:00", "2026-01-05T12:00:00", "2026-01-05T18:00:00",
                "2026-01-06T08:00:00", "2026-01-06T18:00:00",
            ],
            "dwell_minutes" => [10.0, 20.0, 30.0, 15.0, 25.0],
            "trip_duration_minutes" => [5.0, 6.0, 7.0, 8.0, 9.0],
        ]
        .unwrap()
    }

    #[test]
    fn produces_rows_for_all_filter_when_observed_present() {
        let syn = traj_df();
        let obs = traj_df();
        let synthetic = Side {
            df: &syn,
            uid_col: "uid",
            lat_col: "lat",
            lng_col: "lng",
            datetime_col: "dt",
            label: "synthetic",
            duration_col: None,
        };
        // Observed dwell-time uses the auto-detected `duration_col` (Python:
        // `_DURATION_CANDIDATES`), not a hardcoded "dwell_minutes" name like
        // the synthetic side -- set it explicitly here so both sides pull
        // from the same identical column and the "identical data -> zero
        // distance" assertion below is actually valid for every row.
        let observed = Side {
            df: &obs,
            uid_col: "uid",
            lat_col: "lat",
            lng_col: "lng",
            datetime_col: "dt",
            label: "observed",
            duration_col: Some("dwell_minutes"),
        };
        let all_filter = vec![filters().into_iter().find(|f| f.key == "all").unwrap()];
        let rows = wasserstein_metric_rows(
            &synthetic,
            Some(&observed),
            &all_filter,
            AdaptationMode::Auto,
            None,
            10,
        )
        .unwrap();
        // Identical synthetic/observed data -> every directly paired
        // Wasserstein distance should be ~0. Trip duration is deliberately
        // different under the Python-compatible branch: when the synthetic
        // side has `trip_duration_minutes`, the observed side is compared via
        // jump-derived car-time, not an observed trip-duration column.
        assert!(!rows.is_empty());
        for row in &rows {
            if row.metric_name == "Trip duration" {
                continue;
            }
            assert!(row.value.abs() < 1e-9, "{}: {}", row.metric_name, row.value);
        }
    }

    #[test]
    fn no_observed_produces_no_rows() {
        let syn = traj_df();
        let synthetic = Side {
            df: &syn,
            uid_col: "uid",
            lat_col: "lat",
            lng_col: "lng",
            datetime_col: "dt",
            label: "synthetic",
            duration_col: None,
        };
        let all_filter = vec![filters().into_iter().find(|f| f.key == "all").unwrap()];
        let rows = wasserstein_metric_rows(
            &synthetic,
            None,
            &all_filter,
            AdaptationMode::Auto,
            None,
            10,
        )
        .unwrap();
        assert!(rows.is_empty());
    }
}
