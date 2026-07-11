//! Mirrors the Wasserstein-metric-row assembly inside
//! `payload/legacy.py::_build_comparison_payload`'s `distribution_group`
//! closure (the part of it that appends to the `wasserstein` metrics list --
//! the ECDF-block half of that same closure lives in `sections::distributions`,
//! not yet ported).

use crate::comparison::features::{JumpsRog, jumps_rog_for_filters};
use crate::comparison::filters::FilterMeta;
use crate::comparison::metric_row::{MetricRow, metric_row};
use crate::comparison::metrics::wasserstein_distance;
use crate::comparison::panel::{AdaptationMode, adapt_evaluation_dataframe};
use crate::comparison::{CAR_SPEED_KMH, panel::collapse_to_stays};
use polars::prelude::*;
use std::collections::HashMap;

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
