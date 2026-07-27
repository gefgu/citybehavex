//! `distributions` chart section: ECDF blocks for jump-lengths,
//! visits-per-user, radius of gyration, dwell time, and trip duration.
//! Mirrors the ECDF-block half of `legacy.py::_build_comparison_payload`'s
//! `distribution_group` closure (the Wasserstein-metric-row half lives in
//! `sections::metrics`).

use crate::comparison::CAR_SPEED_KMH;
use crate::comparison::ecdf::ecdf_block;
use crate::comparison::filters::filter_df;
use crate::comparison::metrics::waiting_times_minutes;
use crate::comparison::panel::{AdaptationMode, adapt_evaluation_dataframe, collapse_to_stays};
use crate::comparison::trajectory::load_trajectory;
use crate::payload::{ComparisonContext, choose_filter, duration_col, empty_chart_payload};
use polars::prelude::*;
use serde_json::{Value, json};

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

pub fn distributions_section_payload(
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
