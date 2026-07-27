//! `mobility-laws` chart section: native law-block rendering for travel
//! distance, radius of gyration, daily locations, and distance-frequency.
//! Mirrors `legacy Python backend/payload/legacy.py`'s mobility-laws section.

use crate::columns::{ACTIVITY_CANDIDATES, LOCATION_CANDIDATES, detect_in};
use crate::comparison::filters::filter_df;
use crate::comparison::mobility_laws::{
    daily_location_lognormal_dataset, distance_frequency_dataset, mobility_law_visits,
    truncated_powerlaw_dataset,
};
use crate::comparison::panel::{AdaptationMode, adapt_evaluation_dataframe};
use crate::comparison::trajectory::load_trajectory;
use crate::payload::{ComparisonContext, choose_regular_filter, empty_chart_payload};
use polars::prelude::*;
use serde_json::{Value, json};

fn finite_xy(x: &[f64], y: &[f64]) -> Vec<[f64; 2]> {
    x.iter()
        .zip(y.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| [x, y])
        .collect()
}

/// Mirrors `legacy.py::_curve_x`, including its `n: int = 200` default
/// (confirmed all 3 call sites -- travel-distance, radius-of-gyration, and
/// daily-locations blocks -- rely on the default rather than overriding it).
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
    let n = 200usize;
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

pub fn mobility_laws_section_payload(
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
