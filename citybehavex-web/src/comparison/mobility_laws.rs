//! Mirrors the mobility-law dataset builders in `comparison.py`:
//! `_mobility_law_visits`, `_daily_location_lognormal_dataset`,
//! `_distance_frequency_dataset` (+ fkmob's `compute_visitation_law_data`,
//! `bin_visitation_law_data`, `fit_visitation_law`).
//!
//! `_truncated_powerlaw_dataset` (fkmob's `fit_values_to_truncated_powerlaw`,
//! a *bounded* nonlinear least-squares fit via scipy's Trust-Region-Reflective
//! solver) is intentionally NOT ported here yet -- it needs a bounded-NLLS
//! solver decision (a Rust crate, or a hand-rolled Levenberg-Marquardt with a
//! bounds reparameterization) that's a distinct piece of scope from the rest
//! of this module's straightforward Polars/numpy-equivalent ports.

use super::h3::h3_cells;
use super::util::to_datetime_expr;
use polars::prelude::*;

/// **Not yet implemented** -- mirrors fkmob's `fit_values_to_truncated_powerlaw`
/// (`p(x) = c*(x+r0)^-beta * exp(-x/kappa)`, fit via scipy's bounded
/// Trust-Region-Reflective `curve_fit` in log-space over a log-spaced,
/// density-normalized histogram). Feeds the jump-length and
/// radius-of-gyration truncated-power-law mobility-law curves (2 of the 3
/// curve families in that payload section; `distance_frequency_dataset`
/// above is the third and IS done). Deferred pending a bounded-NLLS crate
/// decision -- see the plan's Phase 5 notes.
pub fn truncated_powerlaw_dataset(_values: &[f64], _label: &str) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<f64>, String)> {
    anyhow::bail!(
        "truncated_powerlaw_dataset is not yet implemented (needs a bounded-NLLS solver decision, see plan Phase 5)"
    )
}

/// Mirrors `comparison.py::_mobility_law_visits`.
pub fn mobility_law_visits(
    df: &DataFrame,
    uid_col: &str,
    datetime_col: &str,
    lat_col: &str,
    lng_col: &str,
    location_col: Option<&str>,
    activity_col: Option<&str>,
    location_resolution: u8,
) -> anyhow::Result<DataFrame> {
    let mut columns = vec![col(uid_col), col(datetime_col), col(lat_col), col(lng_col)];
    if let Some(c) = location_col {
        columns.push(col(c));
    }
    if let Some(c) = activity_col {
        columns.push(col(c));
    }

    let schema = df.schema();
    let dt_expr = to_datetime_expr(&schema, datetime_col);
    let source = df
        .clone()
        .lazy()
        .select(columns)
        .with_columns([
            dt_expr.alias(datetime_col),
            col(lat_col).cast(DataType::Float64),
            col(lng_col).cast(DataType::Float64),
        ])
        .drop_nulls(Some(cols([uid_col, datetime_col, lat_col, lng_col])))
        .filter(col(lat_col).is_between(lit(-90.0), lit(90.0), ClosedInterval::Both))
        .filter(col(lng_col).is_between(lit(-180.0), lit(180.0), ClosedInterval::Both))
        .collect()?;

    let lat_series = source.column(lat_col)?.as_materialized_series();
    let lng_series = source.column(lng_col)?.as_materialized_series();

    let location_id: Series = if let Some(location_col) = location_col {
        let loc = source.column(location_col)?.as_materialized_series().cast(&DataType::String)?;
        let missing_mask = loc.is_null();
        if missing_mask.sum().unwrap_or(0) > 0 {
            let fallback = h3_cells(lat_series, lng_series, location_resolution)?.cast(&DataType::String)?;
            let loc_str = loc.str()?;
            let fallback_str = fallback.str()?;
            let combined: StringChunked = loc_str
                .into_iter()
                .zip(fallback_str.into_iter())
                .map(|(l, f)| l.or(f).map(str::to_string))
                .collect();
            combined.into_series()
        } else {
            loc
        }
    } else {
        h3_cells(lat_series, lng_series, location_resolution)?.cast(&DataType::String)?
    };

    let mut visits = df![
        "user_id" => source.column(uid_col)?.as_materialized_series().clone(),
        "timestamp" => source.column(datetime_col)?.as_materialized_series().clone(),
        "lat" => lat_series.clone(),
        "lng" => lng_series.clone(),
    ]?;
    visits.with_column(location_id.with_name("location_id".into()))?;
    if let Some(activity_col) = activity_col {
        visits.with_column(
            source.column(activity_col)?.as_materialized_series().clone().with_name("purpose".into()),
        )?;
    }
    Ok(visits)
}

/// Mirrors `comparison.py::_daily_location_lognormal_dataset`.
pub fn daily_location_lognormal_dataset(
    visits: &DataFrame,
    label: &str,
) -> anyhow::Result<(Vec<f64>, Vec<f64>, f64, f64, String)> {
    let daily = visits
        .clone()
        .lazy()
        .with_columns([col("timestamp").dt().truncate(lit("1d")).alias("date")])
        .group_by([col("user_id"), col("date")])
        .agg([col("location_id").n_unique().alias("_count")])
        .collect()?;

    let values: Vec<f64> = daily
        .column("_count")?
        .as_materialized_series()
        .cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .flatten()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if values.len() < 2 {
        anyhow::bail!("at least two daily location counts are required");
    }

    let log_values: Vec<f64> = values.iter().map(|v| v.ln()).collect();
    let mu = log_values.iter().sum::<f64>() / log_values.len() as f64;
    let variance = log_values.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / log_values.len() as f64;
    let sigma = variance.sqrt();
    if !sigma.is_finite() || sigma <= 1e-12 {
        anyhow::bail!("daily location counts must have positive log variance");
    }

    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut x_points = Vec::new();
    let mut counts = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        x_points.push(sorted[i]);
        counts.push((j - i) as f64);
        i = j;
    }
    let total: f64 = counts.iter().sum();
    let y_points: Vec<f64> = counts.iter().map(|c| c / total).collect();

    Ok((x_points, y_points, mu, sigma, label.to_string()))
}

/// Per-user location with the max value of `count_col`, tie-broken by
/// smallest `location_id` -- shared tie-break pattern behind
/// `compute_visitation_law_data`'s `fallback_home`/`purpose_home` inference.
fn top_location_per_user(df: &DataFrame, count_col: &str) -> anyhow::Result<DataFrame> {
    Ok(df
        .clone()
        .lazy()
        .sort(
            ["user_id", count_col, "location_id"],
            SortMultipleOptions::default().with_order_descending_multi([false, true, false]),
        )
        .unique(Some(cols(["user_id"])), UniqueKeepStrategy::First)
        .select([col("user_id"), col("location_id")])
        .collect()?)
}

/// Mirrors fkmob's `compute_visitation_law_data`: per (user, location) visit
/// counts/frequencies, home-location inference (from an explicit `purpose`
/// column when present, else the most-visited location), and the
/// home-to-location Haversine distance `r_km` via `fkmob-core`'s
/// `visitation_distances_km` kernel (bit-identical to fkmob's own Rust call).
pub fn visitation_law_data(visits: &DataFrame) -> anyhow::Result<DataFrame> {
    let has_purpose = visits.get_column_names().iter().any(|c| c.as_str() == "purpose");

    let loc_coords = visits
        .clone()
        .lazy()
        .select([col("location_id"), col("lat"), col("lng")])
        .unique(Some(cols(["location_id"])), UniqueKeepStrategy::First)
        .collect()?;

    let base = visits
        .clone()
        .lazy()
        .drop_nulls(Some(cols(["user_id", "location_id", "timestamp"])))
        .with_columns([col("timestamp").dt().truncate(lit("1d")).alias("__visit_day__")])
        .drop_nulls(Some(cols(["__visit_day__"])));

    let visit_counts = base
        .group_by([col("user_id"), col("location_id")])
        .agg([len().alias("n_visits"), col("__visit_day__").n_unique().alias("f")])
        .sort(["user_id", "location_id"], SortMultipleOptions::default())
        .collect()?;
    if visit_counts.height() == 0 {
        return empty_law_data();
    }

    let fallback_home = top_location_per_user(&visit_counts, "n_visits")?;

    let home_location = if has_purpose {
        let home_counts = visits
            .clone()
            .lazy()
            .filter(col("purpose").eq(lit("HOME")))
            .group_by([col("user_id"), col("location_id")])
            .agg([len().alias("_home_count")])
            .collect()?;
        if home_counts.height() == 0 {
            fallback_home
        } else {
            let purpose_home = top_location_per_user(&home_counts, "_home_count")?;
            // Users with at least one HOME-purpose visit use `purpose_home`;
            // everyone else falls back to their most-visited location. Built
            // directly over materialized user-id vectors rather than an
            // Expr-level predicate -- simpler and avoids relying on `Expr::map`'s
            // less-common closure-based custom-predicate API for what's a
            // one-off set-membership filter.
            let purpose_home_users: std::collections::HashSet<i64> = purpose_home
                .column("user_id")?
                .i64()?
                .into_iter()
                .flatten()
                .collect();
            let fallback_user_ids = fallback_home.column("user_id")?.i64()?.clone();
            let keep_mask: BooleanChunked = fallback_user_ids
                .into_iter()
                .map(|v| v.map(|v| !purpose_home_users.contains(&v)))
                .collect();
            let fallback_missing = fallback_home.filter(&keep_mask)?;
            purpose_home.vstack(&fallback_missing)?
        }
    } else {
        fallback_home
    };

    let home_with_coords = home_location
        .lazy()
        .join(
            loc_coords.clone().lazy(),
            [col("location_id")],
            [col("location_id")],
            JoinArgs::new(JoinType::Inner),
        )
        .select([col("user_id"), col("location_id").alias("_home_location_id"), col("lat").alias("home_lat"), col("lng").alias("home_lng")]);

    let merged = visit_counts
        .lazy()
        .join(home_with_coords, [col("user_id")], [col("user_id")], JoinArgs::new(JoinType::Inner))
        .join(
            loc_coords.lazy(),
            [col("location_id")],
            [col("location_id")],
            JoinArgs::new(JoinType::Inner),
        )
        .rename(["lat", "lng"], ["loc_lat", "loc_lng"], true)
        .collect()?;
    if merged.height() == 0 {
        return empty_law_data();
    }

    let home_lat: Vec<f64> = merged.column("home_lat")?.f64()?.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let home_lng: Vec<f64> = merged.column("home_lng")?.f64()?.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let loc_lat: Vec<f64> = merged.column("loc_lat")?.f64()?.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let loc_lng: Vec<f64> = merged.column("loc_lng")?.f64()?.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let r_km = fkmob_core::measures::collective::visitation_law::visitation_distances_km(home_lat, home_lng, loc_lat, loc_lng)
        .map_err(|e| anyhow::anyhow!(e))?;

    let mut out = merged.select(["user_id", "location_id", "f", "n_visits"])?;
    out.with_column(Series::new("r_km".into(), r_km.clone()))?;
    let f_vals: Vec<f64> = out.column("f")?.cast(&DataType::Float64)?.f64()?.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let rf: Vec<f64> = r_km.iter().zip(f_vals.iter()).map(|(r, f)| r * f).collect();
    out.with_column(Series::new("rf".into(), rf))?;

    Ok(out
        .lazy()
        .select([col("user_id"), col("location_id"), col("r_km"), col("f"), col("rf"), col("n_visits")])
        .sort(["user_id", "location_id"], SortMultipleOptions::default())
        .collect()?)
}

fn empty_law_data() -> anyhow::Result<DataFrame> {
    Ok(df![
        "user_id" => Vec::<i64>::new(),
        "location_id" => Vec::<String>::new(),
        "r_km" => Vec::<f64>::new(),
        "f" => Vec::<i64>::new(),
        "rf" => Vec::<f64>::new(),
        "n_visits" => Vec::<i64>::new(),
    ]?)
}

/// Mirrors fkmob's `bin_visitation_law_data`: log-spaced binning of
/// `(rf, unique-user-density rho)` for the Schläpfer et al. visitation-law
/// fit. `rho_i(r,f) = (distinct users in this (location, r-bin, f) cell) /
/// (annulus area at radius r, width `distance_bin_width_km`)`.
pub fn bin_visitation_law_data(
    law_data: &DataFrame,
    n_bins: usize,
    distance_bin_width_km: f64,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    if distance_bin_width_km <= 0.0 {
        anyhow::bail!("distance_bin_width_km must be > 0");
    }
    if n_bins == 0 {
        anyhow::bail!("n_bins must be > 0");
    }

    let user_id = law_data.column("user_id")?.i64()?;
    let location_id = law_data.column("location_id")?.str()?;
    let r_km = law_data.column("r_km")?.f64()?;
    let f = law_data.column("f")?.cast(&DataType::Float64)?;
    let f = f.f64()?;

    use std::collections::{HashMap, HashSet};
    // key: (location_id, r_center_bits, f_bits) -> set of user ids
    let mut groups: HashMap<(String, u64, u64), HashSet<i64>> = HashMap::new();
    for i in 0..law_data.height() {
        let (Some(uid), Some(loc), Some(r), Some(freq)) = (user_id.get(i), location_id.get(i), r_km.get(i), f.get(i)) else {
            continue;
        };
        if !(r.is_finite() && r > 0.0 && freq.is_finite() && freq > 0.0) {
            continue;
        }
        let r_center = (r / distance_bin_width_km).floor() * distance_bin_width_km + distance_bin_width_km / 2.0;
        groups
            .entry((loc.to_string(), r_center.to_bits(), freq.to_bits()))
            .or_default()
            .insert(uid);
    }

    let mut rf_values = Vec::new();
    let mut rho_values = Vec::new();
    for ((_, r_center_bits, f_bits), users) in &groups {
        let r_center = f64::from_bits(*r_center_bits);
        let freq = f64::from_bits(*f_bits);
        let annulus_area = 2.0 * std::f64::consts::PI * r_center * distance_bin_width_km;
        if annulus_area <= 0.0 {
            continue;
        }
        let rho = users.len() as f64 / annulus_area;
        let rf = r_center * freq;
        if rf > 0.0 && rho > 0.0 {
            rf_values.push(rf);
            rho_values.push(rho);
        }
    }

    if rf_values.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let rf_min = rf_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let rf_max = rf_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if rf_min == rf_max {
        let mean_rho = rho_values.iter().sum::<f64>() / rho_values.len() as f64;
        return Ok((vec![rf_min], vec![mean_rho]));
    }

    let log_min = rf_min.log10();
    let log_max = rf_max.log10();
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| 10f64.powf(log_min + (log_max - log_min) * (i as f64) / (n_bins as f64)))
        .collect();
    let centers: Vec<f64> = (0..n_bins).map(|i| (edges[i] * edges[i + 1]).sqrt()).collect();

    let mut rf_out = Vec::new();
    let mut rho_out = Vec::new();
    for idx in 0..n_bins {
        let lo = edges[idx];
        let hi = edges[idx + 1];
        let is_last = idx == n_bins - 1;
        let mut bucket = Vec::new();
        for (&rf, &rho) in rf_values.iter().zip(rho_values.iter()) {
            let in_bin = if is_last { rf >= lo && rf <= hi } else { rf >= lo && rf < hi };
            if in_bin {
                bucket.push(rho);
            }
        }
        if !bucket.is_empty() {
            rf_out.push(centers[idx]);
            rho_out.push(bucket.iter().sum::<f64>() / bucket.len() as f64);
        }
    }
    Ok((rf_out, rho_out))
}

/// Mirrors fkmob's `fit_visitation_law`: OLS fit of `log(rho)` on `log(rf)`
/// (`rho(r,f) = mu * (rf)^-eta`), no bounds/scipy needed -- a closed-form
/// degree-1 polynomial fit.
pub fn fit_visitation_law(rf_values: &[f64], rho_values: &[f64]) -> anyhow::Result<(f64, f64, f64)> {
    if rf_values.len() != rho_values.len() {
        anyhow::bail!("rf_values and rho_values must have the same length");
    }
    let xy: Vec<(f64, f64)> = rf_values
        .iter()
        .zip(rho_values.iter())
        .filter(|pair: &(&f64, &f64)| pair.0.is_finite() && pair.1.is_finite() && *pair.0 > 0.0 && *pair.1 > 0.0)
        .map(|(&rf, &rho)| (rf.ln(), rho.ln()))
        .collect();
    if xy.len() < 2 {
        anyhow::bail!("At least two positive finite data points are required to fit.");
    }
    let n = xy.len() as f64;
    let x_mean = xy.iter().map(|(x, _)| x).sum::<f64>() / n;
    let y_mean = xy.iter().map(|(_, y)| y).sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (x, y) in &xy {
        sxy += (x - x_mean) * (y - y_mean);
        sxx += (x - x_mean).powi(2);
    }
    let slope = sxy / sxx;
    let intercept = y_mean - slope * x_mean;
    let eta = -slope;
    let mu = intercept.exp();

    let ss_res: f64 = xy.iter().map(|(x, y)| (y - (intercept + slope * x)).powi(2)).sum();
    let ss_tot: f64 = xy.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
    let r2 = if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot };

    Ok((eta, mu, r2))
}

/// Mirrors `comparison.py::_distance_frequency_dataset`.
pub fn distance_frequency_dataset(visits: &DataFrame, label: &str) -> anyhow::Result<(Vec<f64>, Vec<f64>, f64, f64, String)> {
    let law_data = visitation_law_data(visits)?;
    let (rf_points, rho_points) = bin_visitation_law_data(&law_data, 30, 1.0)?;
    let (eta, mu, _r2) = fit_visitation_law(&rf_points, &rho_points)?;
    if eta <= 0.0 || mu <= 0.0 {
        anyhow::bail!("distance-frequency fit parameters must be positive");
    }
    Ok((rf_points, rho_points, eta, mu, label.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_visitation_law_recovers_known_power_law() {
        // rho = 5.0 * rf^-1.5 exactly, no noise.
        let rf: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let rho: Vec<f64> = rf.iter().map(|&x| 5.0 * x.powf(-1.5)).collect();
        let (eta, mu, r2) = fit_visitation_law(&rf, &rho).unwrap();
        assert!((eta - 1.5).abs() < 1e-9, "eta={eta}");
        assert!((mu - 5.0).abs() < 1e-9, "mu={mu}");
        assert!((r2 - 1.0).abs() < 1e-9, "r2={r2}");
    }

    #[test]
    fn daily_location_lognormal_needs_two_points() {
        let visits = df![
            "user_id" => [1i64],
            "timestamp" => ["2026-01-01T00:00:00"],
            "lat" => [1.0],
            "lng" => [1.0],
            "location_id" => ["a"],
        ]
        .unwrap()
        .lazy()
        .with_columns([col("timestamp").str().to_datetime(Some(TimeUnit::Microseconds), None, StrptimeOptions::default(), lit("raise"))])
        .collect()
        .unwrap();
        assert!(daily_location_lognormal_dataset(&visits, "test").is_err());
    }

    #[test]
    fn bin_visitation_law_data_single_rf_value() {
        let law_data = df![
            "user_id" => [1i64, 2],
            "location_id" => ["a", "a"],
            "r_km" => [2.0, 2.0],
            "f" => [1i64, 1],
            "rf" => [2.0, 2.0],
            "n_visits" => [1i64, 1],
        ]
        .unwrap();
        let (rf, rho) = bin_visitation_law_data(&law_data, 30, 1.0).unwrap();
        assert_eq!(rf.len(), 1);
        assert_eq!(rho.len(), 1);
    }

    /// Cross-checked against the live Python backend's
    /// `_mobility_law_visits` + `_distance_frequency_dataset` on the same
    /// gparis trajectory (with the real `purpose` activity column, no
    /// explicit `location_col` -- the H3-fallback path): 39578 visit rows,
    /// 26 binned (rf, rho) points, `eta=0.821291713354924`,
    /// `mu=0.29080778179024974`.
    #[test]
    #[ignore = "requires repo data at data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet"]
    fn gparis_distance_frequency_matches_python_reference() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let path = repo_root.join("data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet");
        let traj = super::super::trajectory::load_trajectory(&path).unwrap();

        let cols: Vec<&str> = traj.df.get_column_names().iter().map(|s| s.as_str()).collect();
        let activity_col = crate::columns::detect_in(&cols, crate::columns::ACTIVITY_CANDIDATES);
        assert_eq!(activity_col.as_deref(), Some("purpose"));

        let visits = mobility_law_visits(
            &traj.df,
            &traj.uid_col,
            &traj.datetime_col,
            &traj.lat_col,
            &traj.lng_col,
            None,
            activity_col.as_deref(),
            10,
        )
        .unwrap();
        assert_eq!(visits.height(), 39578);

        let (rf_points, rho_points, eta, mu, _label) = distance_frequency_dataset(&visits, "synthetic").unwrap();
        assert_eq!(rf_points.len(), 26);
        assert_eq!(rho_points.len(), 26);
        assert!((eta - 0.821291713354924).abs() < 1e-6, "eta={eta}");
        assert!((mu - 0.29080778179024974).abs() < 1e-6, "mu={mu}");
    }

    /// Cross-checked against the live Python backend's
    /// `_daily_location_lognormal_dataset` on the same visits: 5 distinct
    /// daily-location-count values, `mu=0.9835439941419832`,
    /// `sigma=0.2767228111329802`.
    #[test]
    #[ignore = "requires repo data at data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet"]
    fn gparis_daily_location_lognormal_matches_python_reference() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let path = repo_root.join("data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet");
        let traj = super::super::trajectory::load_trajectory(&path).unwrap();
        let cols: Vec<&str> = traj.df.get_column_names().iter().map(|s| s.as_str()).collect();
        let activity_col = crate::columns::detect_in(&cols, crate::columns::ACTIVITY_CANDIDATES);

        let visits = mobility_law_visits(
            &traj.df,
            &traj.uid_col,
            &traj.datetime_col,
            &traj.lat_col,
            &traj.lng_col,
            None,
            activity_col.as_deref(),
            10,
        )
        .unwrap();

        let (x_points, y_points, mu, sigma, _label) = daily_location_lognormal_dataset(&visits, "synthetic").unwrap();
        assert_eq!(x_points.len(), 5);
        assert_eq!(y_points.len(), 5);
        assert_eq!(x_points, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((mu - 0.9835439941419832).abs() < 1e-9, "mu={mu}");
        assert!((sigma - 0.2767228111329802).abs() < 1e-9, "sigma={sigma}");
        let expected_y = [0.00751366, 0.38797814, 0.44555035, 0.13758782, 0.02137002];
        for (got, want) in y_points.iter().zip(expected_y.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
    }
}
