//! Mirrors `web/backend/app/features.py`'s per-file jump-length/radius-of-
//! gyration computation (`get_jumps_rog`) -- the caching wrapper isn't
//! ported yet (a performance optimization, not a correctness requirement:
//! every call recomputes), but the core math is implemented directly here
//! using this crate's own Haversine/user-grouping primitives rather than
//! calling fkmob-core's batched jump-length/RoG kernels, since the math is
//! identical (both ultimately reduce to consecutive-Haversine-distance and
//! centroid-RMS-Haversine-distance per user) and this avoids the extra
//! indexed/ends calling convention fkmob-core's batch API needs.
//!
//! **Not yet ported**: the road-network-aware variant
//! (`citybehavex.metrics.jump_lengths_km`/`radius_of_gyration_km`, used when
//! `comparison.road_network_distance` is enabled and a road graph exists) --
//! most experiments in this repo don't have a road graph built yet
//! (`road_network_available: false`), so straight-line Haversine (this
//! module) covers the common case; the road-aware path can reuse
//! `citybehavex-core::roads::RoadGraph::batch_road_distances` when needed.

use super::filters::{FilterMeta, filter_df};
use super::panel::{AdaptationMode, adapt_evaluation_dataframe};
use super::util::haversine_km;
use polars::prelude::*;
use std::collections::HashMap;

pub struct JumpsRog {
    pub jumps: Vec<f64>,
    pub rog: Vec<f64>,
}

/// Mirrors `features.py::get_jumps_rog`'s `build()` closure across every
/// filter: per filter, `_filter_df` then `_adapt_evaluation_dataframe`
/// (mode `"auto"`, matching the *default* `evaluation_adaptation_config` the
/// synthetic-side call site in `_build_comparison_payload` passes -- the
/// observed-side call site passes the experiment's actual configured value
/// instead; both sides go through this same adaptation step, not just
/// observed) before computing jumps/RoG on the adapted result.
#[allow(clippy::too_many_arguments)]
pub fn jumps_rog_for_filters(
    df: &DataFrame,
    uid_col: &str,
    lat_col: &str,
    lng_col: &str,
    datetime_col: &str,
    filters: &[FilterMeta],
    label: &str,
    mode: AdaptationMode,
    location_col: Option<&str>,
    h3_resolution: u8,
) -> anyhow::Result<HashMap<String, JumpsRog>> {
    let mut out = HashMap::new();
    for meta in filters {
        let filtered = filter_df(df, Some(datetime_col), meta)?;
        if filtered.height() == 0 {
            out.insert(
                meta.key.clone(),
                JumpsRog {
                    jumps: Vec::new(),
                    rog: Vec::new(),
                },
            );
            continue;
        }
        let adapted = adapt_evaluation_dataframe(
            &filtered,
            label,
            uid_col,
            datetime_col,
            lat_col,
            lng_col,
            mode,
            location_col,
            h3_resolution,
        )?;
        let result = jumps_rog(&adapted.df, uid_col, lat_col, lng_col, datetime_col)?;
        out.insert(meta.key.clone(), result);
    }
    Ok(out)
}

/// Mirrors the non-road-aware branch of `features.py::get_jumps_rog`'s
/// `build()` closure for a single (already filtered/adapted) dataframe:
/// per-user consecutive-row jump lengths (zero-length jumps excluded, they
/// aren't movement) and per-user radius of gyration, matching
/// `fkmob.TrajDataFrame.jump_lengths(merge=True)`/`.radius_of_gyration()`.
pub fn jumps_rog(
    df: &DataFrame,
    uid_col: &str,
    lat_col: &str,
    lng_col: &str,
    datetime_col: &str,
) -> anyhow::Result<JumpsRog> {
    if df.height() == 0 {
        return Ok(JumpsRog {
            jumps: Vec::new(),
            rog: Vec::new(),
        });
    }
    let schema = df.schema();
    let dt_expr = super::util::to_datetime_expr(&schema, datetime_col);
    let sorted = df
        .clone()
        .lazy()
        .select([
            col(uid_col),
            col(lat_col).cast(DataType::Float64),
            col(lng_col).cast(DataType::Float64),
            dt_expr.alias(datetime_col),
        ])
        .drop_nulls(Some(cols([uid_col, lat_col, lng_col, datetime_col])))
        .sort([uid_col, datetime_col], SortMultipleOptions::default())
        .collect()?;
    if sorted.height() == 0 {
        return Ok(JumpsRog {
            jumps: Vec::new(),
            rog: Vec::new(),
        });
    }

    let uid: Vec<i64> = sorted
        .column(uid_col)?
        .cast(&DataType::Int64)?
        .i64()?
        .into_iter()
        .map(|v| v.unwrap_or(i64::MIN))
        .collect();
    let lat: Vec<f64> = sorted
        .column(lat_col)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    let lng: Vec<f64> = sorted
        .column(lng_col)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    let (_indices, ends) = super::activity::contiguous_user_ranges(&uid);
    let mut jumps = Vec::new();
    let mut rog = Vec::new();
    let mut start = 0usize;
    for &end in &ends {
        let user_lat = &lat[start..end];
        let user_lng = &lng[start..end];
        for i in 1..user_lat.len() {
            let d = haversine_km(user_lat[i - 1], user_lng[i - 1], user_lat[i], user_lng[i]);
            if d > 0.0 {
                jumps.push(d);
            }
        }
        rog.push(radius_of_gyration(user_lat, user_lng));
        start = end;
    }
    Ok(JumpsRog { jumps, rog })
}

/// Mirrors fkmob-core's `rog_for_slice`: RMS Haversine distance from each
/// point to the user's centroid (mean lat/lng, i.e. an arithmetic-mean
/// "center of mass" in degree-space, not a geodesic centroid -- matching
/// fkmob-core's own approximation, which is exact enough at city scale).
fn radius_of_gyration(lat: &[f64], lng: &[f64]) -> f64 {
    let n = lat.len();
    if n == 0 {
        return 0.0;
    }
    let cm_lat = lat.iter().sum::<f64>() / n as f64;
    let cm_lng = lng.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = lat
        .iter()
        .zip(lng.iter())
        .map(|(&la, &lo)| {
            let d = haversine_km(cm_lat, cm_lng, la, lo);
            d * d
        })
        .sum();
    (sum_sq / n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_point_user_has_zero_rog_and_no_jumps() {
        let df = df!["uid" => [1i64], "lat" => [48.85], "lng" => [2.35], "dt" => ["2026-01-01T00:00:00"]].unwrap();
        let result = jumps_rog(&df, "uid", "lat", "lng", "dt").unwrap();
        assert!(result.jumps.is_empty());
        assert_eq!(result.rog, vec![0.0]);
    }

    #[test]
    fn two_distinct_points_produce_one_jump_and_positive_rog() {
        let df = df![
            "uid" => [1i64, 1],
            "lat" => [48.85, 48.86],
            "lng" => [2.35, 2.36],
            "dt" => ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
        ]
        .unwrap();
        let result = jumps_rog(&df, "uid", "lat", "lng", "dt").unwrap();
        assert_eq!(result.jumps.len(), 1);
        assert!(result.jumps[0] > 0.0);
        assert_eq!(result.rog.len(), 1);
        assert!(result.rog[0] > 0.0);
    }

    #[test]
    fn zero_length_jumps_are_excluded() {
        let df = df![
            "uid" => [1i64, 1, 1],
            "lat" => [48.85, 48.85, 48.86],
            "lng" => [2.35, 2.35, 2.36],
            "dt" => ["2026-01-01T00:00:00", "2026-01-01T01:00:00", "2026-01-01T02:00:00"],
        ]
        .unwrap();
        let result = jumps_rog(&df, "uid", "lat", "lng", "dt").unwrap();
        assert_eq!(
            result.jumps.len(),
            1,
            "the zero-length repeat should be excluded"
        );
    }

    /// Cross-checked against the live Python backend's
    /// `traj.jump_lengths(merge=True)` (filtered `>0`) and
    /// `traj.radius_of_gyration()` on the same file: 1500 distinct agents,
    /// 38001 jumps summing to 262535.5969848962, mean RoG 5.80716178900295,
    /// and the smallest 5 sorted jump/RoG values.
    #[test]
    #[ignore = "requires repo data at data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet"]
    fn gparis_jumps_rog_match_python_reference() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let path = repo_root.join(
            "data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet",
        );
        let traj = super::super::trajectory::load_trajectory(&path).unwrap();

        let result = jumps_rog(
            &traj.df,
            &traj.uid_col,
            &traj.lat_col,
            &traj.lng_col,
            &traj.datetime_col,
        )
        .unwrap();
        assert_eq!(result.rog.len(), 1500);
        assert_eq!(result.jumps.len(), 38001);

        let sum_jumps: f64 = result.jumps.iter().sum();
        assert!(
            (sum_jumps - 262535.5969848962).abs() < 1e-3,
            "sum_jumps={sum_jumps}"
        );
        let mean_rog: f64 = result.rog.iter().sum::<f64>() / result.rog.len() as f64;
        assert!(
            (mean_rog - 5.80716178900295).abs() < 1e-9,
            "mean_rog={mean_rog}"
        );

        let mut sorted_jumps = result.jumps.clone();
        sorted_jumps.sort_by(f64::total_cmp);
        let expected_jumps = [
            1.11195074e-05,
            4.13498889e-05,
            5.00963216e-05,
            5.00963216e-05,
            5.37127812e-05,
        ];
        for (got, want) in sorted_jumps[..5].iter().zip(expected_jumps.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got} want {want}");
        }

        let mut sorted_rog = result.rog.clone();
        sorted_rog.sort_by(f64::total_cmp);
        let expected_rog = [0.09550834, 0.0986176, 0.10207505, 0.10282916, 0.1029499];
        for (got, want) in sorted_rog[..5].iter().zip(expected_rog.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
    }
}
