//! Mirrors `legacy Python backend/features.py`'s per-file jump-length/radius-of-
//! gyration computation (`get_jumps_rog`) -- the caching wrapper isn't
//! ported yet (a performance optimization, not a correctness requirement:
//! every call recomputes).
//!
//! The whole pipeline -- column resolution, cleaning, arranging by
//! `(uid, datetime)`, group boundaries and the distance kernels -- now lives in
//! `fastmob-rs`, fastmob's public Rust DataFrame API, rather than being
//! re-derived here. That crate's `prepare()` replaces the local
//! select/drop_nulls/sort/materialize block and beats it by ~1.6x at 4M rows
//! (127 ms vs 206 ms), because it arranges via a counting sort over user codes
//! plus a parallel per-group timestamp sort instead of a whole-frame Polars
//! sort. See `fastmob-rs/BENCHMARKS.md`.
//!
//! **Bit-exactness matters here, unlike most other "either formula works"
//! spots**: Python's `fastmob.TrajDataFrame.jump_lengths(merge=True)` reaches
//! the same `fastmob_core` kernel via PyO3, so a different (if mathematically
//! equivalent) hand-rolled formula produces different floating-point rounding
//! at the near-zero-distance boundary -- confirmed on real `gparis_simulation`
//! observed data: of 13293 raw consecutive-row pairs, this crate's own
//! `util::haversine_km` classified 503 fewer of them as exactly zero-length
//! than `fastmob_core`'s kernel does (10832 vs 10329 non-zero jumps), which is
//! large enough to visibly shift the jump-length ECDF. `fastmob-rs` is pinned
//! to the same kernel *and* the same cargo features as the Python extension, so
//! it reproduces Python's output bit-for-bit; its test suite asserts exactly
//! that against dumps generated from the Python path.
//!
//! **Not yet ported**: the road-network-aware variant
//! (`fastmob.measures.individual.network_distance`, used when
//! `comparison.road_network_distance` is enabled and a road graph exists) --
//! most experiments in this repo don't have a road graph built yet
//! (`road_network_available: false`), so straight-line Haversine (this
//! module) covers the common case; the road-aware path should use the
//! `fastmob-core` routing primitives when needed.

use super::filters::{FilterMeta, filter_df};
use super::panel::{AdaptationMode, adapt_evaluation_dataframe};
use polars::prelude::*;
use rustc_hash::FxHashMap;

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
) -> anyhow::Result<FxHashMap<String, JumpsRog>> {
    let mut out = FxHashMap::default();
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
/// `build()` closure for a single (already filtered/adapted) dataframe.
///
/// `fastmob_rs::prepare` cleans and arranges the frame once; both measures then
/// read the same arrangement, so the expensive part is paid a single time per
/// call rather than once per measure.
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

    let prepared = fastmob_rs::prepare(
        df,
        fastmob_rs::Cols::auto()
            .uid(uid_col)
            .lat(lat_col)
            .lng(lng_col)
            .datetime(datetime_col),
    )?;
    if prepared.is_empty() {
        return Ok(JumpsRog {
            jumps: Vec::new(),
            rog: Vec::new(),
        });
    }

    let jumps = fastmob_rs::jump_lengths_flat(&prepared)?
        .into_iter()
        .filter(|d| *d > 0.0)
        .collect();
    let (rog_all, valid) = fastmob_rs::radius_of_gyration_flat(&prepared);
    let rog = rog_all
        .into_iter()
        .zip(valid)
        .filter_map(|(value, is_valid)| is_valid.then_some(value))
        .collect();
    Ok(JumpsRog { jumps, rog })
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

    /// Cross-checked against the legacy Python backend's
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
