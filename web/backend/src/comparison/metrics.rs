//! Numeric primitives `comparison.py` imports from `fastmob`.
//!
//! Every one of these now delegates to `fastmob-rs`, fastmob's public Rust
//! DataFrame API, rather than re-deriving the formula or the data prep here.
//! That includes the pieces fastmob previously had only in Python: the
//! Jensen-Shannon divergence lives in `fastmob-core` now, so this crate and the
//! Python layer share one definition instead of two that have to be kept in
//! step by hand.
//!
//! The functions keep their original signatures so call sites are unaffected.

use polars::prelude::*;

/// Wasserstein distance between two samples, ignoring non-finite values and
/// returning `NaN` when either side ends up empty.
pub fn wasserstein_distance(values1: &[f64], values2: &[f64]) -> f64 {
    fastmob_rs::wasserstein_distance(values1, values2)
}

/// Jensen-Shannon divergence between two distributions, normalising first.
pub fn jensen_shannon_divergence(
    distribution1: &[f64],
    distribution2: &[f64],
) -> anyhow::Result<f64> {
    Ok(fastmob_rs::jensen_shannon_divergence(
        distribution1,
        distribution2,
    )?)
}

/// Mean per-column (time-bin) JSD between two `[n_categories, n_bins]`
/// matrices.
///
/// Callers must already have aligned both matrices to the same category rows;
/// `daily_activity_distribution` always produces both over the same
/// catalog-wide category set, so the Python version's `categories1`/
/// `categories2` re-alignment is not needed here.
pub fn time_bin_matrix_jsd(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> anyhow::Result<f64> {
    Ok(fastmob_rs::time_bin_matrix_jsd(matrix1, matrix2)?)
}

/// Mirrors `comparison.py::_common_part_of_commuters` /
/// `trajectory_common_part_of_commuters_multi`: CPC at several H3 resolutions.
///
/// Takes the two frames directly rather than pre-extracted arrays: each side is
/// arranged once by `fastmob_rs::prepare` and that single arrangement is shared
/// across every requested resolution.
#[allow(clippy::too_many_arguments)]
pub fn common_part_of_commuters(
    df_a: &DataFrame,
    cols_a: TrajectoryColumns<'_>,
    df_b: &DataFrame,
    cols_b: TrajectoryColumns<'_>,
    resolutions: &[u8],
) -> anyhow::Result<Vec<(u8, f64)>> {
    let a = fastmob_rs::prepare(df_a, cols_a.into())?;
    let b = fastmob_rs::prepare(df_b, cols_b.into())?;
    Ok(fastmob_rs::common_part_of_commuters_multi(
        &a,
        &b,
        resolutions,
    )?)
}

/// The four trajectory column names, grouped so multi-frame calls stay legible.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryColumns<'a> {
    pub uid: &'a str,
    pub lat: &'a str,
    pub lng: &'a str,
    pub datetime: &'a str,
}

impl From<TrajectoryColumns<'_>> for fastmob_rs::Cols {
    fn from(value: TrajectoryColumns<'_>) -> Self {
        fastmob_rs::Cols::auto()
            .uid(value.uid)
            .lat(value.lat)
            .lng(value.lng)
            .datetime(value.datetime)
    }
}

/// Mirrors `comparison.py::waiting_times_minutes`: per-user consecutive
/// timestamp differences in minutes, flattened across all users.
///
/// Uses `prepare_temporal` rather than `prepare` because this measure never
/// reads coordinates, and dropping a fix with an unusable one would fuse the
/// gaps on either side of it into a single longer gap. The frame need not carry
/// coordinate columns at all.
pub fn waiting_times_minutes(
    df: &DataFrame,
    uid_col: &str,
    datetime_col: &str,
) -> anyhow::Result<Vec<f64>> {
    let prepared = fastmob_rs::prepare_temporal(
        df,
        fastmob_rs::Cols::auto().uid(uid_col).datetime(datetime_col),
    )?;
    Ok(fastmob_rs::waiting_times_flat(&prepared)?
        .into_iter()
        .map(|seconds| seconds / 60.0)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasserstein_matches_mean_abs_difference_for_equal_sizes() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 3.0, 4.0];
        assert!((wasserstein_distance(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn wasserstein_empty_input_is_nan() {
        assert!(wasserstein_distance(&[], &[1.0]).is_nan());
    }

    #[test]
    fn jsd_identical_distributions_is_zero() {
        let p = [1.0, 2.0, 3.0, 4.0];
        assert!(jensen_shannon_divergence(&p, &p).unwrap().abs() < 1e-12);
    }

    #[test]
    fn jsd_disjoint_distributions_is_ln2() {
        let p = [1.0, 0.0];
        let q = [0.0, 1.0];
        let d = jensen_shannon_divergence(&p, &q).unwrap();
        assert!((d - std::f64::consts::LN_2).abs() < 1e-9, "got {d}");
    }

    #[test]
    fn waiting_times_computes_consecutive_diffs_per_user() {
        let df = df![
            "uid" => [1i64, 1, 1, 2, 2],
            "dt" => [
                "2026-01-01T00:00:00", "2026-01-01T00:10:00", "2026-01-01T00:25:00",
                "2026-01-01T01:00:00", "2026-01-01T01:05:00",
            ],
        ]
        .unwrap();
        let mut waits = waiting_times_minutes(&df, "uid", "dt").unwrap();
        waits.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(waits.len(), 3);
        assert!((waits[0] - 5.0).abs() < 1e-6);
        assert!((waits[1] - 10.0).abs() < 1e-6);
        assert!((waits[2] - 15.0).abs() < 1e-6);
    }

    /// Waiting times must not silently lengthen when a fix has no coordinates:
    /// the row still marks a real observation time.
    #[test]
    fn waiting_times_keep_rows_with_missing_coordinates() {
        let df = df![
            "uid" => [1i64, 1, 1],
            "lat" => [Some(48.85), None, Some(48.87)],
            "lng" => [Some(2.35), Some(2.36), Some(2.37)],
            "dt" => [
                "2026-01-01T00:00:00", "2026-01-01T00:10:00", "2026-01-01T00:20:00",
            ],
        ]
        .unwrap();
        let mut waits = waiting_times_minutes(&df, "uid", "dt").unwrap();
        waits.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(waits, vec![10.0, 10.0]);
    }

    #[test]
    fn cpc_of_a_trajectory_against_itself_is_one() {
        let df = df![
            "uid" => [1i64, 1, 2, 2],
            "lat" => [48.85, 48.86, 40.71, 40.72],
            "lng" => [2.35, 2.36, -74.01, -74.00],
            "dt" => [
                "2026-01-01T00:00:00", "2026-01-01T01:00:00",
                "2026-01-01T00:00:00", "2026-01-01T01:00:00",
            ],
        ]
        .unwrap();
        let cols = TrajectoryColumns {
            uid: "uid",
            lat: "lat",
            lng: "lng",
            datetime: "dt",
        };
        let cpc = common_part_of_commuters(&df, cols, &df, cols, &[7, 8]).unwrap();
        assert_eq!(cpc.len(), 2);
        for (resolution, value) in cpc {
            assert!((value - 1.0).abs() < 1e-12, "resolution {resolution}: {value}");
        }
    }
}
