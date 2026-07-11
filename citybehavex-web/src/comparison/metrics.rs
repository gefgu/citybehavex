//! Numeric primitives comparison.py imports from `fkmob`: `wasserstein_distance`
//! and `jensen_shannon_divergence` are pure-Python wrappers in fkmob's Python
//! layer around a narrow Rust kernel (or, for JSD, no Rust at all) -- ported
//! here to call `fkmob-core` directly where a kernel exists, and to
//! reimplement the plain numpy formula otherwise. `trajectory_common_part_of_commuters_multi`
//! and `waiting_times` are fully Rust-backed in fkmob-core; called directly.

use polars::prelude::*;

/// Mirrors `fkmob/measures/evaluation/metrics.py::_finite_array` +
/// `wasserstein_distance`: drop non-finite values, return `NaN` if either
/// side is empty, otherwise the Rust-backed empirical 1D Wasserstein
/// distance (`fkmob-core`'s `empirical_wasserstein_1d`: mean absolute
/// order-statistic difference when sample sizes match, else the general
/// merge-sweep CDF-area formula).
pub fn wasserstein_distance(values1: &[f64], values2: &[f64]) -> f64 {
    let v1: Vec<f64> = values1.iter().copied().filter(|v| v.is_finite()).collect();
    let v2: Vec<f64> = values2.iter().copied().filter(|v| v.is_finite()).collect();
    if v1.is_empty() || v2.is_empty() {
        return f64::NAN;
    }
    fkmob_core::measures::evaluation::wasserstein::empirical_wasserstein_1d(&v1, &v2)
        .unwrap_or(f64::NAN)
}

fn normalize_distribution(values: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = values
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect();
    let total: f64 = out.iter().sum();
    if total > 0.0 {
        for v in &mut out {
            *v /= total;
        }
    }
    out
}

/// Mirrors `fkmob/measures/evaluation/metrics.py::jensen_shannon_divergence`.
/// fkmob's Python version tries a SIMD-accelerated path first and falls back
/// to this same reference formula on mismatch; since the two only diverge
/// under numerical-stability edge cases (and the reference formula IS the
/// canonical JS-divergence definition), computing it directly here matches
/// fkmob's result for the overwhelming common case.
pub fn jensen_shannon_divergence(
    distribution1: &[f64],
    distribution2: &[f64],
) -> anyhow::Result<f64> {
    let p = normalize_distribution(distribution1);
    let q = normalize_distribution(distribution2);
    if p.len() != q.len() {
        anyhow::bail!(
            "distribution shapes must match, got {} and {}",
            p.len(),
            q.len()
        );
    }
    if p.is_empty() || (p.iter().sum::<f64>() == 0.0 && q.iter().sum::<f64>() == 0.0) {
        return Ok(0.0);
    }
    let mut left = 0.0;
    let mut right = 0.0;
    for (pi, qi) in p.iter().zip(q.iter()) {
        let m = 0.5 * (pi + qi);
        if *pi > 0.0 {
            left += pi * (pi / m).ln();
        }
        if *qi > 0.0 {
            right += qi * (qi / m).ln();
        }
    }
    Ok(0.5 * (left + right))
}

/// Mirrors `fkmob/measures/evaluation/metrics.py::time_bin_matrix_jensen_shannon_divergence`:
/// mean per-column (time-bin) JSD between two `[n_categories, n_bins]`
/// matrices, skipping columns where both sides are entirely zero. Assumes
/// callers have already aligned both matrices to the same category rows
/// (the Python version's `categories1`/`categories2` re-alignment isn't
/// needed here since `daily_activity_distribution`'s Rust port always
/// produces both matrices over the same catalog-wide category set).
pub fn time_bin_matrix_jsd(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> anyhow::Result<f64> {
    if matrix1.len() != matrix2.len() {
        anyhow::bail!("matrix1/matrix2 must have the same number of category rows");
    }
    let n_bins = matrix1.first().map(|r| r.len()).unwrap_or(0);
    if matrix2.first().map(|r| r.len()).unwrap_or(0) != n_bins {
        anyhow::bail!("Number of time bins must match");
    }
    let nan_to_zero = |v: f64| if v.is_nan() { 0.0 } else { v };
    let mut values = Vec::new();
    for col in 0..n_bins {
        let left: Vec<f64> = matrix1.iter().map(|row| nan_to_zero(row[col])).collect();
        let right: Vec<f64> = matrix2.iter().map(|row| nan_to_zero(row[col])).collect();
        if left.iter().sum::<f64>() == 0.0 && right.iter().sum::<f64>() == 0.0 {
            continue;
        }
        values.push(jensen_shannon_divergence(&left, &right)?);
    }
    if values.is_empty() {
        Ok(0.0)
    } else {
        Ok(values.iter().sum::<f64>() / values.len() as f64)
    }
}

/// Builds `(indices, ends)` for `trajectory_common_part_of_commuters_impl`:
/// a stable permutation of row positions sorted by `(uid, timestamp)`, plus
/// the cumulative per-user boundary offsets into that permutation -- mirrors
/// fkmob's Python `_build_time_ordered_user_ranges` helper.
fn time_ordered_user_ranges(uid: &[i64], timestamp_ms: &[i64]) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..uid.len()).collect();
    indices.sort_by(|&a, &b| (uid[a], timestamp_ms[a]).cmp(&(uid[b], timestamp_ms[b])));
    let mut ends = Vec::new();
    let mut i = 0;
    while i < indices.len() {
        let mut j = i + 1;
        while j < indices.len() && uid[indices[j]] == uid[indices[i]] {
            j += 1;
        }
        ends.push(j);
        i = j;
    }
    (indices, ends)
}

/// Mirrors `comparison.py::_common_part_of_commuters` /
/// `trajectory_common_part_of_commuters_multi`: CPC at several H3
/// resolutions, sharing one time-ordering pass per trajectory across all
/// requested resolutions.
pub fn common_part_of_commuters(
    lat_a: &[f64],
    lng_a: &[f64],
    uid_a: &[i64],
    ts_a_ms: &[i64],
    lat_b: &[f64],
    lng_b: &[f64],
    uid_b: &[i64],
    ts_b_ms: &[i64],
    resolutions: &[u8],
) -> anyhow::Result<Vec<(u8, f64)>> {
    let (indices_a, ends_a) = time_ordered_user_ranges(uid_a, ts_a_ms);
    let (indices_b, ends_b) = time_ordered_user_ranges(uid_b, ts_b_ms);
    resolutions
        .iter()
        .map(|&res| {
            let cpc = fkmob_core::measures::evaluation::trajectory_cpc::trajectory_common_part_of_commuters_impl(
                lat_a, lng_a, &indices_a, &ends_a,
                lat_b, lng_b, &indices_b, &ends_b,
                res,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
            Ok((res, cpc))
        })
        .collect()
}

/// Mirrors `comparison.py::waiting_times_minutes`: per-user, time-sorted
/// consecutive timestamp differences (in minutes), flattened across all
/// users (`merge=True` in the Python version). Implemented directly via
/// Polars group-by/sort/diff rather than calling fkmob-core's indexed-range
/// kernel (`waiting_times_indexed_impl`) -- same math (group by user, sort
/// by timestamp, consecutive differences, drop groups with &lt;2 points),
/// without needing to replicate fkmob's permutation-index plumbing (that
/// exists there purely to avoid a physical dataframe sort at fkmob's much
/// larger internal call volume; irrelevant for this one metric).
pub fn waiting_times_minutes(
    df: &DataFrame,
    uid_col: &str,
    datetime_col: &str,
) -> anyhow::Result<Vec<f64>> {
    let schema = df.schema();
    let dt_expr = super::util::to_datetime_expr(&schema, datetime_col);
    let sorted = df
        .clone()
        .lazy()
        .select([col(uid_col), dt_expr.alias(datetime_col)])
        .drop_nulls(None)
        .sort([uid_col, datetime_col], SortMultipleOptions::default())
        .collect()?;

    let uid = sorted.column(uid_col)?.as_materialized_series();
    let ts = sorted
        .column(datetime_col)?
        .as_materialized_series()
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let ts_us = ts.datetime()?.clone();

    let mut waits = Vec::new();
    let mut prev_uid: Option<AnyValue> = None;
    let mut prev_ts: Option<i64> = None;
    for i in 0..sorted.height() {
        let this_uid = uid.get(i)?;
        let this_ts = ts_us.phys.get(i);
        if Some(&this_uid) == prev_uid.as_ref() {
            if let (Some(p), Some(t)) = (prev_ts, this_ts) {
                waits.push((t - p) as f64 / 1_000_000.0 / 60.0);
            }
        }
        prev_uid = Some(this_uid);
        prev_ts = this_ts;
    }
    Ok(waits)
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
}
