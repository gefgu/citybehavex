//! Mirrors `payload/legacy.py`'s ECDF helpers (`_downsample`, `_ecdf`,
//! `_ecdf_block`, `_transport_ecdf_block`). The underlying point-computation
//! algorithm is `skmob_vis::ecdf_points` (`skmob_vis/src/lib.rs`, exposed to
//! Python as `skmob_vis._core.compute_ecdf`) -- reimplemented directly here
//! rather than taking `skmob-vis` as a Cargo dependency, since its `[lib]
//! crate-type = ["cdylib", "rlib"]` forces a PyO3 cdylib build on any
//! consumer of the rlib (see `citybehavex-web/Cargo.toml`'s comment on this).
//! ~30 lines, easy to keep in sync if the upstream implementation changes.

use super::DEFAULT_MODE_ORDER;
use polars::prelude::*;
use serde::Serialize;

pub const MAX_ECDF_POINTS: usize = 400;

/// Mirrors `skmob_vis::ecdf_points`: empirical CDF as `[x, cumulative_probability]`
/// points (one point per distinct value, ties collapsed), truncated at
/// `cdf_cutoff` (the point where `probability == cdf_cutoff` is kept and the
/// curve stops there; if a step would exceed the cutoff, one final point at
/// exactly `cdf_cutoff` is emitted instead).
pub fn ecdf_points(values: &[f64], cdf_cutoff: f64) -> anyhow::Result<Vec<[f64; 2]>> {
    if values.is_empty() {
        anyhow::bail!("values must not be empty");
    }
    if values.iter().any(|v| !v.is_finite()) {
        anyhow::bail!("values must contain only finite values");
    }
    if !(cdf_cutoff.is_finite() && cdf_cutoff > 0.0 && cdf_cutoff <= 1.0) {
        anyhow::bail!("cdf_cutoff must be finite and in the interval (0, 1]");
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    let total = sorted.len() as f64;
    let mut points = Vec::new();
    let mut index = 0usize;
    while index < sorted.len() {
        let x = sorted[index];
        let mut next = index + 1;
        while next < sorted.len() && sorted[next] == x {
            next += 1;
        }
        let probability = next as f64 / total;
        if probability <= cdf_cutoff {
            points.push([x, probability]);
            if probability == cdf_cutoff {
                break;
            }
        } else {
            points.push([x, cdf_cutoff]);
            break;
        }
        index = next;
    }
    Ok(points)
}

/// Mirrors `legacy.py::_downsample`. Minor known deviation: numpy's
/// `.round()` uses banker's (round-half-to-even) rounding, while Rust's
/// `f64::round()` rounds half away from zero -- only matters on an exact
/// `.5` tie, which selects one adjacent index differently and doesn't
/// change the resulting curve's shape.
pub fn downsample(points: &[[f64; 2]], max_points: usize) -> Vec<[f64; 2]> {
    let n = points.len();
    if n <= max_points {
        return points.to_vec();
    }
    let mut idx: Vec<usize> = (0..max_points)
        .map(|i| {
            let t = i as f64 * (n - 1) as f64 / (max_points - 1) as f64;
            t.round() as usize
        })
        .collect();
    idx.sort_unstable();
    idx.dedup();
    idx.into_iter().map(|i| points[i]).collect()
}

/// Mirrors `legacy.py::_ecdf`: drop non-finite values, empty input -> empty
/// output (not an error, unlike `ecdf_points` itself), cutoff fixed at 0.98,
/// downsampled to `MAX_ECDF_POINTS`.
pub fn ecdf(values: &[f64]) -> Vec<[f64; 2]> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Vec::new();
    }
    let points = ecdf_points(&finite, 0.98).unwrap_or_default();
    downsample(&points, MAX_ECDF_POINTS)
}

#[derive(Debug, Clone, Serialize)]
pub struct SeriesPoints {
    pub name: String,
    pub role: String,
    pub points: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EcdfBlock {
    pub x_label: String,
    pub x_unit: String,
    pub series: Vec<SeriesPoints>,
}

/// Mirrors `legacy.py::_ecdf_block`.
pub fn ecdf_block(
    label_syn: &str,
    syn_values: &[f64],
    obs: Option<(&str, &[f64])>,
    x_label: &str,
    x_unit: &str,
) -> EcdfBlock {
    let mut series = vec![SeriesPoints { name: label_syn.to_string(), role: "synthetic".to_string(), points: ecdf(syn_values) }];
    if let Some((label_obs, obs_values)) = obs {
        series.push(SeriesPoints { name: label_obs.to_string(), role: "observed".to_string(), points: ecdf(obs_values) });
    }
    EcdfBlock { x_label: x_label.to_string(), x_unit: x_unit.to_string(), series }
}

fn mode_sort_key(m: &str) -> (usize, String) {
    (DEFAULT_MODE_ORDER.iter().position(|d| *d == m).unwrap_or(99), m.to_string())
}

/// Mirrors `legacy.py::_transport_ecdf_block`: per (source, mode) jump-length
/// ECDF series over the combined synthetic+observed transport-leg records.
pub fn transport_ecdf_block(records: &DataFrame, observed_label: &str) -> anyhow::Result<EcdfBlock> {
    let mut modes: Vec<String> = records
        .column("mode")?
        .as_materialized_series()
        .str()?
        .into_iter()
        .flatten()
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    modes.sort_by_key(|m| mode_sort_key(m));

    let mut series = Vec::new();
    for (source, label) in [("synthetic", "synthetic"), ("observed", observed_label)] {
        let source_df = records.clone().lazy().filter(col("source").eq(lit(source))).collect()?;
        if source_df.height() == 0 {
            continue;
        }
        for mode in &modes {
            let values: Vec<f64> = source_df
                .clone()
                .lazy()
                .filter(col("mode").eq(lit(mode.clone())))
                .select([col("jump_km")])
                .collect()?
                .column("jump_km")?
                .f64()?
                .into_iter()
                .flatten()
                .collect();
            if values.is_empty() {
                continue;
            }
            series.push(SeriesPoints { name: format!("{label} · {mode}"), role: source.to_string(), points: ecdf(&values) });
        }
    }
    Ok(EcdfBlock { x_label: "jump length".to_string(), x_unit: "km".to_string(), series })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecdf_points_matches_hand_computed_steps() {
        // 4 points: 1,1,2,3 -> at x=1, prob=2/4=0.5; x=2, prob=3/4=0.75; x=3, prob=1.0.
        let points = ecdf_points(&[1.0, 1.0, 2.0, 3.0], 1.0).unwrap();
        assert_eq!(points, vec![[1.0, 0.5], [2.0, 0.75], [3.0, 1.0]]);
    }

    #[test]
    fn ecdf_points_truncates_at_cutoff() {
        let points = ecdf_points(&[1.0, 2.0, 3.0, 4.0], 0.5).unwrap();
        // x=1 -> prob=0.25 (kept); x=2 -> prob=0.5 (kept, exact match, stop).
        assert_eq!(points, vec![[1.0, 0.25], [2.0, 0.5]]);
    }

    #[test]
    fn ecdf_points_clamps_overshoot_step_to_cutoff() {
        // 3 values, cutoff 0.5: x=1 -> prob=1/3=0.333 (kept); x=2 -> prob=2/3=0.667
        // which exceeds 0.5, so emit [2.0, 0.5] and stop.
        let points = ecdf_points(&[1.0, 2.0, 3.0], 0.5).unwrap();
        assert_eq!(points, vec![[1.0, 1.0 / 3.0], [2.0, 0.5]]);
    }

    #[test]
    fn ecdf_rejects_empty_and_nonfinite() {
        assert!(ecdf_points(&[], 1.0).is_err());
        assert!(ecdf_points(&[1.0, f64::NAN], 1.0).is_err());
        assert!(ecdf_points(&[1.0], 0.0).is_err());
        assert!(ecdf_points(&[1.0], 1.5).is_err());
    }

    #[test]
    fn ecdf_empty_input_is_empty_not_error() {
        assert_eq!(ecdf(&[]), Vec::<[f64; 2]>::new());
        assert_eq!(ecdf(&[f64::NAN, f64::INFINITY]), Vec::<[f64; 2]>::new());
    }

    #[test]
    fn downsample_keeps_endpoints() {
        let points: Vec<[f64; 2]> = (0..1000).map(|i| [i as f64, i as f64]).collect();
        let out = downsample(&points, 400);
        assert!(out.len() <= 400);
        assert_eq!(out.first(), Some(&[0.0, 0.0]));
        assert_eq!(out.last(), Some(&[999.0, 999.0]));
    }
}
