//! Mirrors `comparison.py::_stvd_hourly_histogram`, `_diff_stvd_layers`,
//! and `_compute_stvd_layers` -- the Spatio-Temporal Visitation Distribution
//! diff map (per-H3-cell volume-difference/peak-shift GeoJSON), tier 1
//! (per-trajectory hourly binning) and tier 2 (diffing two already-binned
//! tables) of the computation.

use super::h3::h3_cells;
use super::util::to_datetime_expr;
use h3o::CellIndex;
use polars::prelude::*;
use std::collections::HashMap;

pub const STVD_ALL_HOURS: [i32; 24] = {
    let mut hours = [0i32; 24];
    let mut i = 0;
    while i < 24 {
        hours[i] = i as i32;
        i += 1;
    }
    hours
};

/// Per-cell hourly row counts for one trajectory at one H3 resolution --
/// `cell -> [count_hour_0, count_hour_1, ..., count_hour_23]`.
pub type HourlyLayer = HashMap<u64, [i64; 24]>;

/// Mirrors `comparison.py::_stvd_hourly_histogram`: per-H3-cell,
/// per-hour-of-day row count, one table per resolution.
pub fn stvd_hourly_histogram(
    df: &DataFrame,
    lat_col: &str,
    lng_col: &str,
    datetime_col: &str,
    resolutions: &[u8],
) -> anyhow::Result<HashMap<u8, HourlyLayer>> {
    let schema = df.schema();
    let dt_expr = to_datetime_expr(&schema, datetime_col);
    let work = df
        .clone()
        .lazy()
        .select([col(lat_col), col(lng_col), dt_expr.alias("_dt")])
        .drop_nulls(Some(cols(["_dt", lat_col, lng_col])))
        .with_columns([col("_dt").dt().hour().cast(DataType::Int32).alias("_hour")])
        .collect()?;

    let mut layers = HashMap::new();
    for &res in resolutions {
        let cells = h3_cells(
            work.column(lat_col)?.as_materialized_series(),
            work.column(lng_col)?.as_materialized_series(),
            res,
        )?;
        let cell_ca = cells.u64()?;
        let hour_ca = work.column("_hour")?.i32()?;

        let mut layer: HourlyLayer = HashMap::new();
        for i in 0..work.height() {
            if let (Some(cell), Some(hour)) = (cell_ca.get(i), hour_ca.get(i)) {
                let entry = layer.entry(cell).or_insert([0i64; 24]);
                entry[hour as usize] += 1;
            }
        }
        layers.insert(res, layer);
    }
    Ok(layers)
}

#[derive(Debug, Clone)]
pub struct StvdFeature {
    pub cell_hex: String,
    /// `[[lng, lat], ...]` ring, closed (first point repeated at the end).
    pub ring: Vec<[f64; 2]>,
    pub volume_diff_pct: f64,
    pub peak_shift_hours: f64,
}

/// Mirrors `comparison.py::_diff_stvd_layers`: volume-diff / peak-shift
/// classification + GeoJSON ring emission from two already-binned
/// per-trajectory hourly tables.
///
/// `peak_shift_hours` intentionally reproduces the Python formula's exact
/// (non-standard-circular-distance) behavior verbatim: for `raw_shift <= 12`
/// it's `min(raw_shift, 12 - raw_shift)`; for `raw_shift > 12` it's just
/// `raw_shift` clamped to a ceiling of `12.0` -- not the `min(raw_shift,
/// 24-raw_shift)` a true 24-hour circular distance would compute.
pub fn diff_stvd_layers(
    syn_hourly: &HashMap<u8, HourlyLayer>,
    real_hourly: &HashMap<u8, HourlyLayer>,
    resolutions: &[u8],
) -> anyhow::Result<HashMap<u8, Vec<StvdFeature>>> {
    let zero_row = [0i64; 24];
    let mut layers = HashMap::new();

    for &res in resolutions {
        let syn_lookup = syn_hourly.get(&res).cloned().unwrap_or_default();
        let real_lookup = real_hourly.get(&res).cloned().unwrap_or_default();
        let mut all_cells: Vec<u64> = syn_lookup
            .keys()
            .chain(real_lookup.keys())
            .copied()
            .collect();
        all_cells.sort_unstable();
        all_cells.dedup();

        let mut features = Vec::with_capacity(all_cells.len());
        for cell in all_cells {
            let syn_row = syn_lookup.get(&cell).unwrap_or(&zero_row);
            let real_row = real_lookup.get(&cell).unwrap_or(&zero_row);

            let syn_vol: f64 = syn_row.iter().sum::<i64>() as f64;
            let real_vol: f64 = real_row.iter().sum::<i64>() as f64;
            // `Iterator::max_by_key` keeps the *last* maximal element on a
            // tie; Python's `max(_STVD_ALL_HOURS, key=...)` keeps the
            // *first* (ascending-hour) one. Iterating in reverse before
            // `max_by_key` flips that to match: the smallest tied hour ends
            // up processed last, so it's the one kept.
            let syn_peak = if syn_vol > 0.0 {
                STVD_ALL_HOURS
                    .iter()
                    .copied()
                    .rev()
                    .max_by_key(|&h| syn_row[h as usize])
                    .unwrap_or(0)
            } else {
                0
            };
            let real_peak = if real_vol > 0.0 {
                STVD_ALL_HOURS
                    .iter()
                    .copied()
                    .rev()
                    .max_by_key(|&h| real_row[h as usize])
                    .unwrap_or(0)
            } else {
                0
            };

            let volume_diff_pct = (syn_vol - real_vol) / real_vol.max(1.0) * 100.0;
            let raw_shift = (syn_peak - real_peak).abs();
            let candidate2 = if raw_shift <= 12 {
                12 - raw_shift
            } else {
                raw_shift
            };
            let peak_shift_hours = (raw_shift.min(candidate2) as f64).min(12.0);

            let cell_index = CellIndex::try_from(cell)
                .map_err(|e| anyhow::anyhow!("invalid H3 cell {cell}: {e}"))?;
            let boundary = cell_index.boundary();
            let mut ring: Vec<[f64; 2]> = boundary.iter().map(|ll| [ll.lng(), ll.lat()]).collect();
            if let Some(&first) = ring.first() {
                ring.push(first);
            }

            features.push(StvdFeature {
                cell_hex: format!("{cell:x}"),
                ring,
                volume_diff_pct: (volume_diff_pct * 10000.0).round() / 10000.0,
                peak_shift_hours: (peak_shift_hours * 10000.0).round() / 10000.0,
            });
        }
        layers.insert(res, features);
    }
    Ok(layers)
}

/// Mirrors `comparison.py::_compute_stvd_layers`: thin composition of the
/// tier-1/tier-2 functions above for callers that don't need per-filter-group
/// caching.
pub fn compute_stvd_layers(
    syn_df: &DataFrame,
    syn_lat_col: &str,
    syn_lng_col: &str,
    syn_datetime_col: &str,
    real_df: &DataFrame,
    real_lat_col: &str,
    real_lng_col: &str,
    real_datetime_col: &str,
    resolutions: &[u8],
) -> anyhow::Result<HashMap<u8, Vec<StvdFeature>>> {
    let syn_hourly = stvd_hourly_histogram(
        syn_df,
        syn_lat_col,
        syn_lng_col,
        syn_datetime_col,
        resolutions,
    )?;
    let real_hourly = stvd_hourly_histogram(
        real_df,
        real_lat_col,
        real_lng_col,
        real_datetime_col,
        resolutions,
    )?;
    diff_stvd_layers(&syn_hourly, &real_hourly, resolutions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        df![
            "lat" => [48.85, 48.85, 48.85, 48.86],
            "lng" => [2.35, 2.35, 2.35, 2.36],
            "dt" => [
                "2026-01-01T08:00:00", "2026-01-01T08:30:00",
                "2026-01-01T18:00:00", "2026-01-01T09:00:00",
            ],
        ]
        .unwrap()
    }

    #[test]
    fn hourly_histogram_counts_per_cell_per_hour() {
        let layers = stvd_hourly_histogram(&sample_df(), "lat", "lng", "dt", &[9]).unwrap();
        let layer = &layers[&9];
        let total: i64 = layer.values().flat_map(|row| row.iter()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn diff_layers_zero_shift_for_identical_trajectories() {
        let df = sample_df();
        let syn = stvd_hourly_histogram(&df, "lat", "lng", "dt", &[9]).unwrap();
        let real = syn.clone();
        let diff = diff_stvd_layers(&syn, &real, &[9]).unwrap();
        for feature in &diff[&9] {
            assert_eq!(feature.volume_diff_pct, 0.0);
            assert_eq!(feature.peak_shift_hours, 0.0);
            assert!(feature.ring.len() >= 2);
            assert_eq!(feature.ring.first(), feature.ring.last());
        }
    }

    #[test]
    fn peak_shift_formula_matches_python_verbatim() {
        // raw_shift=13 (>12) -> candidate2=13 -> min(13,13)=13, then clamp to 12.0.
        // raw_shift=6 (<=12) -> candidate2=6 -> min(6,6)=6.
        // raw_shift=10 (<=12) -> candidate2=2 -> min(10,2)=2.
        for (raw_shift, expected) in [(13i32, 12.0f64), (6, 6.0), (10, 2.0), (0, 0.0), (12, 0.0)] {
            let candidate2 = if raw_shift <= 12 {
                12 - raw_shift
            } else {
                raw_shift
            };
            let peak_shift_hours = (raw_shift.min(candidate2) as f64).min(12.0);
            assert_eq!(peak_shift_hours, expected, "raw_shift={raw_shift}");
        }
    }

    /// Cross-checked against the live Python backend's `_compute_stvd_layers`
    /// on the real gparis synthetic trajectory + observed data at H3
    /// resolution 7: 908 features; sorted by cell hex, the first three are
    /// `871fb0116ffffff` (volume_diff_pct=-100.0, peak_shift_hours=12.0),
    /// `871fb0121ffffff` (700.0, 3.0), `871fb012dffffff` (1300.0, 6.0), with
    /// boundary rings starting at `[2.698163878468575, 48.41851708165447]`,
    /// `[2.786233140172542, 48.515507217271185]`,
    /// `[2.8461272899312875, 48.51781644860264]` respectively.
    #[test]
    #[ignore = "requires repo data at data/gparis/results/... and data/gparis/gparis_visitation_df.parquet"]
    fn gparis_stvd_layers_match_python_reference() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let syn_path = repo_root.join(
            "data/gparis/results/gparis_simulation_core_trajectories_20260710T073952.parquet",
        );
        let obs_path = repo_root.join("data/gparis/gparis_visitation_df.parquet");

        let traj = super::super::trajectory::load_trajectory(&syn_path).unwrap();
        let obs_df = super::super::trajectory::read_parquet(&obs_path).unwrap();

        let layers = compute_stvd_layers(
            &traj.df,
            &traj.lat_col,
            &traj.lng_col,
            &traj.datetime_col,
            &obs_df,
            "lat",
            "lon",
            "start_timestamp",
            &[7],
        )
        .unwrap();
        let mut features = layers[&7].clone();
        features.sort_by(|a, b| a.cell_hex.cmp(&b.cell_hex));
        assert_eq!(features.len(), 908);

        let expected = [
            (
                "871fb0116ffffff",
                -100.0f64,
                12.0f64,
                [2.698163878468575f64, 48.41851708165447],
            ),
            (
                "871fb0121ffffff",
                700.0,
                3.0,
                [2.786233140172542, 48.515507217271185],
            ),
            (
                "871fb012dffffff",
                1300.0,
                6.0,
                [2.8461272899312875, 48.51781644860264],
            ),
        ];
        for (feature, (hex, vol, shift, first_coord)) in features.iter().zip(expected.iter()) {
            assert_eq!(&feature.cell_hex, hex);
            assert!(
                (feature.volume_diff_pct - vol).abs() < 1e-9,
                "{hex} volume_diff_pct"
            );
            assert!(
                (feature.peak_shift_hours - shift).abs() < 1e-9,
                "{hex} peak_shift_hours"
            );
            assert!(
                (feature.ring[0][0] - first_coord[0]).abs() < 1e-9,
                "{hex} ring lng"
            );
            assert!(
                (feature.ring[0][1] - first_coord[1]).abs() < 1e-9,
                "{hex} ring lat"
            );
        }
    }
}
