//! Mirrors `comparison.py::_micro_activity_daily_usage_data`: converts
//! per-activity `arrival`/`departure` intervals (which may cross midnight)
//! into a `(activity_id x time-bin)` percentage-of-time matrix. A pure
//! per-row overlap loop in the Python source (flagged there as a potential
//! hotspot, not vectorized) -- ported directly as a loop here too, since
//! it's the same shape of computation either way.

use crate::settings::catalog;
use chrono::{NaiveDateTime, TimeDelta};
use polars::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct MicroActivitySeries {
    pub activity_id: i64,
    pub name: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MicroActivityUsage {
    pub bin_size_minutes: i64,
    pub n_bins: i64,
    pub x: Vec<String>,
    pub series: Vec<MicroActivitySeries>,
}

/// Mirrors `comparison.py::_micro_activity_daily_usage_data`.
pub fn micro_activity_daily_usage_data(
    activities: &DataFrame,
    bin_size_minutes: i64,
) -> anyhow::Result<MicroActivityUsage> {
    let required = ["uid", "activity", "arrival", "departure"];
    let existing: Vec<&str> = activities
        .get_column_names()
        .iter()
        .map(|s| s.as_str())
        .collect();
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|c| !existing.contains(c))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!("activities table missing columns: {}", missing.join(", "));
    }
    if bin_size_minutes <= 0 || 1440 % bin_size_minutes != 0 {
        anyhow::bail!("bin_size_minutes must be a positive divisor of 1440");
    }

    let schema = activities.schema();
    let arrival_expr = super::util::to_datetime_expr(&schema, "arrival");
    let departure_expr = super::util::to_datetime_expr(&schema, "departure");
    let work = activities
        .clone()
        .lazy()
        .select([
            col("uid"),
            col("activity").cast(DataType::Float64),
            col("arrival"),
            col("departure"),
        ])
        .with_columns([
            arrival_expr.alias("arrival"),
            departure_expr.alias("departure"),
        ])
        .drop_nulls(Some(cols(["arrival", "departure", "activity"])))
        .filter(col("departure").gt(col("arrival")))
        .collect()?;
    if work.height() == 0 {
        anyhow::bail!("activities table has no valid intervals");
    }

    let n_bins = (1440 / bin_size_minutes) as usize;
    let bin_seconds = bin_size_minutes * 60;
    let activity_ids: Vec<i64> = catalog::CATALOG.iter().map(|a| a.idx as i64).collect();
    let labels: std::collections::HashMap<i64, &str> = catalog::CATALOG
        .iter()
        .map(|a| (a.idx as i64, a.name))
        .collect();
    let mut id_to_row: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for (row, &id) in activity_ids.iter().enumerate() {
        id_to_row.insert(id, row);
    }

    let mut seconds = vec![vec![0.0f64; n_bins]; activity_ids.len()];

    let activity_ca = work.column("activity")?.f64()?;
    let arrival_ca = work.column("arrival")?.datetime()?.clone();
    let departure_ca = work.column("departure")?.datetime()?.clone();
    let arrival_unit = arrival_ca.time_unit();
    let departure_unit = departure_ca.time_unit();

    for i in 0..work.height() {
        let (Some(activity), Some(arrival_raw), Some(departure_raw)) = (
            activity_ca.get(i),
            arrival_ca.phys.get(i),
            departure_ca.phys.get(i),
        ) else {
            continue;
        };
        let activity_id = activity as i64;
        let Some(&row) = id_to_row.get(&activity_id) else {
            continue;
        };
        let mut current = datetime_from_units(arrival_raw, arrival_unit);
        let end = datetime_from_units(departure_raw, departure_unit);

        while current < end {
            let midnight = current.date().and_hms_opt(0, 0, 0).unwrap();
            let next_midnight = midnight + TimeDelta::days(1);
            let segment_end = end.min(next_midnight);
            let start_second = (current - midnight).num_seconds();
            let end_second = (segment_end - midnight).num_seconds();
            let start_bin = (start_second / bin_seconds) as usize;
            let end_bin = ((end_second - 1).div_euclid(bin_seconds)).max(start_bin as i64) as usize;

            for bin_idx in start_bin..(end_bin + 1).min(n_bins) {
                let bin_start = midnight + TimeDelta::seconds((bin_idx as i64) * bin_seconds);
                let bin_end = bin_start + TimeDelta::seconds(bin_seconds);
                let overlap = (segment_end.min(bin_end) - current.max(bin_start))
                    .num_microseconds()
                    .unwrap_or(0) as f64
                    / 1_000_000.0;
                if overlap > 0.0 {
                    seconds[row][bin_idx] += overlap;
                }
            }
            current = segment_end;
        }
    }

    let mut totals = vec![0.0f64; n_bins];
    for bin in 0..n_bins {
        totals[bin] = seconds.iter().map(|row| row[bin]).sum();
    }
    let mut percentages = vec![vec![0.0f64; n_bins]; activity_ids.len()];
    for (r, row) in seconds.iter().enumerate() {
        for bin in 0..n_bins {
            if totals[bin] > 0.0 {
                percentages[r][bin] = round6(row[bin] * 100.0 / totals[bin]);
            }
        }
    }

    let x: Vec<String> = (0..1440)
        .step_by(bin_size_minutes as usize)
        .map(|minute| format!("{:02}:{:02}", minute / 60, minute % 60))
        .collect();

    let series = activity_ids
        .iter()
        .enumerate()
        .map(|(row, &id)| MicroActivitySeries {
            activity_id: id,
            name: labels[&id].to_string(),
            values: percentages[row].clone(),
        })
        .collect();

    Ok(MicroActivityUsage {
        bin_size_minutes,
        n_bins: n_bins as i64,
        x,
        series,
    })
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

fn datetime_from_units(raw: i64, unit: TimeUnit) -> NaiveDateTime {
    match unit {
        TimeUnit::Milliseconds => chrono::DateTime::from_timestamp_millis(raw)
            .unwrap()
            .naive_utc(),
        TimeUnit::Microseconds => chrono::DateTime::from_timestamp_micros(raw)
            .unwrap()
            .naive_utc(),
        TimeUnit::Nanoseconds => chrono::DateTime::from_timestamp_nanos(raw).naive_utc(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_activities() -> DataFrame {
        df![
            "uid" => [1i64, 1],
            "activity" => [0i64, 3], // sleep, paidwork
            "arrival" => ["2026-01-01T22:00:00", "2026-01-02T09:00:00"],
            "departure" => ["2026-01-02T07:00:00", "2026-01-02T17:00:00"],
        ]
        .unwrap()
    }

    #[test]
    fn overnight_interval_splits_across_midnight() {
        let usage = micro_activity_daily_usage_data(&sample_activities(), 60).unwrap();
        assert_eq!(usage.n_bins, 24);
        let sleep = usage.series.iter().find(|s| s.name == "sleep").unwrap();
        // Sleep occupies 22:00-24:00 (2h) on day 1 and 00:00-07:00 (7h) on day 2,
        // but each day is binned independently -- both contribute 100% to their
        // respective hour bins since sleep is the only activity in those bins.
        assert!((sleep.values[22] - 100.0).abs() < 1e-6);
        assert!((sleep.values[0] - 100.0).abs() < 1e-6);
        assert!((sleep.values[8] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_invalid_bin_size() {
        assert!(micro_activity_daily_usage_data(&sample_activities(), 7).is_err());
    }

    #[test]
    fn rejects_missing_columns() {
        let df = df!["uid" => [1i64]].unwrap();
        assert!(micro_activity_daily_usage_data(&df, 10).is_err());
    }

    /// Cross-checked against the live Python backend's
    /// `_micro_activity_daily_usage_data(activities, bin_size_minutes=10)`
    /// on the real gparis `_activities.parquet` sidecar (239352 rows):
    /// 144 bins; `sleep` values[0..10] and total sum, `paidwork` values[80..90].
    #[test]
    #[ignore = "requires repo data at data/gparis/results/gparis_simulation_core_trajectories_20260710T073952_activities.parquet"]
    fn gparis_activities_match_python_reference() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let path = repo_root.join("data/gparis/results/gparis_simulation_core_trajectories_20260710T073952_activities.parquet");
        let activities = super::super::trajectory::read_parquet(&path).unwrap();
        assert_eq!(activities.height(), 239352);

        let usage = micro_activity_daily_usage_data(&activities, 10).unwrap();
        assert_eq!(usage.n_bins, 144);

        let sleep = usage.series.iter().find(|s| s.name == "sleep").unwrap();
        let expected_sleep_head = [
            19.143079, 20.025651, 22.187587, 24.522746, 26.597095, 28.342698, 30.049556, 31.72181,
            33.229111, 34.66327,
        ];
        for (got, want) in sleep.values[..10].iter().zip(expected_sleep_head.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
        let sleep_sum: f64 = sleep.values.iter().sum();
        assert!(
            (sleep_sum - 3084.517315).abs() < 1e-3,
            "sleep_sum={sleep_sum}"
        );

        let paidwork = usage.series.iter().find(|s| s.name == "paidwork").unwrap();
        let expected_paidwork_mid = [
            33.533254, 33.120571, 31.780175, 31.830254, 31.621317, 31.612286, 31.812222, 31.336889,
            29.653746, 29.53027,
        ];
        for (got, want) in paidwork.values[80..90]
            .iter()
            .zip(expected_paidwork_mid.iter())
        {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
    }
}
