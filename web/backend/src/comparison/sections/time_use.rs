//! `time-use` chart section (and its `metrics.time_use` JSD-style summary
//! row, consumed by `sections::metrics`): compares synthetic micro-activity
//! durations against an observed MTUS-shaped time-use table (CSV/Parquet;
//! `.dta` must be pre-converted, see `scripts/convert_mtus_time_use.py`).
//! Mirrors `legacy Python backend/payload/legacy.py`'s time-use section.

use crate::comparison::trajectory::read_parquet;
use crate::datasource::quote_path;
use crate::payload::{ComparisonContext, choose_regular_filter, empty_chart_payload};
use chrono::{Datelike, NaiveDateTime};
use polars::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

/// Mirrors `legacy.py`'s `TIME_USE_CATEGORIES` exactly -- the 25 raw MTUS
/// harmonized activity codes, not a coarse rollup (there is no 9-category
/// grouping anywhere in Python's real time-use comparison; a prior version
/// of this port invented one and it silently dropped every visit whose
/// activity fell outside those 9 names). Conveniently, these are also
/// citybehavex's own native activity-catalog names (`settings::catalog`),
/// since the simulation's activity taxonomy was modeled directly on MTUS.
const TIME_USE_CATEGORIES: &[&str] = &[
    "sleep", "eatdrink", "selfcare", "paidwork", "educatn", "foodprep", "cleanetc", "maintain",
    "shopserv", "garden", "petcare", "eldcare", "pkidcare", "ikidcare", "religion", "volorgwk",
    "commute", "travel", "sportex", "tvradio", "read", "compint", "goout", "leisure", "missing",
];

fn sql_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn resolve_time_use_path(path: &Path) -> Option<PathBuf> {
    if path.exists()
        && !path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("dta"))
    {
        return Some(path.to_path_buf());
    }
    if path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("dta"))
    {
        let parquet = path.with_extension("parquet");
        if parquet.exists() {
            return Some(parquet);
        }
        let csv = path.with_extension("csv");
        if csv.exists() {
            return Some(csv);
        }
    }
    None
}

fn duckdb_scan_expr(path: &Path) -> anyhow::Result<String> {
    let quoted = quote_path(path);
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("parquet") => Ok(format!("read_parquet('{quoted}')")),
        Some("csv") => Ok(format!("read_csv_auto('{quoted}')")),
        other => anyhow::bail!("unsupported time-use table format: {other:?}"),
    }
}

fn time_use_day_group_from_label(value: &str) -> Option<&'static str> {
    let raw = value.trim().to_ascii_lowercase();
    match raw.as_str() {
        "saturday" | "sat" | "6" | "sunday" | "sun" | "7" => Some("weekend"),
        "monday" | "mon" | "1" | "tuesday" | "tue" | "2" | "wednesday" | "wed" | "3"
        | "thursday" | "thu" | "4" | "friday" | "fri" | "5" => Some("weekday"),
        _ => None,
    }
}

fn observed_time_use_minutes(
    path: &Path,
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<FxHashMap<String, f64>> {
    let scan = duckdb_scan_expr(path)?;
    let conn = duckdb::Connection::open_in_memory()?;
    let columns: HashSet<String> = conn
        .prepare(&format!("DESCRIBE SELECT * FROM {scan}"))?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<HashSet<_>, _>>()?;
    let mut required: Vec<&str> = TIME_USE_CATEGORIES.to_vec();
    required.push("day");
    required.push(ctx.time_use_weight_col.as_str());
    for col_name in required {
        if !columns.contains(col_name) {
            anyhow::bail!("observed time-use table missing required column {col_name:?}");
        }
    }
    let mut where_clauses = Vec::<String>::new();
    if let Some(country) = &ctx.time_use_country {
        if columns.contains("country") {
            where_clauses.push(format!(
                "CAST(country AS VARCHAR) = {}",
                sql_literal(country)
            ));
        }
    }
    if let Some(survey) = ctx.time_use_survey {
        if columns.contains("survey") {
            where_clauses.push(format!("TRY_CAST(survey AS BIGINT) = {survey}"));
        }
    }
    let where_sql = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };
    let mut select_cols = vec![
        "CAST(day AS VARCHAR) AS day".to_string(),
        format!(
            "TRY_CAST(\"{}\" AS DOUBLE) AS weight",
            ctx.time_use_weight_col.replace('"', "\"\"")
        ),
    ];
    for category in TIME_USE_CATEGORIES {
        select_cols.push(format!(
            "TRY_CAST(\"{}\" AS DOUBLE) AS \"{}\"",
            category.replace('"', "\"\""),
            category.replace('"', "\"\"")
        ));
    }
    let sql = format!("SELECT {} FROM {scan} {where_sql}", select_cols.join(", "));
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut weighted = FxHashMap::<String, f64>::default();
    let mut total_weight = 0.0;
    while let Some(row) = rows.next()? {
        let day = row.get::<_, Option<String>>(0)?.unwrap_or_default();
        let Some(day_group) = time_use_day_group_from_label(&day) else {
            continue;
        };
        if filter_key != "all" && filter_key != day_group {
            continue;
        }
        let weight = row.get::<_, Option<f64>>(1)?.unwrap_or(0.0);
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        total_weight += weight;
        for (idx, category) in TIME_USE_CATEGORIES.iter().enumerate() {
            let minutes = row.get::<_, Option<f64>>(idx + 2)?.unwrap_or(0.0);
            *weighted.entry((*category).to_string()).or_insert(0.0) += weight * minutes;
        }
    }
    if total_weight <= 0.0 {
        anyhow::bail!("observed time-use table has no positive weights after filters");
    }
    for value in weighted.values_mut() {
        *value /= total_weight;
    }
    Ok(weighted)
}

fn add_synthetic_time_use_segment(
    totals: &mut FxHashMap<String, f64>,
    agent_days: &mut BTreeSet<(i64, chrono::NaiveDate)>,
    uid: i64,
    category: &str,
    start_us: i64,
    end_us: i64,
    filter_key: &str,
) {
    let (Some(mut start), Some(end)) = (
        NaiveDateTime::from_timestamp_micros(start_us),
        NaiveDateTime::from_timestamp_micros(end_us),
    ) else {
        return;
    };
    while start < end {
        let date = start.date();
        let Some(next_midnight) = date.succ_opt().and_then(|d| d.and_hms_opt(0, 0, 0)) else {
            break;
        };
        let segment_end = end.min(next_midnight);
        let day_group = if date.weekday().number_from_monday() >= 6 {
            "weekend"
        } else {
            "weekday"
        };
        if filter_key == "all" || filter_key == day_group {
            let minutes = (segment_end - start).num_seconds() as f64 / 60.0;
            if minutes > 0.0 {
                *totals.entry(category.to_string()).or_insert(0.0) += minutes;
                agent_days.insert((uid, date));
            }
        }
        start = segment_end;
    }
}

/// Mirrors `legacy.py`'s "Mean absolute time-use share difference" metric
/// row derivation, called from `sections::metrics` when the `metrics`
/// section wants a `time_use_comparison` block folded into `metrics.time_use`.
pub fn time_use_metric_rows(time_use_comparison: &Value) -> Value {
    let rows: Vec<Value> = time_use_comparison
        .get("groups")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|group| {
            let values: Vec<f64> = group
                .get("block")
                .and_then(|v| v.get("rows"))
                .and_then(Value::as_array)?
                .iter()
                .filter_map(|row| {
                    row.get("share_of_day_difference_pct_points")
                        .and_then(Value::as_f64)
                        .map(f64::abs)
                })
                .collect();
            if values.is_empty() {
                return None;
            }
            Some(json!({
                "filter_key": group.get("filter_key").cloned().unwrap_or(json!("all")),
                "filter_label": group.get("filter_label").cloned().unwrap_or(json!("All")),
                "metric_name": "Mean absolute time-use share difference",
                "name": "Mean absolute time-use share difference",
                "value": values.iter().sum::<f64>() / values.len() as f64,
                "unit": "pct points",
            }))
        })
        .collect();
    Value::Array(rows)
}

pub fn time_use_section_payload(
    ctx: &ComparisonContext,
    filter_key: &str,
) -> anyhow::Result<Value> {
    let Some(filter) = choose_regular_filter(ctx, filter_key)? else {
        return Ok(empty_chart_payload(ctx, vec![filter_key.to_string()]));
    };
    let mut payload = empty_chart_payload(ctx, vec![filter_key.to_string()]);
    let (Some(time_use_path), Some(activities_path)) =
        (&ctx.time_use_path, &ctx.synthetic_activities_path)
    else {
        return Ok(payload);
    };
    if !time_use_path.exists() || !activities_path.exists() {
        return Ok(payload);
    }
    let Some(observed_path) = resolve_time_use_path(time_use_path) else {
        payload["warnings"] = json!([format!(
            "time_use_comparison: .dta time-use file has no same-stem CSV/Parquet conversion: {}",
            time_use_path.display()
        )]);
        return Ok(payload);
    };
    let observed_minutes = match observed_time_use_minutes(&observed_path, ctx, &filter.key) {
        Ok(value) => value,
        Err(err) => {
            payload["warnings"] = json!([format!("time_use_comparison: {err}")]);
            return Ok(payload);
        }
    };

    let activities = read_parquet(activities_path)?;
    let required = ["uid", "activity", "arrival", "departure"];
    if required.iter().any(|c| activities.column(c).is_err()) {
        payload["warnings"] =
            json!(["time_use_comparison: activities table missing required columns"]);
        return Ok(payload);
    }
    let uid = activities.column("uid")?.i64()?;
    let activity = activities.column("activity")?.cast(&DataType::Int64)?;
    let activity = activity.i64()?;
    let arrival = activities
        .column("arrival")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let departure = activities
        .column("departure")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    let arr = arrival.datetime()?;
    let dep = departure.datetime()?;
    let mut totals: FxHashMap<String, f64> = FxHashMap::default();
    let mut agent_days = std::collections::BTreeSet::<(i64, chrono::NaiveDate)>::new();
    for i in 0..activities.height() {
        let (Some(uid), Some(act), Some(a), Some(d)) = (
            uid.get(i),
            activity.get(i),
            arr.phys.get(i),
            dep.phys.get(i),
        ) else {
            continue;
        };
        if d <= a {
            continue;
        }
        let Some(def) = crate::settings::catalog::by_id(act) else {
            continue;
        };
        let category = def.name;
        if !TIME_USE_CATEGORIES.contains(&category) {
            continue;
        }
        add_synthetic_time_use_segment(
            &mut totals,
            &mut agent_days,
            uid,
            category,
            a,
            d,
            &filter.key,
        );
    }
    let denom = (agent_days.len() as f64).max(1.0);
    let rows: Vec<Value> = TIME_USE_CATEGORIES
        .iter()
        .map(|category| {
            let syn = totals.get(*category).copied().unwrap_or(0.0) / denom;
            let obs = observed_minutes.get(*category).copied().unwrap_or(0.0);
            let diff = syn - obs;
            let pct = if obs.abs() > 1e-12 {
                Value::from((diff / obs * 100.0 * 1_000_000.0).round() / 1_000_000.0)
            } else {
                Value::Null
            };
            json!({
                "category": category,
                "mtus_minutes": (obs * 1_000_000.0).round() / 1_000_000.0,
                "simulation_minutes": (syn * 1_000_000.0).round() / 1_000_000.0,
                "observed_minutes": (obs * 1_000_000.0).round() / 1_000_000.0,
                "synthetic_minutes": (syn * 1_000_000.0).round() / 1_000_000.0,
                "difference_minutes": (diff * 1_000_000.0).round() / 1_000_000.0,
                "percent_difference": pct,
                "share_of_day_difference_pct_points": ((diff / 1440.0 * 100.0) * 1_000_000.0).round() / 1_000_000.0,
            })
        })
        .collect();
    payload["time_use_comparison"] = json!({"groups": [{
        "filter_key": filter.key,
        "filter_label": filter.label,
        "block": {
            "categories": TIME_USE_CATEGORIES,
            "labels": [ctx.time_use_label, "synthetic"],
            "rows": rows,
        },
    }]});
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_use_metric_rows_average_absolute_share_difference() {
        let block = json!({
            "groups": [{
                "filter_key": "all",
                "filter_label": "All",
                "block": {"rows": [
                    {"share_of_day_difference_pct_points": -2.0},
                    {"share_of_day_difference_pct_points": 4.0}
                ]}
            }]
        });
        let rows = time_use_metric_rows(&block);
        assert_eq!(
            rows[0]["metric_name"],
            "Mean absolute time-use share difference"
        );
        assert_eq!(rows[0]["unit"], "pct points");
        assert_eq!(rows[0]["value"], 3.0);
    }
}
