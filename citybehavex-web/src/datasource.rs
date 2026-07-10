//! DuckDB helpers for cheap parquet metadata, mirroring
//! `web/backend/app/datasource.py`. The heavy scientific metrics go through
//! the comparison engine (Phase 5); DuckDB here is only for the fast,
//! tabular work the Experiments page needs: row counts, distinct users, and
//! the datetime span of a run's parquet.

use serde::Serialize;
use std::path::Path;

/// Kept in sync with `citybehavex.reports.comparison`'s candidate lists
/// (same duplication the Python side has across `datasource.py` /
/// `reports_bridge.py` / `home_work_data.py` -- noted there as manually
/// kept in sync; same here for `comparison.rs` in Phase 5).
pub const UID_CANDIDATES: &[&str] = &["uid", "user_id", "user", "agent_id", "userid"];
pub const DATETIME_CANDIDATES: &[&str] = &[
    "datetime",
    "start_timestamp",
    "timestamp",
    "check-in_time",
    "start_time",
    "_start_time",
    "checkin_time",
    "time",
    "date",
];

/// Case-insensitive first-match column lookup, mirrors
/// `citybehavex/reports/comparison.py::detect_column`.
pub fn detect_column<'a>(columns: &'a [String], candidates: &[&str]) -> Option<&'a str> {
    for candidate in candidates {
        if let Some(found) = columns.iter().find(|c| c.eq_ignore_ascii_case(candidate)) {
            return Some(found.as_str());
        }
    }
    None
}

fn quote_path(path: &Path) -> String {
    path.display().to_string().replace('\'', "''")
}

pub fn parquet_columns(path: &Path) -> anyhow::Result<Vec<String>> {
    let conn = duckdb::Connection::open_in_memory()?;
    let sql = format!("SELECT name FROM parquet_schema('{}')", quote_path(path));
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut out = Vec::new();
    for row in rows {
        let name = row?;
        if name != "schema" && name != "duckdb_schema" {
            out.push(name);
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct RunSummary {
    pub rows: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uids: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date_start: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date_end: Option<String>,
}

/// Row count, distinct-user count, and datetime span for a run parquet.
/// Columns are auto-detected (schemas differ across cities), matching
/// `datasource.py::run_summary`.
pub fn run_summary(path: &Path) -> anyhow::Result<RunSummary> {
    let columns = parquet_columns(path)?;
    let uid_col = detect_column(&columns, UID_CANDIDATES).map(str::to_string);
    let dt_col = detect_column(&columns, DATETIME_CANDIDATES).map(str::to_string);

    let mut select = vec!["count(*) AS rows".to_string()];
    if let Some(uid_col) = &uid_col {
        select.push(format!(r#"count(DISTINCT "{uid_col}") AS uids"#));
    }
    if let Some(dt_col) = &dt_col {
        select.push(format!(r#"min("{dt_col}"::VARCHAR) AS dt_min"#));
        select.push(format!(r#"max("{dt_col}"::VARCHAR) AS dt_max"#));
    }

    let conn = duckdb::Connection::open_in_memory()?;
    let sql = format!(
        "SELECT {} FROM read_parquet('{}')",
        select.join(", "),
        quote_path(path)
    );

    let mut summary = RunSummary::default();
    let mut idx = 1; // column 0 is always `rows`
    conn.query_row(&sql, [], |row| {
        summary.rows = row.get::<_, i64>(0)?;
        if uid_col.is_some() {
            summary.uids = row.get::<_, Option<i64>>(idx)?;
            idx += 1;
        }
        if dt_col.is_some() {
            summary.date_start = row.get::<_, Option<String>>(idx)?;
            summary.date_end = row.get::<_, Option<String>>(idx + 1)?;
        }
        Ok(())
    })?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_column_matches_case_insensitively() {
        let cols = vec!["UID".to_string(), "Datetime".to_string(), "lat".to_string()];
        assert_eq!(detect_column(&cols, UID_CANDIDATES), Some("UID"));
        assert_eq!(detect_column(&cols, DATETIME_CANDIDATES), Some("Datetime"));
        assert_eq!(detect_column(&cols, &["missing"]), None);
    }

    #[test]
    fn detect_column_prefers_earlier_candidates() {
        let cols = vec!["user_id".to_string(), "uid".to_string()];
        // "uid" is listed before "user_id" in UID_CANDIDATES, so it wins
        // even though "user_id" appears first in the column list.
        assert_eq!(detect_column(&cols, UID_CANDIDATES), Some("uid"));
    }
}
