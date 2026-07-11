//! DuckDB helpers for cheap parquet metadata, mirroring
//! `web/backend/app/datasource.py`. The heavy scientific metrics go through
//! the comparison engine (`comparison::*`); DuckDB here is only for the
//! fast, tabular work the Experiments page needs: row counts, distinct
//! users, and the datetime span of a run's parquet.

use crate::columns::{DATETIME_CANDIDATES, UID_CANDIDATES, detect_column};
use lru::LruCache;
use serde::Serialize;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::UNIX_EPOCH;

const RUN_SUMMARY_CACHE_CAPACITY: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RunSummaryCacheKey {
    path: String,
    mtime_secs: Option<u64>,
    len: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CachedRunSummary {
    pub summary: Option<RunSummary>,
    pub summary_error: Option<String>,
}

static RUN_SUMMARY_CACHE: LazyLock<Mutex<LruCache<RunSummaryCacheKey, CachedRunSummary>>> =
    LazyLock::new(|| {
        Mutex::new(LruCache::new(
            std::num::NonZeroUsize::new(RUN_SUMMARY_CACHE_CAPACITY).unwrap(),
        ))
    });

pub fn quote_path(path: &Path) -> String {
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

fn run_summary_cache_key(path: &Path) -> RunSummaryCacheKey {
    let meta = std::fs::metadata(path).ok();
    let mtime_secs = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    let len = meta.as_ref().map(|m| m.len());
    RunSummaryCacheKey {
        path: path.display().to_string(),
        mtime_secs,
        len,
    }
}

/// Cached wrapper around `run_summary`, keyed by parquet path + mtime + file
/// length. Both successes and errors are cached so broken/missing runs do not
/// repeatedly pay DuckDB setup cost while the Experiments page reloads.
pub fn cached_run_summary(path: &Path) -> CachedRunSummary {
    let key = run_summary_cache_key(path);
    if let Some(cached) = RUN_SUMMARY_CACHE.lock().unwrap().get(&key).cloned() {
        return cached;
    }

    let computed = match run_summary(path) {
        Ok(summary) => CachedRunSummary {
            summary: Some(summary),
            summary_error: None,
        },
        Err(err) => CachedRunSummary {
            summary: None,
            summary_error: Some(err.to_string()),
        },
    };
    RUN_SUMMARY_CACHE.lock().unwrap().put(key, computed.clone());
    computed
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn missing_file_summary_error_is_cached_shape() {
        let path = std::env::temp_dir().join(format!(
            "citybehavex-missing-summary-{}-{}.parquet",
            std::process::id(),
            "not-there"
        ));
        let summary = cached_run_summary(&path);
        assert!(summary.summary.is_none());
        assert!(summary.summary_error.is_some());
    }
}
