//! Observed daily co-presence graph construction for network validation.
//! Mirrors `citybehavex/reports/network_validation.py`'s
//! `_resolve_observed_location`, `_to_day`, and `_observed_edges_and_persistence`
//! -- the parts of the observed-validation path not already covered by
//! `fastmob-core`'s `co_presence_network` module (which supplies the actual
//! `build_co_presence_edges`/`compute_graph_metrics` kernels).
//!
//! Node/day/location identifiers are factorized via plain `FxHashMap`s rather
//! than Polars group-by/join (as the Python port does), which is simpler and
//! avoids join overhead at observed-graph scale (tens of millions of rows for
//! Shanghai/yjmob) -- the resulting node ordering isn't required to match
//! Python's, since node identity is only used as an opaque graph-metric
//! index here, never displayed.

use crate::columns::{self, detect_in};
use crate::comparison::h3::h3_cells;
use crate::comparison::util::to_datetime_expr;
use crate::settings::reports::LocationMode;
use fastmob_core::measures::collective::co_presence_network::build_co_presence_edges;
use polars::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Debug)]
pub struct ObservedGraph {
    pub node_count: usize,
    pub edge_from: Vec<u32>,
    pub edge_to: Vec<u32>,
    pub persistence: Vec<f64>,
    pub time_steps: usize,
    pub warnings: Vec<String>,
}

/// Mirrors `_resolve_observed_location`: an explicit/auto-detected location
/// column wins unless `location_mode` is `H3`, in which case lat/lng ->
/// H3-cell binning is used instead.
pub fn resolve_observed_location(
    df: &DataFrame,
    location_mode: LocationMode,
    location_col: Option<&str>,
    h3_resolution: u8,
) -> anyhow::Result<(Series, String)> {
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();

    let mut chosen: Option<String> = location_col
        .filter(|c| cols.contains(c))
        .map(|c| c.to_string());
    if chosen.is_none() && matches!(location_mode, LocationMode::Auto) {
        chosen = detect_in(&cols, columns::LOCATION_CANDIDATES);
    }
    if matches!(location_mode, LocationMode::LocationCol) && chosen.is_none() {
        anyhow::bail!(
            "network validation location_col not found: {:?}",
            location_col
        );
    }
    if let Some(name) = &chosen {
        if !matches!(location_mode, LocationMode::H3) {
            let series = df
                .column(name)?
                .as_materialized_series()
                .cast(&DataType::String)?;
            return Ok((series, name.clone()));
        }
    }

    let lat_col = detect_in(&cols, columns::LAT_CANDIDATES).ok_or_else(|| {
        anyhow::anyhow!("h3 network validation requires latitude/longitude columns")
    })?;
    let lng_col = detect_in(&cols, columns::LNG_CANDIDATES).ok_or_else(|| {
        anyhow::anyhow!("h3 network validation requires latitude/longitude columns")
    })?;
    let lat = df.column(&lat_col)?.as_materialized_series();
    let lng = df.column(&lng_col)?.as_materialized_series();
    let cells = h3_cells(lat, lng, h3_resolution)?;
    Ok((cells, format!("h3_{h3_resolution}")))
}

fn truncate_to_day(df: &DataFrame, datetime_col: &str) -> anyhow::Result<Series> {
    let schema = df.schema();
    let out = df
        .clone()
        .lazy()
        .select([to_datetime_expr(&schema, datetime_col)
            .dt()
            .truncate(lit("1d"))
            .alias("__day__")])
        .collect()?;
    Ok(out.column("__day__")?.as_materialized_series().clone())
}

/// Mirrors `_observed_edges_and_persistence`: builds the observed daily
/// co-presence graph from `(uid, datetime, location)` rows via
/// `fastmob-core`'s `co_presence_network::build_co_presence_edges`, the same
/// kernel the Python port uses through `fastmob._core`.
pub fn observed_edges_and_persistence(
    df: &DataFrame,
    uid_col: &str,
    datetime_col: &str,
    location_mode: LocationMode,
    location_col: Option<&str>,
    h3_resolution: u8,
    max_group_size: usize,
) -> anyhow::Result<ObservedGraph> {
    if max_group_size < 2 {
        anyhow::bail!("network validation max_group_size must be at least 2");
    }

    let (location_series, location_source) =
        resolve_observed_location(df, location_mode, location_col, h3_resolution)?;
    let day_series = truncate_to_day(df, datetime_col)?;
    let uid_series = df
        .column(uid_col)?
        .as_materialized_series()
        .cast(&DataType::String)?;
    let day_i64 = day_series.cast(&DataType::Int64)?;
    let location_str = location_series.cast(&DataType::String)?;

    let uid_ca = uid_series.str()?;
    let day_ca = day_i64.i64()?;
    let loc_ca = location_str.str()?;

    let mut uid_codes: FxHashMap<String, i64> = FxHashMap::default();
    let mut day_codes_map: FxHashMap<i64, i64> = FxHashMap::default();
    let mut loc_codes_map: FxHashMap<String, i64> = FxHashMap::default();

    let mut nodes: Vec<i64> = Vec::new();
    let mut day_codes: Vec<i64> = Vec::new();
    let mut location_codes: Vec<i64> = Vec::new();

    let height = df.height();
    for i in 0..height {
        let (Some(u), Some(d), Some(l)) = (uid_ca.get(i), day_ca.get(i), loc_ca.get(i)) else {
            continue;
        };
        let next = uid_codes.len() as i64;
        let node = *uid_codes.entry(u.to_string()).or_insert(next);
        let next = day_codes_map.len() as i64;
        let day_code = *day_codes_map.entry(d).or_insert(next);
        let next = loc_codes_map.len() as i64;
        let loc_code = *loc_codes_map.entry(l.to_string()).or_insert(next);
        nodes.push(node);
        day_codes.push(day_code);
        location_codes.push(loc_code);
    }

    let node_count = uid_codes.len();
    let time_steps = day_codes_map.len();

    if nodes.is_empty() {
        return Ok(ObservedGraph {
            node_count: 0,
            edge_from: Vec::new(),
            edge_to: Vec::new(),
            persistence: Vec::new(),
            time_steps: 0,
            warnings: vec![format!(
                "observed network has no valid rows using {location_source}"
            )],
        });
    }

    let (edge_from, edge_to, persistence, skipped_groups, skipped_rows) = build_co_presence_edges(
        &day_codes,
        &location_codes,
        &nodes,
        max_group_size,
        time_steps,
    );

    let mut warnings = Vec::new();
    if skipped_groups > 0 {
        warnings.push(format!(
            "observed network skipped {skipped_groups} {location_source}/day groups ({skipped_rows} user-presences) larger than max_group_size={max_group_size}"
        ));
    }

    Ok(ObservedGraph {
        node_count,
        edge_from,
        edge_to,
        persistence,
        time_steps,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        df!(
            "uid" => ["a", "b", "c", "a", "b"],
            "ts" => [
                "2024-01-01T08:00:00",
                "2024-01-01T09:00:00",
                "2024-01-01T09:30:00",
                "2024-01-02T08:00:00",
                "2024-01-02T08:30:00",
            ],
            "venue" => ["v1", "v2", "v2", "v3", "v3"],
        )
        .unwrap()
    }

    #[test]
    fn resolve_location_uses_explicit_location_col() {
        let df = sample_df();
        let (series, source) =
            resolve_observed_location(&df, LocationMode::LocationCol, Some("venue"), 9).unwrap();
        assert_eq!(source, "venue");
        assert_eq!(series.len(), 5);
    }

    #[test]
    fn location_col_mode_errors_when_column_missing() {
        let df = sample_df();
        let err =
            resolve_observed_location(&df, LocationMode::LocationCol, Some("nope"), 9).unwrap_err();
        assert!(err.to_string().contains("location_col not found"));
    }

    #[test]
    fn observed_graph_builds_one_edge_per_shared_day_venue() {
        // Day 1: b and c co-present at v2 -> one edge. Day 2: b and... wait,
        // b is at v3 on day 2 along with nobody else from day 1's group, but
        // a (v3, day2) and b (v3, day2) via distinct rows above share v3.
        let df = sample_df();
        let graph = observed_edges_and_persistence(
            &df,
            "uid",
            "ts",
            LocationMode::LocationCol,
            Some("venue"),
            9,
            200,
        )
        .unwrap();
        assert_eq!(graph.node_count, 3);
        assert_eq!(graph.time_steps, 2);
        assert_eq!(graph.edge_from.len(), 2);
        assert!(graph.warnings.is_empty());
    }

    #[test]
    fn max_group_size_below_two_errors() {
        let df = sample_df();
        let err = observed_edges_and_persistence(
            &df,
            "uid",
            "ts",
            LocationMode::LocationCol,
            Some("venue"),
            9,
            1,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("max_group_size must be at least 2")
        );
    }
}
