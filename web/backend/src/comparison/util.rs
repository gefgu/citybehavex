//! Small shared Polars-expression helpers used across the comparison
//! engine's submodules.

use polars::prelude::*;
use rustc_hash::FxHashMap;

/// Mirrors `comparison.py::_to_datetime`: coerce a datetime-ish column
/// (string or already-parsed) to a `Datetime` dtype, coercing unparsable
/// values to null (`strict=False`) rather than erroring.
pub fn to_datetime_expr(schema: &Schema, name: &str) -> Expr {
    match schema.get(name) {
        Some(DataType::String) => col(name).str().to_datetime(
            Some(TimeUnit::Microseconds),
            None,
            StrptimeOptions {
                format: None,
                strict: false,
                exact: true,
                cache: true,
            },
            lit("raise"),
        ),
        Some(DataType::Datetime(_, _)) => col(name),
        _ => col(name).cast(DataType::Datetime(TimeUnit::Microseconds, None)),
    }
}

/// Haversine great-circle distance (km) between two points, as a Polars
/// expression -- mirrors `comparison.py::_haversine_km_expr`, staying inside
/// the lazy/streaming engine instead of forcing eager numpy materialization.
///
/// **Known, deliberate deviation**: the Python source clamps via
/// `pl.min_horizontal(a.sqrt(), pl.lit(1.0))`, but `min_horizontal` skips
/// nulls rather than propagating them (confirmed against the installed
/// polars: `min_horizontal(None, 1.0) == 1.0`). Since `a` is null for every
/// leg's first waypoint (no predecessor), that bug adds a spurious
/// `arcsin(1.0) -> ~20015.09 km` "jump" to every transport leg's total in
/// `_synthetic_transport_leg_records`'s `mean_jump_km` output -- see
/// `transport.rs`'s `gparis_moving_sidecar_matches_python_reference` test
/// for the full writeup and a real-data cross-check confirming both this
/// port's physically-correct numbers and the exact size of Python's bug.
/// `.clip_max()` here correctly propagates null instead, so this port does
/// NOT reproduce that bug.
pub fn haversine_km_expr(lat1: Expr, lng1: Expr, lat2: Expr, lng2: Expr) -> Expr {
    let lat1_r = lat1.radians();
    let lng1_r = lng1.radians();
    let lat2_r = lat2.radians();
    let lng2_r = lng2.radians();
    let dlat = lat2_r.clone() - lat1_r.clone();
    let dlng = lng2_r.clone() - lng1_r.clone();
    let a = (dlat / lit(2.0)).sin().pow(2)
        + lat1_r.cos() * lat2_r.cos() * (dlng / lit(2.0)).sin().pow(2);
    lit(6371.0088) * lit(2.0) * a.sqrt().clip_max(lit(1.0)).arcsin()
}

/// `BooleanChunked::fill_null` takes a `FillNullStrategy`, not a raw value,
/// so filling nulls with a literal `false` (the `~(...).fill_null(False)`
/// pattern used throughout `comparison.py`'s window-function boolean masks)
/// needs a small helper instead.
pub fn fill_null_false(ca: &BooleanChunked) -> BooleanChunked {
    ca.into_iter().map(|v| v.unwrap_or(false)).collect()
}

pub fn count_true(ca: &BooleanChunked) -> i64 {
    ca.into_iter().filter(|v| v.unwrap_or(false)).count() as i64
}

/// Canonicalizes a user-ID column to `Int64` codes suitable for boundary
/// detection (e.g. `contiguous_user_ranges`) or grouping.
///
/// Already-integer dtypes cast straight through (no precision loss for real
/// uid ranges). Anything else (typically `String`, since real observed
/// survey data can use composite identifiers like `"10_2980"`) is factorized
/// into dense codes by first-appearance order instead. **This distinction
/// matters**: a naive `.cast(&DataType::Int64)` silently nulls every
/// unparseable string value, and a common follow-up `.unwrap_or(i64::MIN)`
/// then collapses *every* such user into one fake shared ID -- confirmed on
/// real `gparis_simulation` observed data (string uids like `"10_2980"`):
/// this collapsed all 504 distinct users into a single contiguous group,
/// creating 503 spurious inter-user "jumps" at the 503 false boundaries
/// between what should have been separate users (504 users - 1 = 503),
/// inflating jump-length/radius-of-gyration/activity-transition counts.
pub fn canonical_user_ids(uid: &Series) -> anyhow::Result<Series> {
    if matches!(
        uid.dtype(),
        DataType::Int64
            | DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::UInt64
            | DataType::UInt32
            | DataType::UInt16
            | DataType::UInt8
    ) {
        return Ok(uid.cast(&DataType::Int64)?.with_name("user_id".into()));
    }

    let uid = uid.cast(&DataType::String)?;
    let uid = uid.str()?;
    let mut labels = FxHashMap::<String, i64>::default();
    let mut next = 0i64;
    let values: Vec<Option<i64>> = uid
        .into_iter()
        .map(|value| {
            value.map(|value| {
                *labels.entry(value.to_string()).or_insert_with(|| {
                    let current = next;
                    next += 1;
                    current
                })
            })
        })
        .collect();
    Ok(Series::new("user_id".into(), values))
}

/// Convenience wrapper returning plain `i64`s (nulls as `i64::MIN`, matching
/// how callers already treated unparseable/missing uids) for call sites that
/// just need a `Vec<i64>` for boundary detection rather than a `Series`
/// column.
pub fn canonical_user_ids_vec(uid: &Series) -> anyhow::Result<Vec<i64>> {
    Ok(canonical_user_ids(uid)?
        .i64()?
        .into_iter()
        .map(|v| v.unwrap_or(i64::MIN))
        .collect())
}
