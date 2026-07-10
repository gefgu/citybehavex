//! Small shared Polars-expression helpers used across the comparison
//! engine's submodules.

use polars::prelude::*;

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

/// Plain-`f64`-slice haversine, mirrors `comparison.py::_haversine_km_np`
/// (used where callers already have materialized numpy-equivalent arrays,
/// e.g. `_observed_transport_leg_records`).
pub fn haversine_km(lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> f64 {
    let (lat1_r, lng1_r, lat2_r, lng2_r) =
        (lat1.to_radians(), lng1.to_radians(), lat2.to_radians(), lng2.to_radians());
    let dlat = lat2_r - lat1_r;
    let dlng = lng2_r - lng1_r;
    let a = (dlat / 2.0).sin().powi(2) + lat1_r.cos() * lat2_r.cos() * (dlng / 2.0).sin().powi(2);
    6371.0088 * 2.0 * a.sqrt().min(1.0).asin()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_known_distance() {
        // San Francisco to Los Angeles, ~559 km great-circle.
        let d = haversine_km(37.7749, -122.4194, 34.0522, -118.2437);
        assert!((d - 559.0).abs() < 5.0, "got {d}");
    }
}
