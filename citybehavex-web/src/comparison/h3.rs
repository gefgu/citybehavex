//! Mirrors `comparison.py::_h3_cells` and `_location_resolution`.

use citybehavex_core::h3_batch::{self, INVALID_CELL};
use h3o::{CellIndex, Resolution};
use polars::prelude::*;

/// Vectorized lat/lng -> H3 cell index, via `citybehavex-core` (the same
/// Rust extraction `_h3_cells` calls through `citybehavex._core` on the
/// Python side) instead of a per-row `h3.latlng_to_cell` loop. Returns a
/// nullable `UInt64` series; invalid/non-finite coordinates map to null.
pub fn h3_cells(lat: &Series, lng: &Series, resolution: u8) -> anyhow::Result<Series> {
    let res = Resolution::try_from(resolution)
        .map_err(|e| anyhow::anyhow!("invalid H3 resolution {resolution}: {e}"))?;
    let lat_f64 = lat.cast(&DataType::Float64)?;
    let lng_f64 = lng.cast(&DataType::Float64)?;
    let lat_ca = lat_f64.f64()?;
    let lng_ca = lng_f64.f64()?;
    // A null in a nullable Float64 column has no representable value; NaN is
    // the same "invalid" sentinel `_h3_cells` relies on after `.to_numpy()`
    // (polars converts null floats to NaN there), and `batch_latlng_to_cells`
    // already treats non-finite input as invalid.
    let lats: Vec<f64> = lat_ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let lngs: Vec<f64> = lng_ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();

    let cells = h3_batch::batch_latlng_to_cells(&lats, &lngs, res);
    let opt_cells: Vec<Option<u64>> = cells
        .into_iter()
        .map(|c| if c == INVALID_CELL { None } else { Some(c) })
        .collect();
    Ok(Series::new(lat.name().clone(), opt_cells))
}

/// H3 resolution of a hex-string cell id, mirrors `comparison.py::_location_resolution`
/// (`h3.get_resolution`) for the first parseable value in `location_col`, or
/// `default` if none parse / the column is absent.
pub fn location_resolution(values: impl IntoIterator<Item = String>, default: u8) -> u8 {
    for value in values {
        if let Ok(cell) = value.parse::<CellIndex>() {
            return cell.resolution() as u8;
        }
        // `CellIndex::from_str` expects a hex string; try explicit hex parse
        // too since callers may hand a plain hex string without a leading tag.
        if let Ok(raw) = u64::from_str_radix(&value, 16) {
            if let Ok(cell) = CellIndex::try_from(raw) {
                return cell.resolution() as u8;
            }
        }
    }
    default
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_cell() {
        let lat = Series::new("lat".into(), vec![37.769377f64]);
        let lng = Series::new("lng".into(), vec![-122.388519f64]);
        let cells = h3_cells(&lat, &lng, 9).unwrap();
        let ca = cells.u64().unwrap();
        assert_eq!(ca.get(0), Some(0x89283082e73ffffu64));
    }

    #[test]
    fn null_and_nan_map_to_null_cell() {
        let lat = Series::new("lat".into(), vec![Some(37.769377f64), None, Some(f64::NAN)]);
        let lng = Series::new("lng".into(), vec![Some(-122.388519f64), Some(1.0), Some(2.0)]);
        let cells = h3_cells(&lat, &lng, 9).unwrap();
        let ca = cells.u64().unwrap();
        assert!(ca.get(0).is_some());
        assert_eq!(ca.get(1), None);
        assert_eq!(ca.get(2), None);
    }

    #[test]
    fn location_resolution_parses_hex_cell_string() {
        let cell_hex = format!("{:x}", 0x89283082e73ffffu64);
        assert_eq!(location_resolution(vec![cell_hex], 10), 9);
        assert_eq!(location_resolution(Vec::<String>::new(), 10), 10);
    }
}
