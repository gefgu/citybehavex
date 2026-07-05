//! Batch lat/lng -> H3 cell conversion.
//!
//! `h3o`'s per-point `LatLng::to_cell` is the same algorithm h3-py's scalar
//! `latlng_to_cell` uses, but calling it from a Python `for`/list-comprehension
//! loop pays Python-call overhead per row. Report-comparison code needs this
//! for every row of observed datasets that have no explicit location column
//! (tens to hundreds of millions of rows for the larger simulations), so it's
//! run here across all rows in one call, in parallel.

use h3o::{LatLng, Resolution};
use rayon::prelude::*;

/// `u64::MAX` is not a valid H3 cell index (the top reserved bits are never
/// all-1 for a valid cell), so it doubles as the "invalid input" sentinel for
/// non-finite or out-of-range lat/lng -- the Python caller is expected to
/// treat it the same way it already treats a missing/NaN location.
pub const INVALID_CELL: u64 = u64::MAX;

/// Converts `(lat, lng)` pairs (degrees) to H3 cell indices at `resolution`,
/// in parallel. Invalid coordinates map to [`INVALID_CELL`] rather than
/// failing the whole batch, since real-world check-in data routinely has a
/// few bad rows mixed into an otherwise valid column.
pub(crate) fn batch_latlng_to_cells(lats: &[f64], lngs: &[f64], resolution: Resolution) -> Vec<u64> {
    lats.par_iter()
        .zip(lngs.par_iter())
        .map(|(&lat, &lng)| {
            LatLng::new(lat, lng)
                .map(|ll| u64::from(ll.to_cell(resolution)))
                .unwrap_or(INVALID_CELL)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_cell() {
        // Cross-checked against h3-py: h3.latlng_to_cell(37.769377, -122.388519, 9)
        // == '89283082e73ffff'.
        let cells = batch_latlng_to_cells(&[37.769377], &[-122.388519], Resolution::Nine);
        assert_eq!(cells, vec![0x89283082e73ffffu64]);
    }

    #[test]
    fn invalid_coordinates_map_to_sentinel() {
        let cells = batch_latlng_to_cells(&[f64::NAN, 10.0], &[20.0, f64::INFINITY], Resolution::Nine);
        assert_eq!(cells, vec![INVALID_CELL, INVALID_CELL]);
    }

    #[test]
    fn batch_matches_scalar_one_at_a_time() {
        let lats = [37.769377, -33.865143, 51.507351];
        let lngs = [-122.388519, 151.209900, -0.127758];
        let batch = batch_latlng_to_cells(&lats, &lngs, Resolution::Nine);
        for i in 0..lats.len() {
            let single = batch_latlng_to_cells(&lats[i..=i], &lngs[i..=i], Resolution::Nine);
            assert_eq!(batch[i], single[0]);
        }
    }
}
