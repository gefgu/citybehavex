//! Column-name auto-detection, shared across `datasource.rs` and
//! `comparison::*` (and, later, `home_work.rs`/`timeline.rs`). The Python
//! codebase keeps near-identical copies of these candidate lists in
//! `citybehavex/reports/comparison.py`, `web/backend/app/datasource.py`, and
//! `web/backend/app/home_work_data.py`, each with a comment noting they must
//! be "kept in sync" by hand -- centralized here instead, since Rust's
//! module system makes that free.

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
pub const LAT_CANDIDATES: &[&str] = &["lat", "latitude"];
pub const LNG_CANDIDATES: &[&str] = &["lng", "lon", "longitude", "long"];
pub const DURATION_CANDIDATES: &[&str] =
    &["duration_minutes", "duration", "trip_duration_minutes", "duration_hours"];
pub const ACTIVITY_CANDIDATES: &[&str] =
    &["purpose", "activity", "act", "location_type", "category", "purpose_d"];
pub const LOCATION_CANDIDATES: &[&str] =
    &["location_id", "tile_id", "Code_INSEE_D", "area", "venueId", "location"];
pub const END_TS_CANDIDATES: &[&str] = &["end_timestamp", "_end_time", "end_time"];
pub const TRANSPORT_CANDIDATES: &[&str] =
    &["mode", "transport_mode", "transport", "travel_mode", "trip_mode", "vehicle_mode"];

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

/// Same lookup directly against a Polars schema/frame's column names.
pub fn detect_in(columns: &[&str], candidates: &[&str]) -> Option<String> {
    for candidate in candidates {
        if let Some(found) = columns.iter().find(|c| c.eq_ignore_ascii_case(candidate)) {
            return Some((*found).to_string());
        }
    }
    None
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
        assert_eq!(detect_column(&cols, UID_CANDIDATES), Some("uid"));
    }

    #[test]
    fn detect_in_matches_str_slice() {
        let cols = ["Latitude", "Longitude"];
        assert_eq!(detect_in(&cols, LAT_CANDIDATES), Some("Latitude".to_string()));
    }
}
