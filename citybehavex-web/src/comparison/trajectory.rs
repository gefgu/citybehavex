//! Mirrors `comparison.py::load_trajectory` and the small parts of
//! `fkmob.TrajDataFrame` the web payload path actually touches (column-name
//! bookkeeping) -- not fkmob's Python wrapper class itself, which also
//! carries convenience methods (`.jump_lengths()`, `.radius_of_gyration()`)
//! the web port calls into `fkmob-core` for directly instead (see
//! `metrics.rs`).

use crate::columns::{
    DATETIME_CANDIDATES, LAT_CANDIDATES, LNG_CANDIDATES, UID_CANDIDATES, detect_in,
};
use polars::prelude::*;
use std::path::Path;

pub struct Trajectory {
    pub df: DataFrame,
    pub datetime_col: String,
    pub lat_col: String,
    pub lng_col: String,
    pub uid_col: String,
}

pub fn read_parquet(path: &Path) -> anyhow::Result<DataFrame> {
    let file = std::fs::File::open(path)?;
    Ok(ParquetReader::new(file).finish()?)
}

/// Mirrors `comparison.py::load_trajectory`.
pub fn load_trajectory(path: &Path) -> anyhow::Result<Trajectory> {
    let df = read_parquet(path)?;
    let cols: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
    let datetime_col = detect_in(&cols, DATETIME_CANDIDATES);
    let lat_col = detect_in(&cols, LAT_CANDIDATES);
    let lng_col = detect_in(&cols, LNG_CANDIDATES);
    let uid_col = detect_in(&cols, UID_CANDIDATES);

    let mut missing = Vec::new();
    if datetime_col.is_none() {
        missing.push("datetime");
    }
    if lat_col.is_none() {
        missing.push("latitude");
    }
    if lng_col.is_none() {
        missing.push("longitude");
    }
    if uid_col.is_none() {
        missing.push("user ID");
    }
    if !missing.is_empty() {
        anyhow::bail!(
            "{} is missing recognizable columns for: {}",
            path.display(),
            missing.join(", ")
        );
    }

    Ok(Trajectory {
        df,
        datetime_col: datetime_col.unwrap(),
        lat_col: lat_col.unwrap(),
        lng_col: lng_col.unwrap(),
        uid_col: uid_col.unwrap(),
    })
}
