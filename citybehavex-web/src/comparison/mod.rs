//! Native Rust port of the parts of `citybehavex/reports/comparison.py`
//! that `web/backend/app/payload/legacy.py` actually reuses (via
//! `web/backend/app/reports_bridge.py`'s single import surface) -- not the
//! CLI HTML-report-generation entry points, which the web backend never
//! calls.
//!
//! Uses `polars` (Rust) for the dataframe pipelines (near-1:1 port of the
//! Python `polars` code), `fkmob-core` directly (no PyO3) for the numeric
//! primitives it implements natively (Wasserstein, activity transition
//! counts, motif discovery, visitation-law distances, waiting times,
//! trajectory CPC, STVD-EMD), and `citybehavex-core::h3_batch` for H3
//! binning (mirrors `_h3_cells`'s use of `citybehavex._core.batch_latlng_to_cells`).

pub mod activity;
pub mod h3;
pub mod metrics;
pub mod micro_activity;
pub mod mobility_laws;
pub mod panel;
pub mod stvd;
pub mod trajectory;
pub mod transport;
pub mod util;
pub mod visits;

pub use trajectory::Trajectory;

/// Mirrors `comparison.py::CAR_SPEED_KMH` -- speed used to turn real jump
/// lengths into a car travel-time proxy for the trip-duration comparison.
pub const CAR_SPEED_KMH: f64 = 50.0;
pub const CPC_H3_RESOLUTIONS: [u8; 3] = [7, 8, 9];
pub const DEFAULT_MODE_ORDER: [&str; 4] = ["walk", "bike", "car", "rail"];

/// Mirrors `comparison.py::ActivityVisitsResult`.
pub struct ActivityVisitsResult {
    pub visits: polars::prelude::DataFrame,
    pub used_heuristic: bool,
    pub warning: Option<String>,
}

/// Mirrors `comparison.py::EvaluationAdaptationResult`.
pub struct EvaluationAdaptationResult {
    pub df: polars::prelude::DataFrame,
    pub adapted: bool,
    pub warning: Option<String>,
    pub location_col: Option<String>,
    pub h3_resolution: Option<i64>,
}
