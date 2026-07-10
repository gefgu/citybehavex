//! Shared Rust core for CityBehavEx: H3 batch conversion, contraction-
//! hierarchy road routing, and co-presence/graph-metrics computation.
//!
//! Extracted out of `citybehavex-py` (where it was only reachable via PyO3)
//! so it can be linked directly, with no Python in the loop, by both
//! `citybehavex-py` (thin PyO3 wrappers) and `citybehavex-web` (the axum
//! backend).

pub mod h3_batch;
pub mod network_graph;
pub mod roads;
