//! Shared Rust core for CityBehavEx: contraction-hierarchy road routing.
//!
//! Extracted out of `citybehavex-py` (where it was only reachable via PyO3)
//! so it can be linked directly, with no Python in the loop, by both
//! `citybehavex-py` (thin PyO3 wrappers) and `citybehavex-web` (the axum
//! backend). H3 batch conversion and co-presence/graph-metrics computation
//! used to live here too (`h3_batch`/`network_graph`); both are now
//! consumed directly from `fastmob-core` (both crates already depend on it)
//! instead of maintaining a citybehavex-local duplicate.

pub mod roads;
