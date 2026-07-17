//! CityBehavEx simulation core with social exploration and trip-duration stay emission.
//!
//! Contraction-hierarchy road routing still lives in `citybehavex-core::roads`
//! (also consumed directly, no PyO3, by `citybehavex-web`) and is imported
//! from there below. H3 batch conversion and co-presence/graph-metrics used
//! to live in `citybehavex-core::h3_batch`/`network_graph`; both are now
//! consumed directly from `fastmob-core` instead (no citybehavex-side
//! duplicate), since every Python caller already moved to
//! `fastmob.preprocessing.latlng_to_h3`/`fastmob.measures.collective.contact_network`.

mod activity;
mod engine;
mod inputs;
mod outputs;
mod py_interface;
mod social;
mod types;

pub use py_interface::{RoadNetworkHandle, simulation_core_simulate_agents};
