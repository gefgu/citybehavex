//! CityBehavEx simulation core with social exploration and trip-duration stay emission.
//!
//! H3 batch conversion, contraction-hierarchy road routing, and co-presence/
//! graph-metrics used to live in `h3_batch`/`roads`/`network_graph` modules
//! here; they're now in the shared `citybehavex-core` crate (also consumed
//! directly, no PyO3, by `citybehavex-web`) and imported from there below.

mod activity;
mod engine;
mod inputs;
mod outputs;
mod py_interface;
mod social;
mod types;

pub use py_interface::{
    RoadNetworkHandle, batch_latlng_to_cells_py, build_co_presence_edges_py, graph_metrics_py,
    simulation_core_simulate_agents,
};
