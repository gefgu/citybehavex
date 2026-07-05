//! CityBehavEx simulation core with social exploration and trip-duration stay emission.

mod activity;
mod engine;
mod h3_batch;
mod inputs;
mod network_graph;
mod outputs;
mod py_interface;
mod roads;
mod social;
mod types;

pub use py_interface::{
    batch_latlng_to_cells_py, build_co_presence_edges_py, graph_metrics_py,
    simulation_core_simulate_agents, RoadNetworkHandle,
};
