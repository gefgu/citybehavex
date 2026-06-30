//! CityBehavEx simulation core with social exploration and trip-duration stay emission.

mod activity;
mod engine;
mod inputs;
mod outputs;
mod py_interface;
mod social;
mod types;

pub use py_interface::simulation_core_simulate_agents;
