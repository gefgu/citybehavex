#![deny(clippy::disallowed_types)]

mod simulation_core;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        simulation_core::simulation_core_simulate_agents,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        simulation_core::batch_latlng_to_cells_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        simulation_core::build_co_presence_edges_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(simulation_core::graph_metrics_py, m)?)?;
    m.add_class::<simulation_core::RoadNetworkHandle>()?;
    Ok(())
}
