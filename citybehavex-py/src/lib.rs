mod simulation_core;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        simulation_core::simulation_core_simulate_agents,
        m
    )?)?;
    m.add_class::<simulation_core::RoadNetworkHandle>()?;
    Ok(())
}
