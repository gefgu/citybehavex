mod ditras_trip;
mod trip_ditras_core;
mod trip_sts_epr;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        ditras_trip::trip_ditras_simulate_agents,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        trip_sts_epr::trip_sts_epr_simulate_agents,
        m
    )?)?;
    Ok(())
}
