//! PyO3 bindings: exposes `aegis-core` to Python as `aegis._core`.

use pyo3::prelude::*;

/// Version string of the compiled Rust core.
#[pyfunction]
fn core_version() -> &'static str {
    aegis_core::VERSION
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    Ok(())
}
