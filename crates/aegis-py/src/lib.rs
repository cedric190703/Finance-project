//! PyO3 bindings: exposes `aegis-core` to Python as `aegis._core`.
//!
//! The boundary is kept deliberately thin. Everything numerical lives in
//! `aegis-core`, where it can be tested and benchmarked without a Python
//! interpreter in the loop; this crate only converts types and releases the GIL.
//!
//! Releasing the GIL is not a detail. Without it the rayon threads inside the
//! kernel would be blocked by any other Python thread, and a service handling
//! two valuation requests at once would run them one after the other while
//! reporting that it was parallel.

// Fires inside the code `#[pyfunction]` generates for a fallible return type,
// not in anything written here; there is nothing to remove.
#![allow(clippy::useless_conversion)]

use aegis_core::black::OptionRight;
use aegis_core::mc::{self, EuropeanOption, Sampler, VarianceReduction};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};

/// Prices and standard errors for a whole book, as numpy arrays.
type PricedBook<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Version string of the compiled Rust core.
#[pyfunction]
fn core_version() -> &'static str {
    aegis_core::VERSION
}

fn parse_right(right: &str) -> PyResult<OptionRight> {
    right
        .chars()
        .next()
        .ok_or_else(|| PyValueError::new_err("option right must be 'C' or 'P'"))
        .and_then(|c| OptionRight::from_char(c).map_err(PyValueError::new_err))
}

fn parse_sampler(sampler: &str) -> PyResult<Sampler> {
    match sampler.to_ascii_lowercase().as_str() {
        "pseudo" | "pseudorandom" | "pseudo_random" => Ok(Sampler::PseudoRandom),
        "quasi" | "quasirandom" | "quasi_random" | "sobol" => Ok(Sampler::QuasiRandom),
        other => Err(PyValueError::new_err(format!(
            "sampler must be 'pseudo' or 'quasi', got {other:?}"
        ))),
    }
}

fn reduction(
    antithetic: Option<bool>,
    control_variate: Option<bool>,
    sampler: Sampler,
) -> VarianceReduction {
    let recommended = VarianceReduction::recommended(sampler);
    VarianceReduction {
        antithetic: antithetic.unwrap_or(recommended.antithetic),
        control_variate: control_variate.unwrap_or(recommended.control_variate),
    }
}

/// Price one European option by Monte Carlo.
#[pyfunction]
#[pyo3(signature = (
    forward, strike, vol, time, discount=1.0, right="C", paths=1 << 20, seed=0,
    sampler="quasi", antithetic=None, control_variate=None
))]
#[allow(clippy::too_many_arguments)]
fn mc_european(
    py: Python<'_>,
    forward: f64,
    strike: f64,
    vol: f64,
    time: f64,
    discount: f64,
    right: &str,
    paths: usize,
    seed: u64,
    sampler: &str,
    antithetic: Option<bool>,
    control_variate: Option<bool>,
) -> PyResult<(f64, f64, usize, usize)> {
    let option = EuropeanOption {
        forward,
        strike,
        vol,
        time,
        discount,
        right: parse_right(right)?,
    };
    let sampler = parse_sampler(sampler)?;
    let settings = reduction(antithetic, control_variate, sampler);

    let result = py.allow_threads(|| mc::european(option, paths, seed, sampler, settings));
    Ok((
        result.price,
        result.standard_error,
        result.paths,
        result.replicas,
    ))
}

/// Price a whole book of European options, in parallel across the batch.
#[pyfunction]
#[pyo3(signature = (
    forwards, strikes, vols, times, discounts, rights, paths=1 << 18, seed=0,
    sampler="quasi", antithetic=None, control_variate=None
))]
#[allow(clippy::too_many_arguments)]
fn mc_european_batch<'py>(
    py: Python<'py>,
    forwards: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    vols: PyReadonlyArray1<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    discounts: PyReadonlyArray1<'py, f64>,
    rights: Vec<String>,
    paths: usize,
    seed: u64,
    sampler: &str,
    antithetic: Option<bool>,
    control_variate: Option<bool>,
) -> PyResult<PricedBook<'py>> {
    let forwards = forwards.as_slice()?;
    let strikes = strikes.as_slice()?;
    let vols = vols.as_slice()?;
    let times = times.as_slice()?;
    let discounts = discounts.as_slice()?;

    let n = forwards.len();
    if [
        strikes.len(),
        vols.len(),
        times.len(),
        discounts.len(),
        rights.len(),
    ]
    .iter()
    .any(|len| *len != n)
    {
        return Err(PyValueError::new_err(
            "forwards, strikes, vols, times, discounts and rights must all be the same length",
        ));
    }

    let mut options = Vec::with_capacity(n);
    for i in 0..n {
        options.push(EuropeanOption {
            forward: forwards[i],
            strike: strikes[i],
            vol: vols[i],
            time: times[i],
            discount: discounts[i],
            right: parse_right(&rights[i])?,
        });
    }

    let sampler = parse_sampler(sampler)?;
    let settings = reduction(antithetic, control_variate, sampler);
    let results = py.allow_threads(|| mc::european_batch(&options, paths, seed, sampler, settings));

    let prices: Vec<f64> = results.iter().map(|r| r.price).collect();
    let errors: Vec<f64> = results.iter().map(|r| r.standard_error).collect();
    Ok((prices.to_pyarray_bound(py), errors.to_pyarray_bound(py)))
}

/// Black's formula, evaluated in the compiled core.
///
/// Exposed mostly so the test suite can check that both implementations of the
/// same formula agree across the FFI boundary.
#[pyfunction]
#[pyo3(signature = (forward, strike, vol, time, discount=1.0, right="C"))]
fn black_price(
    forward: f64,
    strike: f64,
    vol: f64,
    time: f64,
    discount: f64,
    right: &str,
) -> PyResult<f64> {
    Ok(aegis_core::black::price(
        forward,
        strike,
        vol,
        time,
        discount,
        parse_right(right)?,
    ))
}

/// Standard normal cumulative distribution.
#[pyfunction]
fn normal_cdf(x: f64) -> f64 {
    aegis_core::normal::cdf(x)
}

/// Inverse standard normal cumulative distribution.
#[pyfunction]
fn normal_inverse_cdf(p: f64) -> f64 {
    aegis_core::normal::inverse_cdf(p)
}

/// Return the first `count` points of the digitally shifted Sobol sequence.
#[pyfunction]
#[pyo3(signature = (count, shift=0))]
fn sobol_points(py: Python<'_>, count: usize, shift: u64) -> Bound<'_, PyArray1<f64>> {
    let mut sequence = aegis_core::sobol::Sobol1d::new(shift);
    let points: Vec<f64> = (0..count).map(|_| sequence.next_point()).collect();
    points.to_pyarray_bound(py)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CHUNK_PATHS", mc::CHUNK_PATHS)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(mc_european, m)?)?;
    m.add_function(wrap_pyfunction!(mc_european_batch, m)?)?;
    m.add_function(wrap_pyfunction!(black_price, m)?)?;
    m.add_function(wrap_pyfunction!(normal_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(normal_inverse_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(sobol_points, m)?)?;
    Ok(())
}
