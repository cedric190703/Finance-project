"""Type stubs for the compiled Rust extension module (`crates/aegis-py`)."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Paths per work unit. Fixed so results do not depend on the thread count.
CHUNK_PATHS: int

def core_version() -> str:
    """Return the version of the compiled kernel library."""

def mc_european(
    forward: float,
    strike: float,
    vol: float,
    time: float,
    discount: float = ...,
    right: str = ...,
    paths: int = ...,
    seed: int = ...,
    sampler: str = ...,
    antithetic: bool | None = ...,
    control_variate: bool | None = ...,
) -> tuple[float, float, int, int]:
    """Return the price, its standard error, paths consumed, and replica count."""

def mc_european_batch(
    forwards: FloatArray,
    strikes: FloatArray,
    vols: FloatArray,
    times: FloatArray,
    discounts: FloatArray,
    rights: list[str],
    paths: int = ...,
    seed: int = ...,
    sampler: str = ...,
    antithetic: bool | None = ...,
    control_variate: bool | None = ...,
) -> tuple[FloatArray, FloatArray]:
    """Return prices and standard errors for a whole book."""

def black_price(
    forward: float,
    strike: float,
    vol: float,
    time: float,
    discount: float = ...,
    right: str = ...,
) -> float:
    """Return Black's formula, evaluated in the compiled core."""

def normal_cdf(x: float) -> float:
    """Return the standard normal cumulative distribution."""

def normal_inverse_cdf(p: float) -> float:
    """Return the inverse standard normal cumulative distribution."""

def sobol_points(count: int, shift: int = ...) -> FloatArray:
    """Return the first `count` points of the digitally shifted Sobol sequence."""
