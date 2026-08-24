"""Curve interpolation schemes.

The choice of scheme is not cosmetic. A curve is interpolated at every date a
cash flow lands on, and a scheme that looks smooth in zero-rate space can imply
a sawtooth in forward space — which then shows up as noise in every DV01 and
carry number computed from it.

Three schemes are offered, in increasing order of smoothness:

``LINEAR_ZERO``
    Straight lines between zero rates. Simple, always arbitrage-free in the
    sense that discount factors stay monotone, but its forwards are piecewise
    linear with kinks at every knot.
``LOG_LINEAR_DISCOUNT``
    Straight lines in log discount factor, which is exactly piecewise-constant
    instantaneous forwards. The market default for a reason: the forwards are
    ugly but honest, and nothing is invented between knots.
``NATURAL_CUBIC_ZERO``
    A natural cubic spline through the zero rates. Smooth forwards, at the price
    of overshoot when the input curve has a kink in it — which a real one does.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt

__all__ = ["Interpolation", "interpolate_zero_rates"]

FloatArray = npt.NDArray[np.float64]


class Interpolation(StrEnum):
    """How a curve is interpolated between its knots."""

    LINEAR_ZERO = "LINEAR_ZERO"
    LOG_LINEAR_DISCOUNT = "LOG_LINEAR_DISCOUNT"
    NATURAL_CUBIC_ZERO = "NATURAL_CUBIC_ZERO"


def interpolate_zero_rates(
    times: FloatArray,
    zeros: FloatArray,
    targets: FloatArray,
    scheme: Interpolation,
) -> FloatArray:
    """Interpolate continuously compounded zero rates onto new times.

    Beyond the last knot every scheme extrapolates flat, which is the
    conservative choice: a curve should not invent a trend it was never shown.
    Below the first knot the first zero rate is held flat for the same reason.

    Args:
        times: Knot times in years, strictly increasing and positive.
        zeros: Continuously compounded zero rates at the knots.
        targets: Times to interpolate onto, in years.
        scheme: The interpolation scheme.

    Returns:
        Zero rates at ``targets``.

    Raises:
        ValueError: if the knots are empty, mismatched or not increasing.
    """
    times = np.asarray(times, dtype=np.float64)
    zeros = np.asarray(zeros, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if times.size == 0 or times.size != zeros.size:
        raise ValueError("curve needs a matching, non-empty set of times and zero rates")
    if np.any(np.diff(times) <= 0):
        raise ValueError("curve times must be strictly increasing")
    if times.size == 1:
        return np.full_like(targets, zeros[0])

    clipped = np.clip(targets, times[0], times[-1])

    match scheme:
        case Interpolation.LINEAR_ZERO:
            return np.interp(clipped, times, zeros)
        case Interpolation.LOG_LINEAR_DISCOUNT:
            # Linear in log DF = linear in -z*t, so interpolate the product and
            # divide the time back out. Equivalent to constant forwards per bucket.
            log_df = -zeros * times
            interpolated = np.interp(clipped, times, log_df)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(clipped > 0, -interpolated / clipped, zeros[0])
        case Interpolation.NATURAL_CUBIC_ZERO:
            return _natural_cubic(times, zeros, clipped)

    raise AssertionError(f"unhandled interpolation: {scheme}")  # pragma: no cover


def _natural_cubic(times: FloatArray, zeros: FloatArray, targets: FloatArray) -> FloatArray:
    """Evaluate a natural cubic spline (zero second derivative at both ends)."""
    n = times.size
    if n < 3:  # a spline through two points is just a line
        return np.interp(targets, times, zeros)

    h = np.diff(times)
    # Solve the tridiagonal system for the second derivatives.
    lower = np.zeros(n)
    diag = np.ones(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)
    lower[1:-1] = h[:-1]
    diag[1:-1] = 2.0 * (h[:-1] + h[1:])
    upper[1:-1] = h[1:]
    rhs[1:-1] = 6.0 * (np.diff(zeros)[1:] / h[1:] - np.diff(zeros)[:-1] / h[:-1])

    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, diag)
    for i in range(1, n - 1):
        matrix[i, i - 1] = lower[i]
        matrix[i, i + 1] = upper[i]
    second = np.linalg.solve(matrix, rhs)

    index = np.clip(np.searchsorted(times, targets, side="right") - 1, 0, n - 2)
    dx = targets - times[index]
    step = h[index]
    a = (times[index + 1] - targets) / step
    b = dx / step
    return (
        a * zeros[index]
        + b * zeros[index + 1]
        + ((a**3 - a) * second[index] + (b**3 - b) * second[index + 1]) * (step**2) / 6.0
    )
