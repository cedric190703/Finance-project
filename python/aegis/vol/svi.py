"""The SVI parameterisation of a volatility slice.

Gatheral's stochastic volatility inspired form describes the *total variance* of
one expiry as a function of log-moneyness:

    w(k) = a + b · [ ρ·(k − m) + √((k − m)² + σ²) ]

Five parameters, each with a reading a trader would recognise: ``a`` is the
overall level, ``b`` the wing slope, ``ρ`` the skew, ``m`` the horizontal shift
of the smile's minimum, and ``σ`` how rounded that minimum is.

Two properties are the reason to use it rather than fitting a spline through the
quotes. The wings are linear in ``k``, which is what Roger Lee's moment formula
says they must be; and the butterfly condition — that the implied risk-neutral
density stays non-negative — can be checked in closed form rather than sampled
and hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

__all__ = ["SviFitError", "SviParameters", "fit_svi_slice"]

FloatArray = npt.NDArray[np.float64]

_MIN_QUOTES = 5
#: Width of the smooth hinge used for the soft constraints, in total variance.
_HINGE_WIDTH = 1e-5


class SviFitError(RuntimeError):
    """Raised when a slice cannot be calibrated from the quotes given."""


@dataclass(frozen=True)
class SviParameters:
    """A calibrated SVI slice.

    Attributes:
        a: Vertical level of total variance.
        b: Wing slope; non-negative.
        rho: Skew, between -1 and 1.
        m: Log-moneyness at which the smile is centred.
        sigma: Curvature of the smile's floor; positive.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, log_moneyness: float | FloatArray) -> FloatArray:
        """Return total variance ``w = σ²T`` at one or more log-moneyness points.

        Args:
            log_moneyness: ``ln(K/F)``, scalar or array.

        Returns:
            Total variance, floored at zero.
        """
        k = np.atleast_1d(np.asarray(log_moneyness, dtype=np.float64))
        shifted = k - self.m
        w = self.a + self.b * (self.rho * shifted + np.sqrt(shifted**2 + self.sigma**2))
        return np.maximum(w, 0.0)

    def implied_vol(self, log_moneyness: float | FloatArray, time: float) -> FloatArray:
        """Return the Black implied volatility implied by the slice.

        Args:
            log_moneyness: ``ln(K/F)``, scalar or array.
            time: Time to expiry in years.

        Returns:
            Annualised implied volatilities.

        Raises:
            ValueError: if the time to expiry is not positive.
        """
        if time <= 0:
            raise ValueError("time to expiry must be positive")
        return np.sqrt(self.total_variance(log_moneyness) / time)

    def derivatives(self, log_moneyness: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``w``, ``w'`` and ``w''`` at the given log-moneyness.

        Args:
            log_moneyness: Points to evaluate at.

        Returns:
            Total variance and its first two derivatives in log-moneyness.
        """
        k = np.asarray(log_moneyness, dtype=np.float64)
        shifted = k - self.m
        root = np.sqrt(shifted**2 + self.sigma**2)
        w = self.a + self.b * (self.rho * shifted + root)
        first = self.b * (self.rho + shifted / root)
        second = self.b * self.sigma**2 / root**3
        return w, first, second

    def butterfly_g(self, log_moneyness: FloatArray) -> FloatArray:
        """Return Gatheral's ``g`` function; negative values mean a negative density.

        A slice is free of butterfly arbitrage exactly when ``g(k) ≥ 0``
        everywhere. Where it dips below zero, the smile implies a negative
        probability, and a butterfly spread struck around that point would be
        worth less than nothing.

        Args:
            log_moneyness: Points to evaluate at.

        Returns:
            ``g(k)`` at each point.
        """
        k = np.asarray(log_moneyness, dtype=np.float64)
        w, first, second = self.derivatives(k)
        safe_w = np.maximum(w, 1e-12)
        return (
            (1.0 - k * first / (2.0 * safe_w)) ** 2
            - (first**2 / 4.0) * (1.0 / safe_w + 0.25)
            + second / 2.0
        )

    def is_butterfly_free(self, span: float = 1.5, points: int = 1001) -> bool:
        """Check the butterfly condition across a range of log-moneyness.

        The default sampling is deliberately finer than any check downstream. A
        repair that only restores the condition at its own grid points leaves it
        broken between them, and whatever samples the slice next will land
        somewhere else and find the gap.

        Args:
            span: How far into each wing to check, in log-moneyness.
            points: How finely to sample.

        Returns:
            ``True`` when ``g(k) ≥ 0`` everywhere sampled.
        """
        grid = np.linspace(-span, span, points)
        return bool(np.all(self.butterfly_g(grid) >= -1e-10))

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """Return the parameters in ``(a, b, rho, m, sigma)`` order."""
        return (self.a, self.b, self.rho, self.m, self.sigma)


def _hinge(x: FloatArray) -> FloatArray:
    """Return a smooth approximation of ``max(x, 0)``.

    The obvious way to penalise a violated constraint is ``max(violation, 0)``,
    which has a kink exactly where the constraint binds — which is exactly where
    the optimizer wants to sit. Trust-region least squares then spends thousands
    of iterations shuffling around the corner, and the fit takes seconds per
    slice instead of milliseconds. Softening the corner over a width far below
    anything economically meaningful costs nothing and removes the problem.

    Args:
        x: The constraint violation; positive where the constraint is broken.

    Returns:
        A smooth, non-negative penalty.
    """
    return _HINGE_WIDTH * np.logaddexp(0.0, x / _HINGE_WIDTH)


def fit_svi_slice(
    log_moneyness: FloatArray,
    total_variance: FloatArray,
    weights: FloatArray | None = None,
    butterfly_grid: FloatArray | None = None,
    floor_grid: FloatArray | None = None,
    variance_floor: FloatArray | None = None,
    penalty_weight: float = 1e3,
) -> tuple[SviParameters, float]:
    """Calibrate an SVI slice to observed total variances.

    The fit is a bounded least-squares problem, started from several different
    guesses. One start is not enough: the objective has local minima, and a fit
    that begins on the wrong side of the skew converges to a smile that leans
    the wrong way and still looks plausible in RMSE terms.

    Two soft constraints are folded into the residual vector rather than checked
    afterwards, because a surface that is only checked for arbitrage is a
    surface that reports arbitrage:

    * **Butterfly.** Wherever Gatheral's ``g(k)`` goes negative on the penalty
      grid, the shortfall is added as a residual. Sparse short-dated chains fit
      unconstrained SVI into smiles with negative implied density otherwise.
    * **Calendar.** Where a floor is supplied — the previous expiry's total
      variance — any shortfall below it is penalised too, which is what keeps
      total variance non-decreasing across expiries.

    They are soft on purpose. A hard constraint would rather fail than fit, and
    a slice that misses the constraint by 1e-9 is more useful than no slice.

    Args:
        log_moneyness: ``ln(K/F)`` for each quote.
        total_variance: Observed ``σ²T`` for each quote.
        weights: Relative weights, typically vega or inverse bid-ask spread.
        butterfly_grid: Log-moneyness points the density condition is enforced
            on. It can span the wings freely: the condition is a property of the
            parameterisation, not of the quotes.
        floor_grid: Log-moneyness points the calendar condition is enforced on.
            This one must stay inside the range both expiries are actually
            quoted over. Enforcing it across the wings compares one
            extrapolation against another and, when it binds, inflates the level
            of a slice to satisfy a constraint at strikes nobody trades.
        variance_floor: Minimum total variance on ``floor_grid``, from the
            previous expiry.
        penalty_weight: How hard the constraints push relative to the quotes.

    Returns:
        The calibrated parameters and the weighted RMSE in total variance.

    Raises:
        SviFitError: if there are too few quotes, or no start converges.
    """
    k = np.asarray(log_moneyness, dtype=np.float64)
    w = np.asarray(total_variance, dtype=np.float64)
    if k.size != w.size:
        raise SviFitError("log-moneyness and total variance must be the same length")
    if k.size < _MIN_QUOTES:
        raise SviFitError(f"need at least {_MIN_QUOTES} quotes to fit a slice, got {k.size}")

    scale = np.ones_like(w) if weights is None else np.asarray(weights, dtype=np.float64)
    scale = scale / np.maximum(scale.sum(), 1e-12)

    grid = butterfly_grid if butterfly_grid is not None else np.linspace(-1.5, 1.5, 61)
    root_penalty = np.sqrt(penalty_weight)

    def residuals(params: FloatArray) -> FloatArray:
        a, b, rho, m, sigma = params
        shifted = k - m
        model = a + b * (rho * shifted + np.sqrt(shifted**2 + sigma**2))
        fit_error = np.sqrt(scale) * (model - w)

        candidate = SviParameters(*(float(p) for p in params))
        butterfly = root_penalty * _hinge(-candidate.butterfly_g(grid))
        if variance_floor is None or floor_grid is None:
            return np.concatenate([fit_error, butterfly])
        shortfall = root_penalty * _hinge(variance_floor - candidate.total_variance(floor_grid))
        return np.concatenate([fit_error, butterfly, shortfall])

    lower = np.array([-1.0, 1e-8, -0.999, -2.0, 1e-4])
    upper = np.array([2.0, 10.0, 0.999, 2.0, 5.0])

    level = float(np.median(w))
    spread = float(max(k.max() - k.min(), 1e-3))
    starts = [
        np.array([level * 0.5, 0.10, -0.5, 0.0, 0.10]),
        np.array([level * 0.9, 0.05, 0.0, 0.0, 0.20]),
        np.array([level * 0.5, 0.20, -0.8, float(np.mean(k)), spread / 2.0]),
        np.array([level * 0.5, 0.20, 0.5, float(np.mean(k)), spread / 2.0]),
    ]

    best: SviParameters | None = None
    best_cost = np.inf
    for start in starts:
        guess = np.clip(start, lower + 1e-9, upper - 1e-9)
        try:
            solution = least_squares(
                residuals, guess, bounds=(lower, upper), method="trf", max_nfev=600
            )
        except ValueError:  # pragma: no cover - guarded by the clip above
            continue
        if solution.cost < best_cost:
            best_cost = float(solution.cost)
            best = SviParameters(*(float(p) for p in solution.x))

    if best is None:  # pragma: no cover - least_squares does not fail silently
        raise SviFitError("no SVI start converged")

    rmse = float(np.sqrt(np.mean((best.total_variance(k) - w) ** 2)))
    return best, rmse
