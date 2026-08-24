"""Bootstrapping a zero curve from the Treasury constant-maturity par curve.

The published H.15 constant-maturity series are *par yields*, not zero rates:
the 10-year quote is the coupon a par bond maturing in ten years would pay. To
discount anything, those have to be turned into zero rates, and that is done
sequentially — each instrument adds one knot, using the knots already solved for
the coupons that land before it.

Two conventions apply, matching how the Treasury actually quotes:

* **One year and shorter** are bill-derived coupon-equivalent yields, treated as
  simple-interest zero rates: ``DF = 1 / (1 + y * t)``.
* **Two years and longer** are semiannual-coupon par bonds, solved so that the
  bond prices exactly to par off the curve being built.

The solver is a bracketed root find rather than the textbook closed-form
recursion, because coupon dates of a longer bond can fall between existing
knots: the unknown zero rate affects the interpolation of its own earlier
coupons, so the equation is implicit.

A sequential pass alone is only exact when the interpolation is local. Under a
spline it is not: adding the 30-year knot changes how the curve reads between
the 5- and 10-year knots, so an instrument that repriced perfectly two steps ago
quietly stops doing so. The sequential pass is therefore treated as a starting
guess, and a global solve then moves all the par-bond knots together until every
instrument reprices at once. On a local scheme it converges immediately and
changes nothing; on a spline it is what makes the curve honest.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from scipy.optimize import brentq, root

from aegis.conventions import DayCount, Tenor
from aegis.curves.discount import DiscountCurve
from aegis.curves.interpolation import Interpolation

__all__ = ["BootstrapError", "bootstrap_treasury_curve", "curve_from_store"]

#: Tenors at or below this are quoted as money-market zero rates, not par bonds.
_MONEY_MARKET_LIMIT_YEARS = 1.0
_COUPON_FREQUENCY = 2
_SOLVER_BRACKET = (-0.50, 1.00)
_SOLVER_TOLERANCE = 1e-14
#: Worst acceptable par-bond repricing error, in price terms, after the global solve.
_GLOBAL_SOLVE_TOLERANCE = 1e-11


class BootstrapError(RuntimeError):
    """Raised when a curve cannot be bootstrapped from the quotes given."""


def bootstrap_treasury_curve(
    reference_date: date,
    quotes: dict[str, float],
    interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
    day_count: DayCount = DayCount.ACT_365F,
    name: str = "UST",
) -> DiscountCurve:
    """Bootstrap a zero curve from constant-maturity par yields.

    Args:
        reference_date: Valuation date of the resulting curve.
        quotes: Tenor label to par yield as a decimal, e.g. ``{"10Y": 0.0458}``.
        interpolation: Scheme the resulting curve interpolates with. It is used
            *during* the bootstrap too, so the curve reprices its own inputs
            under the same scheme it will be read with.
        day_count: Convention the curve measures time with.
        name: Curve identifier.

    Returns:
        A curve whose knots sit at the quoted tenors.

    Raises:
        BootstrapError: if the quotes are empty or a knot cannot be solved.
    """
    if not quotes:
        raise BootstrapError("no quotes to bootstrap from")

    ordered = sorted(
        ((Tenor.parse(label).approximate_years, label, rate) for label, rate in quotes.items()),
        key=lambda item: item[0],
    )

    times: list[float] = []
    zeros: list[float] = []
    par_bonds: list[tuple[int, float, float]] = []

    for tenor_years, label, par_yield in ordered:
        if tenor_years <= _MONEY_MARKET_LIMIT_YEARS:
            zero = _money_market_zero(tenor_years, par_yield)
        else:
            zero = _solve_par_bond_zero(
                reference_date=reference_date,
                tenor_years=tenor_years,
                par_yield=par_yield,
                times=times,
                zeros=zeros,
                interpolation=interpolation,
                day_count=day_count,
                label=label,
            )
            par_bonds.append((len(times), tenor_years, par_yield))
        times.append(tenor_years)
        zeros.append(zero)

    knot_times = np.array(times, dtype=np.float64)
    knot_zeros = _refine_globally(
        reference_date=reference_date,
        times=knot_times,
        zeros=np.array(zeros, dtype=np.float64),
        par_bonds=par_bonds,
        interpolation=interpolation,
        day_count=day_count,
    )

    return DiscountCurve(
        reference_date=reference_date,
        times=knot_times,
        zero_rates=knot_zeros,
        day_count=day_count,
        interpolation=interpolation,
        name=name,
    )


def _refine_globally(
    reference_date: date,
    times: np.ndarray,
    zeros: np.ndarray,
    par_bonds: list[tuple[int, float, float]],
    interpolation: Interpolation,
    day_count: DayCount,
) -> np.ndarray:
    """Move every par-bond knot together until all instruments reprice at once.

    Money-market knots are held fixed: a discount factor *at* a knot is
    ``exp(-z t)`` whatever the interpolation does between knots, so those are
    already exact and cannot be improved.

    Args:
        reference_date: Curve valuation date.
        times: Knot times, in years.
        zeros: Zero rates from the sequential pass, used as the starting guess.
        par_bonds: Index, maturity and quoted par yield of each coupon instrument.
        interpolation: Scheme the curve is read with.
        day_count: Curve day count.

    Returns:
        The refined zero rates.

    Raises:
        BootstrapError: if the global solve fails to converge.
    """
    if not par_bonds:
        return zeros

    indices = np.array([index for index, _, _ in par_bonds])
    coupon_schedules = [
        (
            np.array(
                [(i + 1) / _COUPON_FREQUENCY for i in range(round(t * _COUPON_FREQUENCY))],
                dtype=np.float64,
            ),
            par_yield / _COUPON_FREQUENCY,
        )
        for _, t, par_yield in par_bonds
    ]

    def residuals(candidate: np.ndarray) -> np.ndarray:
        trial_zeros = zeros.copy()
        trial_zeros[indices] = candidate
        trial = DiscountCurve(
            reference_date=reference_date,
            times=times,
            zero_rates=trial_zeros,
            day_count=day_count,
            interpolation=interpolation,
        )
        errors = np.empty(len(coupon_schedules))
        for position, (coupon_times, coupon) in enumerate(coupon_schedules):
            discounts = trial.discount_factor(coupon_times)
            errors[position] = coupon * discounts.sum() + discounts[-1] - 1.0
        return errors

    # The convergence test is the residual itself, not the solver's status flag:
    # hybr refuses an xtol at machine precision and reports failure even when it
    # has landed on an exact root, which it routinely does here.
    solution = root(residuals, zeros[indices], method="hybr")
    worst = float(np.max(np.abs(residuals(solution.x))))
    if worst > _GLOBAL_SOLVE_TOLERANCE:
        raise BootstrapError(
            f"global curve solve did not converge: worst repricing error {worst:.3e} "
            f"({solution.message})"
        )

    refined = zeros.copy()
    refined[indices] = solution.x
    return refined


def _money_market_zero(tenor_years: float, par_yield: float) -> float:
    """Convert a simple-interest money-market yield to a continuous zero rate."""
    discount = 1.0 / (1.0 + par_yield * tenor_years)
    return float(-np.log(discount) / tenor_years)


def _solve_par_bond_zero(
    reference_date: date,
    tenor_years: float,
    par_yield: float,
    times: list[float],
    zeros: list[float],
    interpolation: Interpolation,
    day_count: DayCount,
    label: str,
) -> float:
    """Solve the zero rate at ``tenor_years`` that prices the par bond at 100."""
    coupon = par_yield / _COUPON_FREQUENCY
    count = round(tenor_years * _COUPON_FREQUENCY)
    coupon_times = np.array([(i + 1) / _COUPON_FREQUENCY for i in range(count)], dtype=np.float64)

    def price_error(candidate_zero: float) -> float:
        trial = DiscountCurve(
            reference_date=reference_date,
            times=np.array([*times, tenor_years], dtype=np.float64),
            zero_rates=np.array([*zeros, candidate_zero], dtype=np.float64),
            day_count=day_count,
            interpolation=interpolation,
        )
        discounts = trial.discount_factor(coupon_times)
        return float(coupon * discounts.sum() + discounts[-1] - 1.0)

    low, high = _SOLVER_BRACKET
    if price_error(low) * price_error(high) > 0:
        raise BootstrapError(
            f"cannot bracket a zero rate for {label} at a par yield of {par_yield:.4%}"
        )
    solved = brentq(price_error, low, high, xtol=_SOLVER_TOLERANCE, maxiter=200)
    return float(solved)


def curve_from_store(
    quotes: pl.DataFrame,
    value_date: date,
    interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT,
    curve_name: str = "UST",
) -> DiscountCurve:
    """Bootstrap a curve for one session out of rows read from the store.

    Args:
        quotes: Rows from the ``curve_point`` table; must cover ``value_date``.
        value_date: The session to build.
        interpolation: Scheme for the resulting curve.
        curve_name: Which curve in the frame to use.

    Returns:
        The bootstrapped curve for that session.

    Raises:
        BootstrapError: if the frame carries no quotes for that date and curve.
    """
    selected = quotes.filter((pl.col("value_date") == value_date) & (pl.col("curve") == curve_name))
    if selected.is_empty():
        raise BootstrapError(f"no {curve_name} quotes for {value_date}")

    mapping = dict(zip(selected["tenor"].to_list(), selected["rate"].to_list(), strict=True))
    return bootstrap_treasury_curve(
        value_date, mapping, interpolation=interpolation, name=curve_name
    )
