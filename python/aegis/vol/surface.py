"""Building a volatility surface from a listed option chain.

The pipeline, in the order a desk would defend it:

1. **Clean the quotes.** Drop anything not two-sided, anything whose spread is
   wider than its mid, and anything with no open interest. A screen full of
   stale wing quotes will otherwise dominate a least-squares fit.
2. **Read the forward off the market.** Do not assume the forward is
   ``S·e^{(r−q)T}`` with a dividend yield picked from somewhere. Put-call parity
   says ``C − P = DF·(F − K)``, so with the discount factor from the rates curve
   every paired strike implies a forward directly. Whatever the market thinks of
   dividends, borrow and repo is already in there; the honest thing is to listen
   rather than to assume a yield.
3. **Invert each mid to an implied volatility** against that forward.
4. **Fit an SVI slice per expiry** in total variance, weighted by vega so the
   fit cares about the strikes where the price actually responds to vol.
5. **Check for arbitrage** — butterfly within each slice, calendar across them —
   and report what fails rather than quietly shipping a surface that admits
   free money.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

from aegis.conventions import DayCount
from aegis.curves import DiscountCurve
from aegis.pricing.black_scholes import OptionRight, black_vega, implied_volatility
from aegis.vol.svi import SviFitError, SviParameters, fit_svi_slice

__all__ = ["ArbitrageReport", "SurfaceError", "VolSlice", "VolSurface", "build_surface"]

FloatArray = npt.NDArray[np.float64]

#: Quotes further than this in log-moneyness are ignored: they are mostly noise.
_MAX_ABS_LOG_MONEYNESS = 1.5
#: Slices with fewer usable quotes than this cannot be calibrated meaningfully.
_MIN_QUOTES_PER_SLICE = 6
#: Expiries closer than this are dropped; a few days to expiry is all gamma and
#: no vol, and the inversion is numerically hopeless.
_MIN_DAYS_TO_EXPIRY = 7
#: Where the fit's soft constraints are sampled.
_PENALTY_POINTS = 61
#: Where the repairs are checked. Deliberately finer than the penalty grid: a
#: lift that only closes the gap at the penalty nodes leaves it open between
#: them, and the arbitrage check samples somewhere else again.
_REPAIR_POINTS = 601
#: Two expiries must share at least this much log-moneyness before their total
#: variances are compared at all.
_MIN_OVERLAP = 0.05
#: How many times the two repairs may be alternated before giving up.
_REPAIR_ROUNDS = 12
#: Bisection steps used to find the largest arbitrage-free wing slope.
_BISECTION_STEPS = 60


class SurfaceError(RuntimeError):
    """Raised when no usable surface can be built from a chain."""


@dataclass(frozen=True)
class VolSlice:
    """One expiry's calibrated smile.

    Attributes:
        expiry: The expiry date.
        time: Time to expiry in years.
        forward: Forward implied by put-call parity against the curve discount.
        discount: Discount factor used, taken from the rates curve.
        regressed_discount: Discount factor the unconstrained parity regression
            implies. A diagnostic only: when it strays far from ``discount`` the
            expiry's quotes are too sparse or too wide to trust.
        parameters: The calibrated SVI parameters.
        rmse_total_variance: Weighted RMSE of the fit, in total-variance units.
        quote_count: How many cleaned quotes the fit used.
        quoted_range: Lowest and highest log-moneyness the fit actually saw.
            Outside it the slice is an extrapolation, and it is treated as one.
        repaired: Whether the no-arbitrage repairs had to move the fitted
            parameters. A repaired slice is arbitrage-free but no longer the
            best fit to its quotes, and on a thin expiry the repair can flatten
            the smile entirely. That is a real limitation, not a detail to hide:
            it is reported here and it shows up as an elevated RMSE.
    """

    expiry: date
    time: float
    forward: float
    discount: float
    regressed_discount: float
    parameters: SviParameters
    rmse_total_variance: float
    quote_count: int
    quoted_range: tuple[float, float]
    repaired: bool

    def implied_vol(self, strike: float | FloatArray) -> FloatArray:
        """Return the fitted implied volatility at one or more strikes.

        Args:
            strike: Strike price, scalar or array.

        Returns:
            Annualised implied volatilities.
        """
        strikes = np.atleast_1d(np.asarray(strike, dtype=np.float64))
        log_moneyness = np.log(np.maximum(strikes, 1e-12) / self.forward)
        return self.parameters.implied_vol(log_moneyness, self.time)

    @property
    def atm_vol(self) -> float:
        """Return the at-the-money-forward implied volatility."""
        return float(self.parameters.implied_vol(0.0, self.time)[0])

    @property
    def implied_rate(self) -> float:
        """Return the continuously compounded rate implied by the parity fit."""
        return float(-np.log(max(self.discount, 1e-12)) / self.time)


@dataclass(frozen=True)
class ArbitrageReport:
    """What the no-arbitrage checks found.

    The worst magnitudes are reported alongside the failures on purpose. The
    calendar and butterfly constraints are enforced as penalties during the fit,
    so where they bind they bind exactly, and the optimizer leaves slack of
    order 1e-7 in total variance behind. Calling that an arbitrage would be
    theatre: it is four orders of magnitude below a hundredth of a vol point,
    and no butterfly spread could be lifted for it. What matters is not whether
    a violation exists but how big it is, so both are reported.

    Attributes:
        butterfly_failures: Expiries whose fitted smile implies a negative density.
        calendar_failures: Adjacent expiry pairs whose total variance decreases.
        worst_butterfly: Most negative value of ``g(k)`` seen, in the same units.
        worst_calendar: Largest decrease in total variance across an expiry pair.
        checked_slices: How many slices were examined.
        tolerance: The threshold a violation had to exceed to be reported.
    """

    butterfly_failures: tuple[date, ...]
    calendar_failures: tuple[tuple[date, date], ...]
    worst_butterfly: float
    worst_calendar: float
    checked_slices: int
    tolerance: float

    @property
    def is_clean(self) -> bool:
        """Return whether the surface passed every check."""
        return not self.butterfly_failures and not self.calendar_failures

    def __str__(self) -> str:
        """Return a one-line summary."""
        if self.is_clean:
            return (
                f"{self.checked_slices} slices, no arbitrage above {self.tolerance:.0e} "
                f"(worst butterfly {self.worst_butterfly:+.2e}, "
                f"calendar {self.worst_calendar:+.2e})"
            )
        return (
            f"{self.checked_slices} slices, "
            f"{len(self.butterfly_failures)} butterfly (worst {self.worst_butterfly:+.2e}) and "
            f"{len(self.calendar_failures)} calendar (worst {self.worst_calendar:+.2e}) violations"
        )


@dataclass(frozen=True)
class VolSurface:
    """A calibrated volatility surface.

    Attributes:
        underlying: The underlying ticker.
        reference_date: Valuation date.
        spot: Spot price observed with the chain.
        slices: Calibrated slices, ordered by expiry.
    """

    underlying: str
    reference_date: date
    spot: float
    slices: tuple[VolSlice, ...]

    def slice_for(self, expiry: date) -> VolSlice:
        """Return the calibrated slice for an expiry.

        Args:
            expiry: The expiry to look up.

        Returns:
            The matching slice.

        Raises:
            KeyError: if that expiry was not calibrated.
        """
        for candidate in self.slices:
            if candidate.expiry == expiry:
                return candidate
        raise KeyError(f"no calibrated slice for {expiry}")

    def implied_vol(self, strike: float, expiry: date) -> float:
        """Return the implied volatility for any strike and expiry.

        Between calibrated expiries the surface interpolates in *total
        variance* at fixed log-moneyness, which is the interpolation that
        preserves the calendar no-arbitrage condition. Interpolating implied
        volatility directly does not.

        Args:
            strike: Strike price.
            expiry: Expiry date, calibrated or not.

        Returns:
            The implied volatility.

        Raises:
            SurfaceError: if the surface has no slices.
        """
        if not self.slices:
            raise SurfaceError("surface has no calibrated slices")

        time = DayCount.ACT_365F.year_fraction(self.reference_date, expiry)
        if time <= 0:
            raise SurfaceError(f"expiry {expiry} is not in the future")

        times = np.array([s.time for s in self.slices])
        if time <= times[0]:
            return float(self.slices[0].implied_vol(strike)[0])
        if time >= times[-1]:
            return float(self.slices[-1].implied_vol(strike)[0])

        upper = int(np.searchsorted(times, time))
        left, right = self.slices[upper - 1], self.slices[upper]
        # Same log-moneyness on both sides, each against its own forward.
        w_left = float(left.parameters.total_variance(np.log(strike / left.forward))[0])
        w_right = float(right.parameters.total_variance(np.log(strike / right.forward))[0])
        weight = (time - left.time) / (right.time - left.time)
        total_variance = (1.0 - weight) * w_left + weight * w_right
        return float(np.sqrt(max(total_variance, 0.0) / time))

    def check_arbitrage(
        self, span: float = 1.5, points: int = 121, tolerance: float = 1e-6
    ) -> ArbitrageReport:
        """Run the butterfly and calendar no-arbitrage checks.

        Args:
            span: How far into the wings to check, in log-moneyness. The density
                condition is checked across the whole span; the calendar
                condition only where two expiries are both quoted.
            points: How finely to sample.
            tolerance: How large a violation must be before it counts. The
                default sits well below anything tradeable and well above the
                slack a penalised fit leaves on a binding constraint.

        Returns:
            The report; a violation is information, not an exception.
        """
        grid = np.linspace(-span, span, points)

        butterfly: list[date] = []
        worst_butterfly = 0.0
        for candidate in self.slices:
            g = candidate.parameters.butterfly_g(grid)
            worst_butterfly = min(worst_butterfly, float(g.min()))
            if np.any(g < -tolerance):
                butterfly.append(candidate.expiry)

        calendar: list[tuple[date, date]] = []
        worst_calendar = 0.0
        for earlier, later in zip(self.slices, self.slices[1:], strict=False):
            # Compared only where both expiries are quoted. Outside that range
            # the comparison is between two extrapolations, and a crossing there
            # says something about the parameterisation, not about a trade
            # anybody could put on.
            low = max(earlier.quoted_range[0], later.quoted_range[0])
            high = min(earlier.quoted_range[1], later.quoted_range[1])
            if high - low < _MIN_OVERLAP:
                continue
            shared = grid[(grid >= low) & (grid <= high)]
            gap = later.parameters.total_variance(shared) - earlier.parameters.total_variance(
                shared
            )
            worst_calendar = min(worst_calendar, float(gap.min()))
            if np.any(gap < -tolerance):
                calendar.append((earlier.expiry, later.expiry))

        return ArbitrageReport(
            tuple(butterfly),
            tuple(calendar),
            worst_butterfly,
            worst_calendar,
            len(self.slices),
            tolerance,
        )

    def scaled(self, factor: float) -> VolSurface:
        """Return the surface with every implied volatility multiplied by a factor.

        Total variance is the square of volatility, so scaling volatility by
        ``f`` means scaling ``w`` by ``f²`` — and because SVI is affine in ``a``
        and ``b``, that is exactly scaling those two parameters. The skew, the
        smile's centre and its curvature are untouched, which is what a parallel
        volatility shock is supposed to mean.

        Args:
            factor: Multiplier applied to every implied volatility.

        Returns:
            A new surface; the original is untouched.

        Raises:
            ValueError: if the factor is not positive.
        """
        if factor <= 0.0:
            raise ValueError("volatility scale factor must be positive")
        squared = factor * factor
        shocked = tuple(
            replace(
                s,
                parameters=replace(
                    s.parameters, a=s.parameters.a * squared, b=s.parameters.b * squared
                ),
            )
            for s in self.slices
        )
        return replace(self, slices=shocked)

    def to_frame(self) -> pl.DataFrame:
        """Return one row per calibrated slice, for display or storage.

        Returns:
            A frame of slice-level diagnostics.
        """
        return pl.DataFrame(
            {
                "expiry": [s.expiry for s in self.slices],
                "years": [s.time for s in self.slices],
                "forward": [s.forward for s in self.slices],
                "discount": [s.discount for s in self.slices],
                "regressed_discount": [s.regressed_discount for s in self.slices],
                "implied_rate": [s.implied_rate for s in self.slices],
                "atm_vol": [s.atm_vol for s in self.slices],
                "rmse_total_variance": [s.rmse_total_variance for s in self.slices],
                "quotes": [s.quote_count for s in self.slices],
                "repaired": [s.repaired for s in self.slices],
                "butterfly_free": [s.parameters.is_butterfly_free() for s in self.slices],
            }
        )


def implied_forward(
    strikes: FloatArray,
    call_mid: FloatArray,
    put_mid: FloatArray,
    discount: float | None = None,
) -> tuple[float, float]:
    """Extract the forward from put-call parity.

    ``C − P = DF·(F − K)`` is linear in the strike: the slope is ``−DF`` and the
    intercept is ``DF·F``, so a least-squares line through the strikes recovers
    both from the option quotes alone.

    That two-parameter fit is only trustworthy when the strike range is wide and
    the quotes are tight. On a three-week expiry with a handful of paired
    strikes it is not: the slope is estimated over a short lever arm from noisy
    mids, and it routinely comes back implying a discount factor above one — a
    negative interest rate that nobody traded. So when a discount factor is
    supplied it is taken as given and only the forward is implied, which is what
    a desk does in practice. Each strike gives ``F = K + (C − P)/DF``, and those
    estimates are averaged with weight on the strikes nearest the money, where
    both legs carry real time value.

    Args:
        strikes: Strikes at which both a call and a put are quoted.
        call_mid: Call mid prices.
        put_mid: Put mid prices.
        discount: Discount factor to expiry. When ``None``, both the forward and
            the discount factor are regressed out of the quotes.

    Returns:
        The implied forward and the discount factor used.

    Raises:
        SurfaceError: if there are too few strikes, or the fit is degenerate.
    """
    if strikes.size < 3:
        raise SurfaceError("need at least three paired strikes to imply a forward")

    if discount is None:
        basis = np.vstack([np.ones_like(strikes), strikes]).T
        coefficients, *_ = np.linalg.lstsq(basis, call_mid - put_mid, rcond=None)
        intercept, slope = float(coefficients[0]), float(coefficients[1])
        regressed = -slope
        if not 0.0 < regressed <= 1.5:
            raise SurfaceError(
                f"parity fit implied an impossible discount factor of {regressed:.4f}"
            )
        return intercept / regressed, regressed

    if not 0.0 < discount <= 1.0:
        raise SurfaceError(f"discount factor must lie in (0, 1], got {discount:.4f}")

    per_strike = strikes + (call_mid - put_mid) / discount
    centre = float(np.median(per_strike))
    weights = 1.0 / (1.0 + ((strikes - centre) / max(centre, 1e-8) / 0.05) ** 2)
    return float(np.average(per_strike, weights=weights)), discount


def build_surface(
    chain: pl.DataFrame,
    curve: DiscountCurve,
    reference_date: date | None = None,
    min_open_interest: int = 1,
) -> VolSurface:
    """Build a calibrated surface from a cleaned option chain.

    Args:
        chain: Rows as returned by the Cboe adapter.
        curve: Discount curve, used only as a cross-check against the discount
            factor the option quotes imply.
        reference_date: Valuation date; defaults to the chain's own value date.
        min_open_interest: Contracts with less open interest than this are dropped.

    Returns:
        The calibrated surface.

    Raises:
        SurfaceError: if no expiry yields enough usable quotes.
    """
    if chain.is_empty():
        raise SurfaceError("empty option chain")

    as_of = reference_date or chain["value_date"][0]
    spot = float(chain["spot"][0])
    underlying = str(chain["underlying"][0])

    quotes = _clean(chain, min_open_interest)
    if quotes.is_empty():
        raise SurfaceError("no quotes survived cleaning")

    # Expiries are calibrated front to back so each slice can be floored by the
    # one before it. Total variance that decreases with maturity is a calendar
    # arbitrage, and the cheapest place to prevent it is during the fit.
    butterfly_grid = np.linspace(-_MAX_ABS_LOG_MONEYNESS, _MAX_ABS_LOG_MONEYNESS, _PENALTY_POINTS)
    slices: list[VolSlice] = []
    previous: VolSlice | None = None
    for (expiry,), group in quotes.sort("expiry").group_by(["expiry"], maintain_order=True):
        assert isinstance(expiry, date)  # noqa: S101 - group key type is known
        time = DayCount.ACT_365F.year_fraction(as_of, expiry)
        if time * 365.0 < _MIN_DAYS_TO_EXPIRY:
            continue
        try:
            calibrated = _fit_slice(group, expiry, time, curve, butterfly_grid, previous)
        except (SurfaceError, SviFitError):
            continue
        slices.append(calibrated)
        previous = calibrated

    if not slices:
        raise SurfaceError(f"no expiry in the {underlying} chain produced a usable slice")

    return VolSurface(underlying, as_of, spot, tuple(sorted(slices, key=lambda s: s.expiry)))


def _overlap_grid(
    quoted_range: tuple[float, float], previous: VolSlice | None
) -> FloatArray | None:
    """Return the log-moneyness range both expiries are quoted over.

    Args:
        quoted_range: The current slice's quoted range.
        previous: The previous expiry's slice, or ``None``.

    Returns:
        A grid spanning the overlap, or ``None`` when there is no previous
        expiry or the two ranges barely meet.
    """
    if previous is None:
        return None
    low = max(quoted_range[0], previous.quoted_range[0])
    high = min(quoted_range[1], previous.quoted_range[1])
    if high - low < _MIN_OVERLAP:
        return None
    return np.linspace(low, high, _PENALTY_POINTS)


def _enforce_constraints(
    parameters: SviParameters,
    previous: VolSlice | None,
    floor_grid: FloatArray | None,
) -> SviParameters:
    """Repair the slack a penalised fit leaves on its constraints.

    Both constraints are enforced during the fit as soft penalties, so where
    they bind the optimizer stops just short — by around 1e-5 in total variance,
    or 1e-4 in the density function. Those are untradeable amounts, but they are
    still violations, and the honest fix is to remove them rather than to widen
    the tolerance until they stop being reported.

    Each has a natural repair, and each repair uses the one parameter that moves
    the constraint without reshaping the smile where the quotes are:

    * **Calendar.** Raising ``a`` lifts total variance by a constant at every
      strike, closing a gap below the previous expiry exactly.
    * **Butterfly.** Shrinking ``b`` flattens the wings, which is what a
      negative implied density is objecting to. The largest ``b`` that restores
      non-negativity is found by bisection, so the correction is the smallest
      one that works.

    The two interact — flattening the wings lowers total variance, which can
    reopen a calendar gap — so they are alternated to a fixed point. That
    terminates: in the limit ``b → 0`` the slice is flat at ``a``, which is both
    butterfly-free and liftable above any floor.

    Args:
        parameters: The fitted slice.
        previous: The previous expiry's slice, or ``None`` for the front one.
        floor_grid: Where the calendar condition applies — the range both
            expiries are quoted over.

    Returns:
        A slice satisfying both constraints, as close to the fit as possible.
    """
    if previous is None or floor_grid is None:
        dense, floor = None, None
    else:
        dense = np.linspace(float(floor_grid[0]), float(floor_grid[-1]), _REPAIR_POINTS)
        floor = previous.parameters.total_variance(dense)
    repaired = parameters

    for _ in range(_REPAIR_ROUNDS):
        if floor is not None and dense is not None:
            shortfall = float(np.max(floor - repaired.total_variance(dense)))
            if shortfall > 0.0:
                repaired = replace(repaired, a=repaired.a + shortfall)

        if repaired.is_butterfly_free():
            if floor is None or dense is None or np.all(repaired.total_variance(dense) >= floor):
                return repaired
            continue

        low, high = 0.0, 1.0
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (low + high)
            if replace(repaired, b=repaired.b * mid).is_butterfly_free():
                low = mid
            else:
                high = mid
        repaired = replace(repaired, b=repaired.b * low)

    return repaired


def _clean(chain: pl.DataFrame, min_open_interest: int) -> pl.DataFrame:
    """Keep only quotes a desk would be willing to fit to."""
    return (
        chain.filter(
            (pl.col("bid") > 0.0)
            & (pl.col("ask") > pl.col("bid"))
            & (pl.col("open_interest") >= min_open_interest)
        )
        .with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
            (pl.col("ask") - pl.col("bid")).alias("spread"),
        )
        # A spread wider than the mid means nobody is really quoting it.
        .filter(pl.col("spread") < pl.col("mid"))
        .sort(["expiry", "strike", "option_right"])
    )


def _fit_slice(
    group: pl.DataFrame,
    expiry: date,
    time: float,
    curve: DiscountCurve,
    butterfly_grid: FloatArray,
    previous: VolSlice | None,
) -> VolSlice:
    """Imply the forward, invert the mids and calibrate one SVI slice."""
    calls = group.filter(pl.col("option_right") == "C").select("strike", "mid")
    puts = group.filter(pl.col("option_right") == "P").select("strike", "mid")
    paired = calls.join(puts, on="strike", how="inner", suffix="_put")
    if paired.height < 3:
        raise SurfaceError(f"{expiry}: too few paired strikes to imply a forward")

    strikes = paired["strike"].to_numpy()
    call_mid, put_mid = paired["mid"].to_numpy(), paired["mid_put"].to_numpy()
    discount = float(curve.discount_factor(time)[0])
    forward, _ = implied_forward(strikes, call_mid, put_mid, discount=discount)
    try:
        _, regressed_discount = implied_forward(strikes, call_mid, put_mid)
    except SurfaceError:
        regressed_discount = float("nan")

    # Out-of-the-money quotes carry the information: an in-the-money option is
    # mostly intrinsic value, so its vol is inferred from a sliver of premium.
    otm = group.filter(
        ((pl.col("option_right") == "C") & (pl.col("strike") >= forward))
        | ((pl.col("option_right") == "P") & (pl.col("strike") < forward))
    )
    log_moneyness = np.log(otm["strike"].to_numpy() / forward)
    inside = np.abs(log_moneyness) <= _MAX_ABS_LOG_MONEYNESS
    if int(inside.sum()) < _MIN_QUOTES_PER_SLICE:
        raise SurfaceError(f"{expiry}: only {int(inside.sum())} usable quotes")

    strikes_used = otm["strike"].to_numpy()[inside]
    mids = otm["mid"].to_numpy()[inside]
    rights = otm["option_right"].to_numpy()[inside]
    log_moneyness = log_moneyness[inside]

    vols = np.empty_like(mids)
    for right in (OptionRight.CALL, OptionRight.PUT):
        mask = rights == right.value
        if not np.any(mask):
            continue
        vols[mask] = implied_volatility(
            mids[mask], forward, strikes_used[mask], time, discount, right
        )

    usable = np.isfinite(vols) & (vols > 1e-4)
    if int(usable.sum()) < _MIN_QUOTES_PER_SLICE:
        raise SurfaceError(f"{expiry}: only {int(usable.sum())} quotes inverted cleanly")

    log_moneyness, vols, strikes_used = log_moneyness[usable], vols[usable], strikes_used[usable]
    total_variance = vols**2 * time
    weights = black_vega(forward, strikes_used, vols, time, discount)

    quoted_range = (float(log_moneyness.min()), float(log_moneyness.max()))
    floor_grid = _overlap_grid(quoted_range, previous)
    parameters, rmse = fit_svi_slice(
        log_moneyness,
        total_variance,
        weights,
        butterfly_grid=butterfly_grid,
        floor_grid=floor_grid,
        variance_floor=(
            None
            if previous is None or floor_grid is None
            else previous.parameters.total_variance(floor_grid)
        ),
    )
    enforced = _enforce_constraints(parameters, previous, floor_grid)
    was_repaired = enforced != parameters
    if was_repaired:
        parameters = enforced
        rmse = float(
            np.sqrt(np.mean((parameters.total_variance(log_moneyness) - total_variance) ** 2))
        )
    return VolSlice(
        expiry=expiry,
        time=time,
        forward=forward,
        discount=discount,
        regressed_discount=regressed_discount,
        parameters=parameters,
        rmse_total_variance=rmse,
        quote_count=int(usable.sum()),
        quoted_range=quoted_range,
        repaired=was_repaired,
    )
