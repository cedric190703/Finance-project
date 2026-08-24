"""The discount curve.

Everything downstream — bond prices, option forwards, carry, DV01 — is a
question asked of this object. It stores continuously compounded zero rates at a
set of knots and answers discount factors, zero rates and forward rates in
between.

Curves are immutable. A bump returns a new curve, which is what makes the risk
engine in phase 7 safe to run in parallel: nothing can mutate the curve another
scenario is still reading.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import numpy.typing as npt

from aegis.conventions import DayCount, Tenor
from aegis.curves.interpolation import Interpolation, interpolate_zero_rates

__all__ = ["DiscountCurve"]

FloatArray = npt.NDArray[np.float64]

_BASIS_POINT = 1e-4


@dataclass(frozen=True)
class DiscountCurve:
    """A zero-coupon discount curve.

    Attributes:
        reference_date: The curve's valuation date; time is measured from here.
        times: Knot times in years, strictly increasing.
        zero_rates: Continuously compounded zero rates at the knots.
        day_count: Convention used to turn dates into year fractions.
        interpolation: Scheme used between knots.
        name: Identifier, e.g. ``"UST"``.
    """

    reference_date: date
    times: FloatArray
    zero_rates: FloatArray
    day_count: DayCount = DayCount.ACT_365F
    interpolation: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT
    name: str = "UST"

    def __post_init__(self) -> None:
        """Validate the knot structure."""
        times = np.asarray(self.times, dtype=np.float64)
        zeros = np.asarray(self.zero_rates, dtype=np.float64)
        if times.size != zeros.size or times.size == 0:
            raise ValueError("times and zero_rates must be non-empty and the same length")
        if np.any(times <= 0):
            raise ValueError("curve times must be positive; the reference date is time zero")
        if np.any(np.diff(times) <= 0):
            raise ValueError("curve times must be strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "zero_rates", zeros)

    # ------------------------------------------------------------------ lookups

    def year_fraction(self, day: date) -> float:
        """Return the time in years from the reference date to a date.

        Args:
            day: The date to measure to.

        Returns:
            The year fraction under the curve's day count.
        """
        return self.day_count.year_fraction(self.reference_date, day)

    def zero_rate(self, time: float | FloatArray) -> FloatArray:
        """Return the continuously compounded zero rate at one or more times.

        Args:
            time: Time in years, or an array of them.

        Returns:
            The interpolated zero rates.
        """
        targets = np.atleast_1d(np.asarray(time, dtype=np.float64))
        return interpolate_zero_rates(self.times, self.zero_rates, targets, self.interpolation)

    def discount_factor(self, time: float | FloatArray) -> FloatArray:
        """Return the discount factor for one or more times.

        Args:
            time: Time in years, or an array of them. Times at or before zero
                discount at one: a cash flow that has already happened is not
                discounted, it is simply not in the future.

        Returns:
            ``exp(-z(t) * t)`` for each time.
        """
        targets = np.atleast_1d(np.asarray(time, dtype=np.float64))
        rates = self.zero_rate(targets)
        return np.where(targets <= 0.0, 1.0, np.exp(-rates * targets))

    def discount_to(self, day: date) -> float:
        """Return the discount factor to a calendar date.

        Args:
            day: The date to discount to.

        Returns:
            The discount factor.
        """
        return float(self.discount_factor(self.year_fraction(day))[0])

    def forward_rate(self, start: float, end: float) -> float:
        """Return the continuously compounded forward rate between two times.

        Args:
            start: Start of the forward period, in years.
            end: End of the forward period, in years.

        Returns:
            The forward rate.

        Raises:
            ValueError: if the period is not strictly positive.
        """
        if end <= start:
            raise ValueError(f"forward period must be positive, got [{start}, {end}]")
        df_start = float(self.discount_factor(start)[0])
        df_end = float(self.discount_factor(end)[0])
        return float(np.log(df_start / df_end) / (end - start))

    def par_rate(self, tenor_years: float, frequency: int = 2) -> float:
        """Return the par coupon rate of a bond of a given maturity.

        This is the inverse of the bootstrap: feeding a bootstrapped curve back
        through it must recover the yields it was built from, which is the
        round-trip test the phase leans on.

        Args:
            tenor_years: Maturity in years.
            frequency: Coupon payments per year.

        Returns:
            The annualised par coupon rate.

        Raises:
            ValueError: if the maturity is not positive.
        """
        if tenor_years <= 0:
            raise ValueError("tenor must be positive")
        times = self._coupon_times(tenor_years, frequency)
        discounts = self.discount_factor(times)
        annuity = float(discounts.sum()) / frequency
        return float((1.0 - discounts[-1]) / annuity)

    def _coupon_times(self, tenor_years: float, frequency: int) -> FloatArray:
        count = max(round(tenor_years * frequency), 1)
        return np.array([(i + 1) / frequency for i in range(count)], dtype=np.float64)

    # -------------------------------------------------------------------- bumps

    def shift_parallel(self, basis_points: float) -> DiscountCurve:
        """Return a copy with every zero rate shifted.

        Args:
            basis_points: Size of the shift, in basis points.

        Returns:
            A new curve; the original is untouched.
        """
        return replace(self, zero_rates=self.zero_rates + basis_points * _BASIS_POINT)

    def shift_key_rate(
        self, tenor_years: float, basis_points: float, width_years: float | None = None
    ) -> DiscountCurve:
        """Return a copy with a tent-shaped bump centred on one tenor.

        Key-rate bumps are how a book's rate exposure is decomposed: the sum of
        the key-rate DV01s reconciles to the parallel DV01, and the shape of the
        decomposition is what tells a desk whether it is long the front end and
        short the back.

        Args:
            tenor_years: Centre of the bump, in years.
            basis_points: Peak size of the bump, in basis points.
            width_years: Half-width of the tent. Defaults to the distance to the
                neighbouring knots, which makes the bumps sum to a parallel shift.

        Returns:
            A new curve with the bump applied.
        """
        weights = self._tent_weights(tenor_years, width_years)
        return replace(self, zero_rates=self.zero_rates + weights * basis_points * _BASIS_POINT)

    def _tent_weights(self, centre: float, width: float | None) -> FloatArray:
        times = self.times
        index = int(np.argmin(np.abs(times - centre)))
        left = times[index - 1] if index > 0 else times[index] - (width or 1.0)
        right = times[index + 1] if index < times.size - 1 else times[index] + (width or 1.0)
        if width is not None:
            left, right = times[index] - width, times[index] + width

        weights = np.zeros_like(times)
        rising = (times > left) & (times <= times[index])
        falling = (times > times[index]) & (times < right)
        weights[rising] = (times[rising] - left) / (times[index] - left)
        weights[falling] = (right - times[falling]) / (right - times[index])
        weights[index] = 1.0
        return weights

    # ------------------------------------------------------------- construction

    @classmethod
    def flat(
        cls,
        reference_date: date,
        rate: float,
        horizon_years: float = 30.0,
        day_count: DayCount = DayCount.ACT_365F,
        name: str = "FLAT",
    ) -> DiscountCurve:
        """Build a flat curve, mostly for tests and analytic cross-checks.

        Args:
            reference_date: The curve's valuation date.
            rate: The continuously compounded zero rate, applied everywhere.
            horizon_years: Where to place the far knot.
            day_count: Convention for date-to-time conversion.
            name: Curve identifier.

        Returns:
            A curve that discounts at ``exp(-rate * t)`` for every ``t``.
        """
        return cls(
            reference_date=reference_date,
            times=np.array([1.0 / 365.0, horizon_years], dtype=np.float64),
            zero_rates=np.array([rate, rate], dtype=np.float64),
            day_count=day_count,
            interpolation=Interpolation.LINEAR_ZERO,
            name=name,
        )

    def with_interpolation(self, scheme: Interpolation) -> DiscountCurve:
        """Return the same knots read under a different interpolation scheme.

        Args:
            scheme: The scheme to switch to.

        Returns:
            A new curve.
        """
        return replace(self, interpolation=scheme)

    def knot_table(self) -> list[tuple[str, float, float, float]]:
        """Return the curve as rows for display.

        Returns:
            Tuples of tenor label, time in years, zero rate and discount factor.
        """
        discounts = self.discount_factor(self.times)
        return [
            (_label(t), float(t), float(z), float(df))
            for t, z, df in zip(self.times, self.zero_rates, discounts, strict=True)
        ]


def _label(time_years: float) -> str:
    if time_years < 1.0:
        return str(Tenor(round(time_years * 12), Tenor.parse("1M").unit))
    return str(Tenor(round(time_years), Tenor.parse("1Y").unit))
