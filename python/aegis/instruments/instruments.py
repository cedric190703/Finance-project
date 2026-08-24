"""The instruments the engine can value.

Each instrument is an immutable description of a contract — what it pays, when,
and in what currency — and nothing else. It knows how to value itself against a
[`MarketSnapshot`][aegis.instruments.market.MarketSnapshot] and how to report its
own risk, but it holds no market data, no cached price and no state that could
go stale between one valuation and the next.

Sensitivities are returned in the units a desk quotes them in, which are not the
units the maths produces:

* **Delta** is the cash change in position value for a 1% move in the underlying,
  not the dimensionless derivative.
* **Vega** is per volatility *point* — a move from 20% to 21% — not per unit.
* **Theta** is per calendar day, not per year.
* **DV01** is the value change for one basis point on the curve, signed so that
  a long bond has a positive number.

Getting these conventions wrong is how a book looks hedged on a report and is
not hedged in the market.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from aegis.conventions import (
    NYSE,
    BusinessDayConvention,
    DayCount,
    Frequency,
    generate_schedule,
)
from aegis.instruments.market import MarketSnapshot
from aegis.pricing import (
    OptionRight,
    black_delta,
    black_gamma,
    black_price,
    black_theta,
    black_vega,
)

__all__ = [
    "Cash",
    "EquityOption",
    "EquityPosition",
    "FixedRateBond",
    "Instrument",
    "Position",
]

_DAYS_PER_YEAR = 365.0
_VOL_POINT = 0.01
_ONE_PERCENT = 0.01
_BASIS_POINT = 1e-4


@dataclass(frozen=True)
class Instrument(ABC):
    """Base class for anything the engine can value.

    Attributes:
        id: Unique identifier, used in reports and reconciliations.
        currency: Currency the instrument settles in.
    """

    id: str
    currency: str

    @abstractmethod
    def present_value(self, market: MarketSnapshot) -> float:
        """Return the value of the position in its own currency.

        Args:
            market: The market state to value against.

        Returns:
            The present value.
        """

    def risk_factors(self) -> tuple[str, ...]:
        """Return the market factors this instrument is exposed to.

        Returns:
            Identifiers such as ``"SPOT:KO"`` or ``"RATE:USD"``. The risk engine
            uses these to decide which bumps are worth running: revaluing an
            equity position under a volatility shock is wasted work.
        """
        return ()

    def sensitivities(self, market: MarketSnapshot) -> dict[str, float]:
        """Return the position's analytic sensitivities.

        Args:
            market: The market state to compute against.

        Returns:
            A mapping of greek name to value, in desk units.
        """
        return {}


@dataclass(frozen=True)
class Cash(Instrument):
    """A cash balance.

    Attributes:
        amount: The balance, positive for an asset.
    """

    amount: float = 0.0

    def present_value(self, market: MarketSnapshot) -> float:
        """Return the balance itself; cash is worth its face value today.

        Args:
            market: Unused; present for interface symmetry.

        Returns:
            The balance.
        """
        del market
        return self.amount


@dataclass(frozen=True)
class EquityPosition(Instrument):
    """A holding in a listed equity or index.

    Attributes:
        symbol: The ticker.
        quantity: Number of shares, negative when short.
    """

    symbol: str = ""
    quantity: float = 0.0

    def present_value(self, market: MarketSnapshot) -> float:
        """Return quantity times spot.

        Args:
            market: The market state.

        Returns:
            The position value.
        """
        return self.quantity * market.spot(self.symbol)

    def risk_factors(self) -> tuple[str, ...]:
        """Return the single spot factor this position is exposed to."""
        return (f"SPOT:{self.symbol}",)

    def sensitivities(self, market: MarketSnapshot) -> dict[str, float]:
        """Return the cash delta for a 1% move in the underlying.

        Args:
            market: The market state.

        Returns:
            A mapping with a single ``delta`` entry.
        """
        return {"delta": self.present_value(market) * _ONE_PERCENT}


@dataclass(frozen=True)
class EquityOption(Instrument):
    """A European option on a listed equity.

    Attributes:
        underlying: Ticker of the underlying.
        strike: Strike price.
        expiry: Expiry date.
        right: Call or put.
        quantity: Number of contracts, negative when short.
        contract_size: Shares per contract; 100 for US listed equity options.
    """

    underlying: str = ""
    strike: float = 0.0
    expiry: date = date(2099, 12, 31)
    right: OptionRight = OptionRight.CALL
    quantity: float = 0.0
    contract_size: float = 100.0

    @property
    def multiplier(self) -> float:
        """Return the number of underlying shares the position represents."""
        return self.quantity * self.contract_size

    def time_to_expiry(self, market: MarketSnapshot) -> float:
        """Return the time to expiry in years, floored at zero.

        Args:
            market: The market state, for its value date.

        Returns:
            Year fraction under ACT/365F.
        """
        return max(DayCount.ACT_365F.year_fraction(market.value_date, self.expiry), 0.0)

    def _terms(self, market: MarketSnapshot) -> tuple[float, float, float, float]:
        """Return the forward, volatility, time and discount factor."""
        time = self.time_to_expiry(market)
        forward = market.forward(self.underlying, self.expiry, self.currency)
        discount = float(market.curve(self.currency).discount_factor(time)[0])
        if time <= 0.0:
            return forward, 0.0, 0.0, discount
        vol = market.implied_vol(self.underlying, self.strike, self.expiry)
        return forward, vol, time, discount

    def present_value(self, market: MarketSnapshot) -> float:
        """Return the option's value, in its settlement currency.

        Args:
            market: The market state.

        Returns:
            The position value, including the contract multiplier.
        """
        forward, vol, time, discount = self._terms(market)
        premium = float(black_price(forward, self.strike, vol, time, discount, self.right)[0])
        return self.multiplier * premium

    def risk_factors(self) -> tuple[str, ...]:
        """Return the spot, volatility and rate factors the option depends on."""
        return (
            f"SPOT:{self.underlying}",
            f"VOL:{self.underlying}",
            f"RATE:{self.currency}",
        )

    def sensitivities(self, market: MarketSnapshot) -> dict[str, float]:
        """Return the option's greeks in desk units.

        The Black formulas are written against the forward, so the forward delta
        and gamma are converted to spot terms by the chain rule: with the forward
        proportional to spot, ``dF/dS = F/S``, and gamma picks that factor up
        twice.

        Args:
            market: The market state.

        Returns:
            Delta, gamma, vega, theta and rho, scaled to the position.
        """
        forward, vol, time, discount = self._terms(market)
        spot = market.spot(self.underlying)
        if time <= 0.0 or spot <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        carry = forward / spot
        forward_delta = float(black_delta(forward, self.strike, vol, time, discount, self.right)[0])
        forward_gamma = float(black_gamma(forward, self.strike, vol, time, discount)[0])
        vega = float(black_vega(forward, self.strike, vol, time, discount)[0])
        rate = float(market.curve(self.currency).zero_rate(time)[0])
        theta = float(black_theta(forward, self.strike, vol, time, discount, self.right, rate)[0])

        spot_delta = forward_delta * carry
        spot_gamma = forward_gamma * carry * carry
        return {
            # Cash delta for a 1% move, and the second-order term for the same move.
            "delta": self.multiplier * spot_delta * spot * _ONE_PERCENT,
            "gamma": 0.5 * self.multiplier * spot_gamma * (spot * _ONE_PERCENT) ** 2,
            # Per volatility point, and per calendar day.
            "vega": self.multiplier * vega * _VOL_POINT,
            "theta": self.multiplier * theta / _DAYS_PER_YEAR,
            # Per basis point on the discount rate.
            "rho": self.multiplier * time * discount * self.strike * _BASIS_POINT,
        }


@dataclass(frozen=True)
class FixedRateBond(Instrument):
    """A fixed-coupon bond, valued off the discount curve.

    Attributes:
        face: Face value of the holding, negative when short.
        coupon: Annual coupon rate as a decimal.
        maturity: Redemption date.
        issue_date: First accrual date.
        frequency: Coupons per year.
        day_count: Accrual convention.
    """

    face: float = 0.0
    coupon: float = 0.0
    maturity: date = date(2099, 12, 31)
    issue_date: date = date(2000, 1, 1)
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCount = DayCount.ACT_ACT_ISDA
    _schedule: tuple[date, ...] = field(default=(), repr=False, compare=False)

    def schedule(self) -> tuple[date, ...]:
        """Return the coupon schedule, generated once and cached.

        Returns:
            Accrual boundary dates, from issue to maturity.
        """
        if self._schedule:
            return self._schedule
        generated = tuple(
            generate_schedule(
                self.issue_date,
                self.maturity,
                self.frequency,
                NYSE,
                BusinessDayConvention.MODIFIED_FOLLOWING,
            )
        )
        object.__setattr__(self, "_schedule", generated)
        return generated

    def remaining_cash_flows(self, value_date: date) -> list[tuple[date, float]]:
        """Return the cash flows still to be paid.

        Args:
            value_date: Flows on or before this date are already gone.

        Returns:
            Pairs of payment date and amount per unit of face.
        """
        schedule = self.schedule()
        rate = self.coupon / self.frequency.value
        flows = [(day, rate) for day in schedule[1:] if day > value_date]
        if flows:
            last_date, last_amount = flows[-1]
            flows[-1] = (last_date, last_amount + 1.0)
        return flows

    def accrued_interest(self, value_date: date) -> float:
        """Return interest accrued since the last coupon, per unit of face.

        Args:
            value_date: The settlement date.

        Returns:
            The accrued amount; zero outside the bond's life.
        """
        schedule = self.schedule()
        if value_date <= schedule[0] or value_date >= schedule[-1]:
            return 0.0
        previous = max(day for day in schedule if day <= value_date)
        following = min(day for day in schedule if day > value_date)
        elapsed = self.day_count.year_fraction(previous, value_date)
        full = self.day_count.year_fraction(previous, following)
        if full <= 0:
            return 0.0
        return (self.coupon / self.frequency.value) * (elapsed / full)

    def dirty_price(self, market: MarketSnapshot) -> float:
        """Return the present value per unit of face, including accrued.

        Args:
            market: The market state.

        Returns:
            The dirty price, where 1.0 is par.
        """
        curve = market.curve(self.currency)
        total = 0.0
        for payment_date, amount in self.remaining_cash_flows(market.value_date):
            total += amount * curve.discount_to(payment_date)
        return total

    def clean_price(self, market: MarketSnapshot) -> float:
        """Return the quoted price: present value less accrued interest.

        Args:
            market: The market state.

        Returns:
            The clean price, where 1.0 is par.
        """
        return self.dirty_price(market) - self.accrued_interest(market.value_date)

    def yield_to_maturity(self, market: MarketSnapshot) -> float:
        """Return the internal rate of return that reproduces the dirty price.

        Args:
            market: The market state.

        Returns:
            The annualised yield, compounded at the coupon frequency.
        """
        from scipy.optimize import brentq

        target = self.dirty_price(market)
        flows = self.remaining_cash_flows(market.value_date)
        if not flows:
            return 0.0
        times = [self.day_count.year_fraction(market.value_date, day) for day, _ in flows]
        amounts = [amount for _, amount in flows]
        frequency = float(self.frequency.value)

        def price_at(rate: float) -> float:
            factor = 1.0 + rate / frequency
            return float(
                sum(
                    amount / factor ** (frequency * time)
                    for amount, time in zip(amounts, times, strict=True)
                )
            )

        low, high = -0.9, 5.0
        if (price_at(low) - target) * (price_at(high) - target) > 0:
            return float("nan")
        solved = brentq(lambda r: price_at(r) - target, low, high, xtol=1e-14)
        return float(solved)

    def present_value(self, market: MarketSnapshot) -> float:
        """Return the value of the holding in its settlement currency.

        Args:
            market: The market state.

        Returns:
            Face times the dirty price.
        """
        return self.face * self.dirty_price(market)

    def risk_factors(self) -> tuple[str, ...]:
        """Return the rate factor this bond is exposed to."""
        return (f"RATE:{self.currency}",)

    def sensitivities(self, market: MarketSnapshot) -> dict[str, float]:
        """Return DV01, modified duration and convexity.

        DV01 comes from an actual one-basis-point bump of the curve rather than
        from a closed form on the yield. The two agree closely for a plain bond,
        but only the bumped number stays right once the curve is not flat — and
        it is the number that reconciles with the key-rate decomposition in the
        risk report.

        Args:
            market: The market state.

        Returns:
            ``dv01``, ``duration`` and ``convexity``.
        """
        curve = market.curve(self.currency)
        base = self.present_value(market)
        up = self.face * _price_on(self, market, curve.shift_parallel(1.0))
        down = self.face * _price_on(self, market, curve.shift_parallel(-1.0))

        dv01 = 0.5 * (down - up)
        convexity = (up + down - 2.0 * base) / (base * _BASIS_POINT**2) if base else 0.0
        duration = dv01 / (base * _BASIS_POINT) if base else 0.0
        return {"dv01": dv01, "duration": duration, "convexity": convexity}


def _price_on(bond: FixedRateBond, market: MarketSnapshot, curve: object) -> float:
    """Reprice a bond against a replacement curve."""
    from dataclasses import replace as dataclass_replace

    from aegis.curves import DiscountCurve

    assert isinstance(curve, DiscountCurve)  # noqa: S101 - internal helper contract
    shifted = dataclass_replace(market, curves={**market.curves, bond.currency: curve})
    return bond.dirty_price(shifted)


#: A position is just an instrument; the alias exists because portfolios read
#: better when they hold positions rather than instruments.
Position = Instrument


def total_value(instruments: list[Instrument], market: MarketSnapshot) -> float:
    """Return the base-currency value of a list of positions.

    Args:
        instruments: The positions to value.
        market: The market state.

    Returns:
        The sum, converted into the snapshot's base currency.
    """
    return float(
        np.sum([i.present_value(market) * market.fx_rate(i.currency) for i in instruments])
    )
