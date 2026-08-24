"""The market snapshot: everything a valuation is allowed to look at.

A valuation takes exactly two inputs — the trade and this object — and nothing
else. No global state, no lazy fetch from a database halfway through pricing, no
"current" anything. That constraint is what makes the risk engine in phase 7
possible: a scenario is just a modified snapshot, and running a thousand of them
in parallel is safe because none of them can reach anything the others can see.

It is also what makes a valuation reproducible. Given the same snapshot, the
same trade prices to the same number in a year's time, which is the difference
between a P&L explain that can be defended and one that can only be asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date

from aegis.curves import DiscountCurve
from aegis.vol import VolSurface

__all__ = ["MarketSnapshot", "MissingMarketDataError"]


class MissingMarketDataError(KeyError):
    """Raised when a valuation asks the snapshot for something it does not hold.

    Deliberately loud. The alternative — defaulting a missing volatility to
    twenty percent, or a missing FX rate to one — turns a data outage into a
    plausible-looking number, which is far more dangerous than a failed run.
    """


@dataclass(frozen=True)
class MarketSnapshot:
    """Market state as of one moment, for one set of beliefs.

    Attributes:
        value_date: The date everything is observed as of.
        base_currency: Currency the portfolio reports in.
        spots: Spot price per symbol, in that symbol's own currency.
        curves: Discount curve per currency.
        surfaces: Volatility surface per underlying symbol.
        fx_rates: Units of the quote currency per unit of the base currency,
            keyed by the six-letter pair (``"EURUSD"`` is dollars per euro).
        dividend_yields: Continuous dividend yield per symbol. Used only where
            no volatility surface is available to imply the forward from the
            option market directly.
    """

    value_date: date
    base_currency: str = "USD"
    spots: dict[str, float] = field(default_factory=dict)
    curves: dict[str, DiscountCurve] = field(default_factory=dict)
    surfaces: dict[str, VolSurface] = field(default_factory=dict)
    fx_rates: dict[str, float] = field(default_factory=dict)
    dividend_yields: dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- lookups

    def spot(self, symbol: str) -> float:
        """Return the spot price of a symbol.

        Args:
            symbol: The instrument's underlying.

        Returns:
            The spot price in the symbol's own currency.

        Raises:
            MissingMarketDataError: if the snapshot has no price for it.
        """
        try:
            return self.spots[symbol]
        except KeyError:
            raise MissingMarketDataError(f"no spot price for {symbol}") from None

    def curve(self, currency: str) -> DiscountCurve:
        """Return the discount curve for a currency.

        Args:
            currency: ISO currency code.

        Returns:
            The curve.

        Raises:
            MissingMarketDataError: if the snapshot has no curve for it.
        """
        try:
            return self.curves[currency]
        except KeyError:
            raise MissingMarketDataError(f"no discount curve for {currency}") from None

    def surface(self, symbol: str) -> VolSurface:
        """Return the volatility surface for an underlying.

        Args:
            symbol: The underlying symbol.

        Returns:
            The surface.

        Raises:
            MissingMarketDataError: if the snapshot has no surface for it.
        """
        try:
            return self.surfaces[symbol]
        except KeyError:
            raise MissingMarketDataError(f"no volatility surface for {symbol}") from None

    def fx_rate(self, currency: str) -> float:
        """Return the conversion factor from one currency into the base currency.

        Args:
            currency: The currency an amount is denominated in.

        Returns:
            How many units of the base currency one unit of ``currency`` buys.

        Raises:
            MissingMarketDataError: if neither the pair nor its inverse is quoted.
        """
        if currency == self.base_currency:
            return 1.0
        direct = f"{currency}{self.base_currency}"
        if direct in self.fx_rates:
            return self.fx_rates[direct]
        inverse = f"{self.base_currency}{currency}"
        if inverse in self.fx_rates:
            return 1.0 / self.fx_rates[inverse]
        raise MissingMarketDataError(f"no FX rate between {currency} and {self.base_currency}")

    def dividend_yield(self, symbol: str) -> float:
        """Return the continuous dividend yield assumed for a symbol.

        Args:
            symbol: The underlying symbol.

        Returns:
            The yield, defaulting to zero when none is configured.
        """
        return self.dividend_yields.get(symbol, 0.0)

    def forward(self, symbol: str, expiry: date, currency: str) -> float:
        """Return the forward price of a symbol to a date.

        Where a surface is available its own forwards are used, because those are
        the forwards the quoted volatilities were struck against. Recomputing
        from a spot and an assumed dividend yield would open a basis between the
        forward used to price and the forward used to imply, and that basis
        reappears later as unexplained P&L nobody can source.

        The surface's forward is not used directly, though: it is turned into a
        *carry ratio* against the spot the surface was calibrated at, and that
        ratio is applied to the current spot. The difference matters as soon as
        anything is bumped. A forward read straight off the surface does not move
        when spot moves, so every spot delta in the book would come out at zero —
        the risk engine would report a perfectly hedged portfolio and mean
        nothing by it. Holding the carry fixed and letting the forward follow
        spot is also what actually happens in the market: a two-percent move in
        the shares does not change the dividend or the repo rate.

        Args:
            symbol: The underlying symbol.
            expiry: The date to compute the forward to.
            currency: Currency whose curve carries the funding rate.

        Returns:
            The forward price.

        Raises:
            MissingMarketDataError: if neither a surface nor a spot price exists.
        """
        # A forward to a date that has already passed is just the spot: there is
        # no carry left to apply, and an expiring option settles against the
        # share price, not against a forward the surface still happens to quote.
        if expiry <= self.value_date:
            return self.spot(symbol)

        surface = self.surfaces.get(symbol)
        if surface is not None and surface.slices and surface.spot > 0.0:
            carry = _interpolate_forward(surface, expiry, self.value_date) / surface.spot
            return self.spot(symbol) * carry

        spot = self.spot(symbol)
        curve = self.curve(currency)
        time = curve.year_fraction(expiry)
        if time <= 0:
            return spot
        rate = float(curve.zero_rate(time)[0])
        return spot * math.exp((rate - self.dividend_yield(symbol)) * time)

    def implied_vol(self, symbol: str, strike: float, expiry: date) -> float:
        """Return the implied volatility for a strike and expiry.

        Args:
            symbol: The underlying symbol.
            strike: Strike price.
            expiry: Expiry date.

        Returns:
            The volatility from the calibrated surface.

        Raises:
            MissingMarketDataError: if the snapshot has no surface for the underlying.
        """
        return self.surface(symbol).implied_vol(strike, expiry)

    # ------------------------------------------------------------- scenarios

    def with_spots(self, **shifts: float) -> MarketSnapshot:
        """Return a copy with spot prices multiplied by the given factors.

        Args:
            **shifts: Symbol to relative shift, e.g. ``AAPL=0.99`` for a 1% fall.

        Returns:
            A new snapshot; the original is untouched.
        """
        moved = dict(self.spots)
        for symbol, factor in shifts.items():
            moved[symbol] = self.spot(symbol) * factor
        return replace(self, spots=moved)

    def with_curves_shifted(self, basis_points: float) -> MarketSnapshot:
        """Return a copy with every curve shifted in parallel.

        Args:
            basis_points: Size of the shift.

        Returns:
            A new snapshot.
        """
        return replace(
            self,
            curves={ccy: curve.shift_parallel(basis_points) for ccy, curve in self.curves.items()},
        )

    def with_value_date(self, value_date: date) -> MarketSnapshot:
        """Return a copy dated to a different day, for theta and carry.

        Only the observation date moves; the market data does not. That is the
        point: it isolates the passage of time from everything else, which is
        exactly what a P&L attribution needs.

        Args:
            value_date: The new observation date.

        Returns:
            A new snapshot.
        """
        return replace(self, value_date=value_date)


def _interpolate_forward(surface: VolSurface, expiry: date, value_date: date) -> float:
    """Interpolate the surface's implied forwards, linearly in time."""
    times = [(s.expiry - value_date).days for s in surface.slices]
    target = (expiry - value_date).days
    if target <= times[0]:
        return surface.slices[0].forward
    if target >= times[-1]:
        return surface.slices[-1].forward
    for left, right, left_time, right_time in zip(
        surface.slices, surface.slices[1:], times, times[1:], strict=False
    ):
        if left_time <= target <= right_time:
            weight = (target - left_time) / (right_time - left_time)
            return left.forward * (1.0 - weight) + right.forward * weight
    return surface.slices[-1].forward  # pragma: no cover - covered by the bounds above
