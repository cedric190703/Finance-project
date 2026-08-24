"""Explain a day's P&L into the moves a risk desk can act on.

The explain deliberately starts from yesterday's greeks.  Using end-of-day
sensitivities would make the components fit more neatly while hiding the risk
that was actually carried into the day.  Any gap is retained as an explicit
``unexplained`` residual: it is a control signal, not a bucket to smooth away.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import polars as pl

from aegis.instruments import EquityOption, EquityPosition, FixedRateBond, MarketSnapshot
from aegis.portfolio import Portfolio

__all__ = ["PnlExplain", "explain_pnl"]

_ONE_PERCENT = 0.01
_BASIS_POINT = 1e-4


@dataclass(frozen=True)
class PnlExplain:
    """A reconciled daily P&L explain.

    All components are in the portfolio base currency.  ``unexplained`` is
    calculated as the residual required for the components to reconcile to the
    actual change in portfolio value.
    """

    start_date: date
    end_date: date
    opening_value: float
    closing_value: float
    components: pl.DataFrame

    @property
    def total_pnl(self) -> float:
        """Return the actual mark-to-market P&L."""
        return self.closing_value - self.opening_value

    @property
    def explained_pnl(self) -> float:
        """Return P&L attributed to known drivers, excluding the residual."""
        return float(self.components.filter(pl.col("component") != "unexplained")["pnl"].sum())

    @property
    def unexplained_pnl(self) -> float:
        """Return the reconciliation residual."""
        return float(self.components.filter(pl.col("component") == "unexplained")["pnl"].sum())

    @property
    def reconciles(self) -> bool:
        """Return whether the components sum to the actual P&L."""
        return abs(float(self.components["pnl"].sum()) - self.total_pnl) < 1e-8


def explain_pnl(portfolio: Portfolio, start: MarketSnapshot, end: MarketSnapshot) -> PnlExplain:
    """Attribute a frozen book's P&L between two market snapshots.

    Args:
        portfolio: The positions held unchanged over the interval.
        start: Opening market snapshot and source of the greeks.
        end: Closing market snapshot.

    Returns:
        A reconciled explain containing delta, gamma, vega, theta, carry, rate,
        FX and unexplained components.

    Raises:
        ValueError: if the snapshots use different base currencies or do not
            advance in time.
    """
    if start.base_currency != end.base_currency:
        raise ValueError("P&L explain needs a common base currency")
    if end.value_date < start.value_date:
        raise ValueError("closing market must not precede opening market")

    elapsed_days = (end.value_date - start.value_date).days
    components = dict.fromkeys(("delta", "gamma", "vega", "theta", "carry", "rates", "fx"), 0.0)

    for position in portfolio.positions:
        fx_start = start.fx_rate(position.currency)
        sensitivities = position.sensitivities(start)

        if isinstance(position, (EquityPosition, EquityOption)):
            symbol = (
                position.symbol if isinstance(position, EquityPosition) else position.underlying
            )
            relative_move = end.spot(symbol) / start.spot(symbol) - 1.0
            move_in_percent = relative_move / _ONE_PERCENT
            components["delta"] += sensitivities.get("delta", 0.0) * move_in_percent * fx_start
            components["gamma"] += sensitivities.get("gamma", 0.0) * move_in_percent**2 * fx_start

        if isinstance(position, EquityOption):
            start_vol = start.implied_vol(position.underlying, position.strike, position.expiry)
            end_vol = end.implied_vol(position.underlying, position.strike, position.expiry)
            components["vega"] += (
                sensitivities.get("vega", 0.0) * (end_vol - start_vol) / _ONE_PERCENT * fx_start
            )
            components["theta"] += sensitivities.get("theta", 0.0) * elapsed_days * fx_start
            rate_move = _rate_move(start, end, position.currency, position.expiry)
            components["rates"] += (
                sensitivities.get("rho", 0.0) * rate_move / _BASIS_POINT * fx_start
            )

        if isinstance(position, FixedRateBond):
            rate_move = _rate_move(start, end, position.currency, position.maturity)
            # DV01 is positive for a bond that gains when yields fall.
            components["rates"] -= (
                sensitivities.get("dv01", 0.0) * rate_move / _BASIS_POINT * fx_start
            )

        if position.currency != start.base_currency:
            components["fx"] += position.present_value(start) * (
                end.fx_rate(position.currency) - fx_start
            )

    opening_value = portfolio.value(start)
    closing_value = portfolio.value(end)
    frozen_value = portfolio.value(_roll_market(start, end.value_date))
    components["carry"] = frozen_value - opening_value - components["theta"]
    components["unexplained"] = closing_value - opening_value - sum(components.values())

    order = ("delta", "gamma", "vega", "theta", "carry", "rates", "fx", "unexplained")
    frame = pl.DataFrame(
        {
            "component": list(order),
            "pnl": [components[name] for name in order],
            "cumulative_pnl": _cumulative([components[name] for name in order]),
        }
    )
    return PnlExplain(start.value_date, end.value_date, opening_value, closing_value, frame)


def _rate_move(start: MarketSnapshot, end: MarketSnapshot, currency: str, maturity: date) -> float:
    """Return the matched-maturity zero-rate move."""
    start_time = max(start.curve(currency).year_fraction(maturity), 0.0)
    end_time = max(end.curve(currency).year_fraction(maturity), 0.0)
    before = float(start.curve(currency).zero_rate(start_time)[0])
    after = float(end.curve(currency).zero_rate(end_time)[0])
    return after - before


def _roll_market(market: MarketSnapshot, value_date: date) -> MarketSnapshot:
    """Roll unchanged curves and surfaces forward to isolate calendar carry."""
    curves = {
        currency: replace(curve, reference_date=value_date)
        for currency, curve in market.curves.items()
    }
    surfaces = {
        symbol: replace(surface, reference_date=value_date)
        for symbol, surface in market.surfaces.items()
    }
    return replace(market, value_date=value_date, curves=curves, surfaces=surfaces)


def _cumulative(values: list[float]) -> list[float]:
    """Return running component totals for a waterfall chart."""
    total = 0.0
    output = []
    for value in values:
        total += value
        output.append(total)
    return output
