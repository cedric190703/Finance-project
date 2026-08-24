"""Assembling the daily risk report.

The pieces exist elsewhere; this is the part that puts a market together from
the store, values the book against it, and produces the numbers a desk actually
reads in the morning. It is deliberately the only place that knows how the
demo book's factors map onto the free data sources, so that mapping is
configuration rather than something buried in the risk maths.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from aegis.curves import DiscountCurve, curve_from_store
from aegis.instruments import MarketSnapshot
from aegis.marketdata import MarketStore
from aegis.portfolio import Portfolio
from aegis.risk.factors import (
    ABSOLUTE,
    RELATIVE,
    FactorHistory,
    FactorMapping,
    build_factor_history,
)
from aegis.risk.scenarios import Scenario, load_scenarios
from aegis.risk.var import (
    VarResult,
    contribution_report,
    historical_var,
    monte_carlo_var,
    parametric_var,
)
from aegis.vol import build_surface

__all__ = ["RiskReport", "build_market", "default_factor_mappings", "run_report"]

#: How far back a curve is looked for when the exact session has no publication.
_CURVE_LOOKBACK_DAYS = 30


def default_factor_mappings() -> list[FactorMapping]:
    """Return the factor mapping used by the demo book.

    Two proxies are declared explicitly rather than quietly assumed. Neither is
    ideal; both are what a desk does when a name has no usable history, and the
    point is that the report says so on its face.

    Returns:
        The mappings, in report order.
    """
    return [
        FactorMapping("SPOT:SP500", "price_eod", "symbol = 'SP500'", "close", RELATIVE),
        FactorMapping(
            "SPOT:KO",
            "price_eod",
            "symbol = 'SP500'",
            "close",
            RELATIVE,
            note="proxied by the index: the free sources carry no daily single-name history",
        ),
        FactorMapping(
            "VOL:KO",
            "price_eod",
            "symbol = 'VIXCLS'",
            "close",
            RELATIVE,
            note="single-name implied volatility proxied by the VIX",
        ),
        FactorMapping("RATE:USD:2Y", "curve_point", "tenor = '2Y'", "rate", ABSOLUTE),
        FactorMapping("RATE:USD:10Y", "curve_point", "tenor = '10Y'", "rate", ABSOLUTE),
        FactorMapping("RATE:USD:30Y", "curve_point", "tenor = '30Y'", "rate", ABSOLUTE),
        FactorMapping("FX:EUR", "fx_rate", "pair = 'EURUSD'", "rate", RELATIVE),
    ]


@dataclass(frozen=True)
class RiskReport:
    """Everything the morning report contains.

    Attributes:
        value_date: The session reported on.
        portfolio_value: Book value in base currency.
        valuations: Per-position breakdown.
        greeks: Aggregated book sensitivities.
        var_results: VaR and ES by method.
        contributions: Component VaR by factor.
        stress: Scenario name to P&L.
        factor_summary: Per-factor statistics, including any proxy notes.
    """

    value_date: date
    portfolio_value: float
    valuations: pl.DataFrame
    greeks: dict[str, float]
    var_results: tuple[VarResult, ...]
    contributions: pl.DataFrame
    stress: dict[str, float]
    factor_summary: pl.DataFrame


def build_market(
    store: MarketStore,
    value_date: date,
    underlying: str = "KO",
    eur_rate: float = 0.022,
    eurusd: float = 1.16,
    knowledge_date: date | None = None,
) -> MarketSnapshot:
    """Assemble a market snapshot for one session out of the store.

    Args:
        store: The bitemporal market store.
        value_date: The session to build.
        underlying: Symbol whose option chain supplies the volatility surface.
        eur_rate: Flat euro discount rate, standing in for a curve the free
            sources do not provide at daily frequency.
        eurusd: EURUSD spot, used when the store has no fixing for the session.
        knowledge_date: Rebuild using only what was known on this date.

    Returns:
        The snapshot.

    Raises:
        ValueError: if the store holds no curve near the session.
    """
    quotes = store.as_of(
        "curve_point",
        knowledge_date=knowledge_date,
        start=value_date - timedelta(days=_CURVE_LOOKBACK_DAYS),
        end=value_date,
    )
    if quotes.is_empty():
        raise ValueError(f"no curve quotes within 30 days of {value_date}")
    curve = curve_from_store(quotes, max(quotes["value_date"].to_list()))

    spots: dict[str, float] = {}
    prices = store.as_of("price_eod", knowledge_date=knowledge_date, end=value_date)
    for symbol in prices["symbol"].unique():
        series = prices.filter(pl.col("symbol") == symbol).sort("value_date")
        spots[str(symbol)] = float(series["close"][-1])

    surfaces = {}
    chain = store.as_of(
        "option_quote",
        knowledge_date=knowledge_date,
        end=value_date,
        where=f"underlying = '{underlying}'",
    )
    if not chain.is_empty():
        session = max(chain["value_date"].to_list())
        surface = build_surface(
            chain.filter(pl.col("value_date") == session), curve, reference_date=value_date
        )
        surfaces[underlying] = surface
        spots.setdefault(underlying, surface.spot)

    fx = store.as_of(
        "fx_rate", knowledge_date=knowledge_date, end=value_date, where="pair = 'EURUSD'"
    )
    rate = float(fx.sort("value_date")["rate"][-1]) if not fx.is_empty() else eurusd

    return MarketSnapshot(
        value_date=value_date,
        base_currency="USD",
        spots=spots,
        curves={"USD": curve, "EUR": DiscountCurve.flat(value_date, eur_rate, name="EUR")},
        surfaces=surfaces,
        fx_rates={"EURUSD": rate},
    )


def run_report(
    store: MarketStore,
    portfolio: Portfolio,
    value_date: date,
    lookback_days: int = 730,
    confidence: float = 0.99,
    scenarios: list[Scenario] | None = None,
    scenario_path: Path | str | None = None,
    knowledge_date: date | None = None,
) -> RiskReport:
    """Produce the full risk report for one session.

    Args:
        store: The market store.
        portfolio: The book.
        value_date: Session to report on.
        lookback_days: Length of the historical window for VaR.
        confidence: Confidence level for the VaR numbers.
        scenarios: Stress scenarios; loaded from ``scenario_path`` when omitted.
        scenario_path: Where to load scenarios from.
        knowledge_date: Rebuild using only what was known on this date.

    Returns:
        The assembled report.
    """
    market = build_market(store, value_date, knowledge_date=knowledge_date)
    history = build_factor_history(
        store,
        default_factor_mappings(),
        value_date - timedelta(days=lookback_days),
        value_date,
        knowledge_date=knowledge_date,
    )
    stress = _stress(portfolio, market, scenarios, scenario_path)

    return RiskReport(
        value_date=value_date,
        portfolio_value=portfolio.value(market),
        valuations=portfolio.valuations(market),
        greeks=portfolio.sensitivities(market),
        var_results=(
            historical_var(portfolio, market, history, confidence),
            parametric_var(portfolio, market, history, confidence),
            parametric_var(portfolio, market, history, confidence, cornish_fisher=True),
            monte_carlo_var(portfolio, market, history, confidence, scenarios=5_000, seed=1),
        ),
        contributions=contribution_report(portfolio, market, history),
        stress=stress,
        factor_summary=history.summary(),
    )


def _stress(
    portfolio: Portfolio,
    market: MarketSnapshot,
    scenarios: list[Scenario] | None,
    scenario_path: Path | str | None,
) -> dict[str, float]:
    if scenarios is None:
        if scenario_path is None:
            return {}
        scenarios = load_scenarios(scenario_path)
    base = portfolio.value(market)
    return {s.name: portfolio.value(s.apply(market)) - base for s in scenarios}


def factor_history_for(
    store: MarketStore, value_date: date, lookback_days: int = 730
) -> FactorHistory:
    """Return the standard factor history ending on a session.

    Args:
        store: The market store.
        value_date: Last observation date.
        lookback_days: Window length in calendar days.

    Returns:
        The factor history.
    """
    return build_factor_history(
        store,
        default_factor_mappings(),
        value_date - timedelta(days=lookback_days),
        value_date,
    )
