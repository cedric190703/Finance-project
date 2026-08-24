"""P&L explain: explain what changed and retain the residual."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from aegis.curves import DiscountCurve
from aegis.instruments import EquityPosition, MarketSnapshot
from aegis.pnl import explain_pnl
from aegis.portfolio import Portfolio


@pytest.fixture
def opening_market() -> MarketSnapshot:
    day = date(2026, 8, 20)
    return MarketSnapshot(
        value_date=day,
        spots={"USD_STOCK": 100.0, "EUR_STOCK": 50.0},
        curves={"USD": DiscountCurve.flat(day, 0.03), "EUR": DiscountCurve.flat(day, 0.02)},
        fx_rates={"EURUSD": 1.10},
    )


@pytest.fixture
def book() -> Portfolio:
    return Portfolio(
        "pnl-test",
        (
            EquityPosition(id="USD", currency="USD", symbol="USD_STOCK", quantity=100),
            EquityPosition(id="EUR", currency="EUR", symbol="EUR_STOCK", quantity=100),
        ),
    )


def test_linear_spot_and_fx_pnl_is_fully_explained(
    book: Portfolio, opening_market: MarketSnapshot
) -> None:
    closing_market = replace(
        opening_market,
        value_date=opening_market.value_date + timedelta(days=1),
        spots={"USD_STOCK": 105.0, "EUR_STOCK": 55.0},
        fx_rates={"EURUSD": 1.20},
    )
    result = explain_pnl(book, opening_market, closing_market)
    rows = dict(result.components.select("component", "pnl").iter_rows())
    assert result.total_pnl == pytest.approx(1_600.0)
    assert rows["delta"] == pytest.approx(1_050.0)
    assert rows["fx"] == pytest.approx(500.0)
    assert rows["unexplained"] == pytest.approx(50.0)
    assert result.reconciles


def test_explain_is_an_ordered_waterfall(book: Portfolio, opening_market: MarketSnapshot) -> None:
    closing_market = replace(
        opening_market,
        value_date=opening_market.value_date + timedelta(days=1),
        spots={"USD_STOCK": 99.0, "EUR_STOCK": 49.0},
        fx_rates={"EURUSD": 1.09},
    )
    result = explain_pnl(book, opening_market, closing_market)
    assert result.components["component"].to_list()[-1] == "unexplained"
    assert result.components["cumulative_pnl"][-1] == pytest.approx(result.total_pnl)


def test_explain_rejects_incompatible_market_snapshots(
    book: Portfolio, opening_market: MarketSnapshot
) -> None:
    different_currency = replace(opening_market, base_currency="EUR")
    with pytest.raises(ValueError, match="common base currency"):
        explain_pnl(book, opening_market, different_currency)

    earlier = replace(opening_market, value_date=opening_market.value_date - timedelta(days=1))
    with pytest.raises(ValueError, match="must not precede"):
        explain_pnl(book, opening_market, earlier)
