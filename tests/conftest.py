"""Shared fixtures.

The archive under ``tests/fixtures/archive`` holds real payloads captured once
from FRED, the ECB and Cboe. Replaying them keeps the suite deterministic and
offline while still exercising the parsers against genuine market data, quirks
and all.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from aegis.marketdata import CboeProvider, EcbProvider, FredProvider, MarketStore

FIXTURE_ARCHIVE = Path(__file__).parent / "fixtures" / "archive"

#: The window the committed fixtures cover.
FIXTURE_START = date(2023, 1, 2)
FIXTURE_END = date(2024, 12, 31)


@pytest.fixture(scope="module")
def fred() -> FredProvider:
    """A FRED adapter replaying the committed fixtures."""
    provider = FredProvider.replaying(FIXTURE_ARCHIVE)
    assert isinstance(provider, FredProvider)
    return provider


@pytest.fixture(scope="module")
def ecb() -> EcbProvider:
    """An ECB adapter replaying the committed fixtures."""
    provider = EcbProvider.replaying(FIXTURE_ARCHIVE)
    assert isinstance(provider, EcbProvider)
    return provider


@pytest.fixture(scope="module")
def cboe() -> CboeProvider:
    """A Cboe adapter replaying the committed fixtures."""
    provider = CboeProvider.replaying(FIXTURE_ARCHIVE)
    assert isinstance(provider, CboeProvider)
    return provider


@pytest.fixture
def store() -> Iterator[MarketStore]:
    """An empty in-memory market store."""
    with MarketStore(":memory:") as market_store:
        yield market_store


def number(value: object) -> float:
    """Narrow a Polars scalar to a float for assertions.

    Polars types every scalar accessor as a wide union covering dates, strings
    and lists, which turns ordinary numeric assertions into type errors. This
    asserts what the test already knows and keeps the suite under strict mypy.

    Args:
        value: A scalar pulled out of a frame or series.

    Returns:
        The value as a float.
    """
    assert isinstance(value, (int, float))
    return float(value)


def day(value: object) -> date:
    """Narrow a Polars scalar to a date for assertions.

    Args:
        value: A scalar pulled out of a frame or series.

    Returns:
        The value as a date.
    """
    assert isinstance(value, date)
    return value


@pytest.fixture(scope="module")
def risk_store(fred: FredProvider, ecb: EcbProvider, cboe: CboeProvider) -> Iterator[MarketStore]:
    """A store loaded with two years of real curve, index and FX history."""
    with MarketStore(":memory:") as store:
        end = date(2026, 8, 24)
        start = date(2024, 8, 1)
        store.append("curve_point", fred.treasury_curve(start, end), source="fred")
        store.append("price_eod", fred.series(["SP500", "VIXCLS"], start, end), source="fred")
        store.append("fx_rate", ecb.fx_rates("USD", start, end), source="ecb")
        store.append("price_eod", cboe.underlying_quote("KO"), source="cboe")
        yield store
