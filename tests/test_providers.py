"""Provider adapters, replayed against real captured payloads."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from aegis.marketdata import CboeProvider, EcbProvider, FetchError, FredProvider
from aegis.marketdata.providers import TREASURY_SERIES, parse_osi_symbol
from tests.conftest import FIXTURE_END, FIXTURE_START, day, number


def test_treasury_curve_covers_every_quoted_tenor(fred: FredProvider) -> None:
    curve = fred.treasury_curve(FIXTURE_START, FIXTURE_END)

    assert set(curve["tenor"].unique()) == set(TREASURY_SERIES.values())
    assert day(curve["value_date"].min()) >= FIXTURE_START
    assert day(curve["value_date"].max()) <= FIXTURE_END
    assert curve["knowledge_date"].n_unique() == 1


def test_treasury_yields_are_decimals_not_percent(fred: FredProvider) -> None:
    curve = fred.treasury_curve(FIXTURE_START, FIXTURE_END)
    # The 2023-24 window ran between roughly 3% and 6%; percent would be 100x this.
    assert 0.0 < number(curve["rate"].min()) < 0.10
    assert 0.0 < number(curve["rate"].max()) < 0.10


def test_treasury_curve_is_dated_and_ordered(fred: FredProvider) -> None:
    curve = fred.treasury_curve(date(2024, 12, 30), FIXTURE_END)
    one_day = curve.filter(pl.col("value_date") == FIXTURE_END)

    assert one_day.height == len(TREASURY_SERIES)
    assert one_day["tenor_years"].to_list() == sorted(one_day["tenor_years"].to_list())


def test_treasury_curve_honours_the_requested_window(fred: FredProvider) -> None:
    narrow = fred.treasury_curve(date(2024, 1, 2), date(2024, 1, 31))
    assert day(narrow["value_date"].min()) >= date(2024, 1, 2)
    assert day(narrow["value_date"].max()) <= date(2024, 1, 31)


def test_index_series_are_shaped_like_prices(fred: FredProvider) -> None:
    frame = fred.series(["SP500", "VIXCLS"], FIXTURE_START, FIXTURE_END)

    assert set(frame.columns) == {"symbol", "value_date", "close", "adj_close", "knowledge_date"}
    assert set(frame["symbol"].unique()) == {"SP500", "VIXCLS"}
    assert number(frame.filter(pl.col("symbol") == "SP500")["close"].min()) > 1000


def test_ecb_fixings_are_units_of_currency_per_euro(ecb: EcbProvider) -> None:
    frame = ecb.fx_rates("USD", FIXTURE_START, FIXTURE_END)

    assert frame["pair"].unique().to_list() == ["EURUSD"]
    # EURUSD traded between roughly 1.03 and 1.13 across 2023-24.
    assert 1.0 < number(frame["rate"].min()) < 1.2
    assert 1.0 < number(frame["rate"].max()) < 1.2


def test_ecb_series_is_sorted_and_gap_free_on_weekends(ecb: EcbProvider) -> None:
    frame = ecb.fx_rates("USD", FIXTURE_START, FIXTURE_END)

    assert frame["value_date"].to_list() == sorted(frame["value_date"].to_list())
    assert all(day.weekday() < 5 for day in frame["value_date"])


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("KO260918C00090000", ("KO", date(2026, 9, 18), "C", 90.0)),
        ("SPY241220P00500000", ("SPY", date(2024, 12, 20), "P", 500.0)),
        ("AAPL260824C00205000", ("AAPL", date(2026, 8, 24), "C", 205.0)),
    ],
)
def test_osi_symbols_parse(symbol: str, expected: tuple[str, date, str, float]) -> None:
    assert parse_osi_symbol(symbol) == expected


@pytest.mark.parametrize("symbol", ["", "KO", "KO2609C00090000", "KO260918X00090000"])
def test_osi_parsing_rejects_malformed_symbols(symbol: str) -> None:
    with pytest.raises(ValueError, match="not an OSI option symbol"):
        parse_osi_symbol(symbol)


def test_option_chain_is_a_two_sided_grid(cboe: CboeProvider) -> None:
    chain = cboe.option_chain("KO")

    assert chain.height > 100
    assert set(chain["option_right"].unique()) == {"C", "P"}
    assert chain["expiry"].n_unique() > 5
    assert (chain["expiry"] > chain["value_date"]).all()
    assert chain["spot"].n_unique() == 1


def test_option_chain_carries_the_spot_it_was_quoted_against(cboe: CboeProvider) -> None:
    chain = cboe.option_chain("KO")
    spot = number(chain["spot"][0])

    assert spot > 0
    # A liquid name is quoted around the money, not only in the wings.
    assert number(chain["strike"].min()) < spot < number(chain["strike"].max())


def test_underlying_quote_matches_the_chain_snapshot(cboe: CboeProvider) -> None:
    quote = cboe.underlying_quote("KO")
    chain = cboe.option_chain("KO")

    assert quote.height == 1
    assert quote["close"][0] == chain["spot"][0]
    assert number(quote["low"][0]) <= number(quote["close"][0]) <= number(quote["high"][0])


def test_replaying_provider_refuses_to_reach_the_network(fred: FredProvider) -> None:
    with pytest.raises(FetchError, match="no archived fred/series payload"):
        fred.series(["UNRATE"], FIXTURE_START, FIXTURE_END)


def test_a_window_outside_the_data_returns_nothing_rather_than_refetching(
    fred: FredProvider,
) -> None:
    # The payload holds the full published history, so an empty window is an
    # ordinary empty answer, not a cache miss.
    assert fred.treasury_curve(date(1900, 1, 1), date(1900, 12, 31)).is_empty()


def test_point_in_time_request_never_falls_back_to_a_live_fetch(
    cboe: CboeProvider,
) -> None:
    # Nothing was archived that early, and a point-in-time read must fail loudly
    # rather than quietly serve today's revised chain.
    with pytest.raises(FetchError, match="as known on"):
        cboe.option_chain("KO", knowledge_date=date(2020, 1, 1))
