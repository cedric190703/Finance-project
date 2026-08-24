"""The bitemporal market store.

The interesting tests here are the ones about *knowledge* time: a restatement
must not overwrite what we believed earlier, and a point-in-time read must
return the earlier belief.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from aegis.marketdata import CboeProvider, EcbProvider, FredProvider, MarketStore, ingest_treasury
from tests.conftest import FIXTURE_END, FIXTURE_START


def _prices(close: float, knowledge_date: date, symbol: str = "SP500") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "value_date": [date(2024, 3, 14)],
            "close": [close],
            "adj_close": [close],
            "knowledge_date": [knowledge_date],
        }
    )


def test_appending_is_idempotent_within_a_knowledge_date(store: MarketStore) -> None:
    frame = _prices(5200.0, date(2024, 3, 14))

    assert store.append("price_eod", frame, source="fred") == 1
    assert store.append("price_eod", frame, source="fred") == 0
    assert store.as_of("price_eod").height == 1


def test_a_restatement_is_kept_alongside_the_original(store: MarketStore) -> None:
    store.append("price_eod", _prices(5200.0, date(2024, 3, 14)), source="fred")
    store.append("price_eod", _prices(5175.5, date(2024, 3, 20)), source="fred")

    assert store.sql("SELECT count(*) AS n FROM price_eod")["n"][0] == 2
    assert store.as_of("price_eod").height == 1


def test_the_latest_belief_wins_by_default(store: MarketStore) -> None:
    store.append("price_eod", _prices(5200.0, date(2024, 3, 14)), source="fred")
    store.append("price_eod", _prices(5175.5, date(2024, 3, 20)), source="fred")

    assert store.as_of("price_eod")["close"][0] == pytest.approx(5175.5)


def test_a_point_in_time_read_returns_what_was_known_then(store: MarketStore) -> None:
    store.append("price_eod", _prices(5200.0, date(2024, 3, 14)), source="fred")
    store.append("price_eod", _prices(5175.5, date(2024, 3, 20)), source="fred")

    # This is the whole point of the design: a rebuild dated 15 March must not
    # see a correction that only arrived on the 20th.
    as_known_then = store.as_of("price_eod", knowledge_date=date(2024, 3, 15))
    assert as_known_then["close"][0] == pytest.approx(5200.0)


def test_a_point_in_time_read_before_anything_was_known_is_empty(store: MarketStore) -> None:
    store.append("price_eod", _prices(5200.0, date(2024, 3, 14)), source="fred")
    assert store.as_of("price_eod", knowledge_date=date(2024, 3, 1)).is_empty()


def test_revisions_surface_only_values_that_actually_changed(store: MarketStore) -> None:
    store.append("price_eod", _prices(5200.0, date(2024, 3, 14)), source="fred")
    store.append("price_eod", _prices(5175.5, date(2024, 3, 20)), source="fred")
    store.append("price_eod", _prices(99.0, date(2024, 3, 14), symbol="KO"), source="cboe")
    store.append("price_eod", _prices(99.0, date(2024, 3, 21), symbol="KO"), source="cboe")

    restated = store.revisions("price_eod")
    assert set(restated["symbol"].unique()) == {"SP500"}
    assert restated.height == 2


def test_value_date_filters_apply(store: MarketStore) -> None:
    for day in (date(2024, 1, 2), date(2024, 6, 3), date(2024, 12, 2)):
        store.append(
            "price_eod",
            pl.DataFrame(
                {
                    "symbol": ["SP500"],
                    "value_date": [day],
                    "close": [5000.0],
                    "knowledge_date": [date(2025, 1, 1)],
                }
            ),
            source="fred",
        )

    windowed = store.as_of("price_eod", start=date(2024, 2, 1), end=date(2024, 7, 1))
    assert windowed["value_date"].to_list() == [date(2024, 6, 3)]


def test_where_predicates_apply(store: MarketStore) -> None:
    store.append("price_eod", _prices(5200.0, date(2024, 3, 14)), source="fred")
    store.append("price_eod", _prices(99.0, date(2024, 3, 14), symbol="KO"), source="cboe")

    assert store.as_of("price_eod", where="symbol = 'KO'")["close"].to_list() == [99.0]


def test_appending_an_empty_frame_is_a_no_op(store: MarketStore) -> None:
    assert store.append("price_eod", pl.DataFrame(), source="fred") == 0


def test_rows_without_a_knowledge_date_are_rejected(store: MarketStore) -> None:
    frame = pl.DataFrame({"symbol": ["SP500"], "value_date": [date(2024, 1, 2)], "close": [1.0]})
    with pytest.raises(ValueError, match="must carry a knowledge_date"):
        store.append("price_eod", frame, source="fred")


@pytest.mark.parametrize("method", ["append", "as_of", "revisions"])
def test_unknown_tables_are_rejected(store: MarketStore, method: str) -> None:
    args = (
        (pl.DataFrame({"knowledge_date": [date(2024, 1, 1)]}), "fred") if method == "append" else ()
    )
    with pytest.raises(ValueError, match="unknown table"):
        getattr(store, method)("trades", *args)


def test_coverage_reports_every_table(store: MarketStore) -> None:
    store.append("price_eod", _prices(5200.0, date(2024, 3, 14)), source="fred")
    coverage = store.coverage()

    assert set(coverage["table_name"]) == {"curve_point", "fx_rate", "option_quote", "price_eod"}
    assert coverage.filter(pl.col("table_name") == "price_eod")["rows"][0] == 1
    assert coverage.filter(pl.col("table_name") == "fx_rate")["rows"][0] == 0


def test_a_full_ingest_round_trip_from_the_fixtures(
    store: MarketStore, fred: FredProvider, ecb: EcbProvider, cboe: CboeProvider
) -> None:
    curve = ingest_treasury(store, fred, FIXTURE_START, FIXTURE_END)
    assert curve.inserted == curve.fetched > 5000

    store.append("fx_rate", ecb.fx_rates("USD", FIXTURE_START, FIXTURE_END), source="ecb")
    store.append("option_quote", cboe.option_chain("KO"), source="cboe")
    store.append("price_eod", cboe.underlying_quote("KO"), source="cboe")

    coverage = store.coverage().sort("table_name")
    assert coverage["rows"].to_list() == [5500, 511, 1040, 1]

    # A curve read for one session is exactly the eleven quoted tenors.
    one_day = store.as_of("curve_point", start=FIXTURE_END, end=FIXTURE_END)
    assert one_day.height == 11
    assert one_day["source"].unique().to_list() == ["fred"]


def test_re_ingesting_the_same_day_inserts_nothing(store: MarketStore, fred: FredProvider) -> None:
    first = ingest_treasury(store, fred, FIXTURE_START, FIXTURE_END)
    second = ingest_treasury(store, fred, FIXTURE_START, FIXTURE_END)

    assert second.fetched == first.fetched
    assert second.inserted == 0
    assert "already known" in str(second)
