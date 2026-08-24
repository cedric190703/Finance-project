"""FRED: the US Treasury constant-maturity par yield curve.

The St. Louis Fed publishes the H.15 constant-maturity series as plain CSV with
no key required. These eleven series are the raw material the discount curve is
bootstrapped from in phase 3.
"""

from __future__ import annotations

from datetime import date
from io import StringIO
from typing import ClassVar

import polars as pl

from aegis.conventions import Tenor
from aegis.marketdata.providers.base import Provider

__all__ = ["INDEX_SERIES", "TREASURY_SERIES", "FredProvider"]

_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: FRED series id -> the tenor it quotes on the constant-maturity curve.
TREASURY_SERIES: dict[str, str] = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}


#: FRED series id -> the daily level series it publishes.
INDEX_SERIES: dict[str, str] = {
    "SP500": "S&P 500 index level",
    "NASDAQCOM": "NASDAQ Composite index level",
    "DJIA": "Dow Jones Industrial Average",
    "VIXCLS": "CBOE VIX index",
}


class FredProvider(Provider):
    """Adapter for the FRED CSV download endpoint."""

    name: ClassVar[str] = "fred"

    def treasury_curve(
        self,
        start: date,
        end: date,
        knowledge_date: date | None = None,
    ) -> pl.DataFrame:
        """Fetch the constant-maturity Treasury par curve over a date range.

        Args:
            start: First date requested, inclusive.
            end: Last date requested, inclusive.
            knowledge_date: Replay the archive as it stood on this date.

        Returns:
            Long-format rows of ``curve``, ``value_date``, ``tenor``,
            ``tenor_years``, ``rate`` (decimal, not percent) and
            ``knowledge_date``. Days the Treasury did not publish are dropped.
        """
        # The request deliberately carries no date window. fredgraph applies
        # cosd/coed only to single-series downloads and silently ignores them
        # once several ids are asked for, so the window is applied below instead.
        # Keeping it out of the request also keeps the archive key stable: one
        # captured payload serves every sub-window anybody later asks for.
        params = {"id": ",".join(TREASURY_SERIES)}
        payload = self._payload(
            "treasury", _CSV_URL, params, suffix="csv", knowledge_date=knowledge_date
        )
        raw = pl.read_csv(StringIO(payload.text()), null_values=["."], try_parse_dates=True)

        date_column = raw.columns[0]
        long = (
            raw.rename({date_column: "value_date"})
            .unpivot(index="value_date", variable_name="series", value_name="percent")
            .drop_nulls("percent")
            .with_columns(pl.col("percent").cast(pl.Float64))
        )
        return (
            long.filter(pl.col("value_date").is_between(start, end))
            .with_columns(
                pl.lit("UST").alias("curve"),
                pl.col("series").replace_strict(TREASURY_SERIES).alias("tenor"),
                (pl.col("percent") / 100.0).alias("rate"),
                pl.lit(payload.knowledge_date).alias("knowledge_date"),
            )
            .with_columns(
                pl.col("tenor")
                .map_elements(lambda t: Tenor.parse(t).approximate_years, return_dtype=pl.Float64)
                .alias("tenor_years")
            )
            .select("curve", "value_date", "tenor", "tenor_years", "rate", "knowledge_date")
            .sort(["value_date", "tenor_years"])
        )

    def series(
        self,
        series_ids: list[str],
        start: date,
        end: date,
        knowledge_date: date | None = None,
    ) -> pl.DataFrame:
        """Fetch daily level series and shape them like end-of-day prices.

        Index levels are not tradeable instruments, but they carry a long, clean
        and always-available history, which makes them the dependable backbone
        for the historical VaR window in phase 7.

        Args:
            series_ids: FRED series identifiers, e.g. ``["SP500", "VIXCLS"]``.
            start: First date requested, inclusive.
            end: Last date requested, inclusive.
            knowledge_date: Replay the archive as it stood on this date.

        Returns:
            Rows shaped like ``price_eod``: ``symbol``, ``value_date``,
            ``close``, ``adj_close`` and ``knowledge_date``. Non-publication
            days are dropped.
        """
        params = {"id": ",".join(series_ids)}  # windowed locally; see treasury_curve
        payload = self._payload(
            "series", _CSV_URL, params, suffix="csv", knowledge_date=knowledge_date
        )
        raw = pl.read_csv(StringIO(payload.text()), null_values=["."], try_parse_dates=True)
        date_column = raw.columns[0]
        return (
            raw.rename({date_column: "value_date"})
            .unpivot(index="value_date", variable_name="symbol", value_name="close")
            .drop_nulls("close")
            .filter(pl.col("value_date").is_between(start, end))
            .with_columns(
                pl.col("close").cast(pl.Float64),
                pl.col("close").cast(pl.Float64).alias("adj_close"),
                pl.lit(payload.knowledge_date).alias("knowledge_date"),
            )
            .select("symbol", "value_date", "close", "adj_close", "knowledge_date")
            .sort(["symbol", "value_date"])
        )
