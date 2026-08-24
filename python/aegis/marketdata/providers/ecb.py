"""ECB SDMX: euro foreign-exchange reference rates.

The ECB publishes one daily fixing per currency against the euro, quoted as
units of the foreign currency per euro. Free, keyless, and stable — the opposite
of scraping a broker page.
"""

from __future__ import annotations

from datetime import date
from io import StringIO
from typing import ClassVar

import polars as pl

from aegis.marketdata.providers.base import Provider

__all__ = ["EcbProvider"]

_SDMX_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.{currency}.EUR.SP00.A"


class EcbProvider(Provider):
    """Adapter for the ECB SDMX data API."""

    name: ClassVar[str] = "ecb"

    def fx_rates(
        self,
        currency: str,
        start: date,
        end: date,
        knowledge_date: date | None = None,
    ) -> pl.DataFrame:
        """Fetch the daily EUR reference fixing for one currency.

        Args:
            currency: ISO code of the foreign currency, e.g. ``"USD"``.
            start: First date requested, inclusive.
            end: Last date requested, inclusive.
            knowledge_date: Replay the archive as it stood on this date.

        Returns:
            Rows of ``pair`` (e.g. ``EURUSD``), ``value_date``, ``rate`` — the
            units of ``currency`` per euro — and ``knowledge_date``.
        """
        # As with FRED, the full published series is fetched once and windowed
        # locally, so one archived payload answers every later date range.
        params = {"format": "csvdata"}
        payload = self._payload(
            "exr",
            _SDMX_URL.format(currency=currency.upper()),
            params,
            suffix="csv",
            knowledge_date=knowledge_date,
        )
        raw = pl.read_csv(StringIO(payload.text()), try_parse_dates=True)
        return (
            raw.select(
                pl.lit(f"EUR{currency.upper()}").alias("pair"),
                pl.col("TIME_PERIOD").cast(pl.Date).alias("value_date"),
                pl.col("OBS_VALUE").cast(pl.Float64).alias("rate"),
                pl.lit(payload.knowledge_date).alias("knowledge_date"),
            )
            .drop_nulls("rate")
            .filter(pl.col("value_date").is_between(start, end))
            .sort("value_date")
        )
