"""Yahoo Finance: daily equity bars and option chains.

Yahoo is free, unauthenticated and rate limited, and its adjusted close is
restated whenever a split or dividend lands. That restatement is precisely why
the store downstream is bitemporal.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, ClassVar

import polars as pl

from aegis.marketdata.http import BROWSER_USER_AGENT
from aegis.marketdata.providers.base import Provider

__all__ = ["YahooProvider"]

_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
_OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{symbol}"


def _epoch(day: date) -> str:
    return str(int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()))


class YahooProvider(Provider):
    """Adapter for the Yahoo Finance chart and options endpoints."""

    name: ClassVar[str] = "yahoo"
    #: Yahoo returns 429 to anything that does not claim to be a browser.
    user_agent: ClassVar[str | None] = BROWSER_USER_AGENT

    def daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        knowledge_date: date | None = None,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV bars for one symbol.

        Args:
            symbol: Yahoo ticker, e.g. ``"AAPL"``.
            start: First date requested, inclusive.
            end: Last date requested, inclusive.
            knowledge_date: Replay the archive as it stood on this date.

        Returns:
            One row per session with columns ``symbol``, ``value_date``,
            ``open``, ``high``, ``low``, ``close``, ``adj_close``, ``volume``
            and ``knowledge_date``.

        Raises:
            ValueError: if the payload carries no price series for the symbol.
        """
        params = {
            "period1": _epoch(start),
            "period2": _epoch(end),
            "interval": "1d",
            "events": "div,split",
        }
        payload = self._payload(
            "chart", _CHART_URL.format(symbol=symbol), params, knowledge_date=knowledge_date
        )
        body: Any = payload.json()
        results = body.get("chart", {}).get("result") or []
        if not results:
            raise ValueError(f"yahoo returned no chart data for {symbol}")

        result = results[0]
        stamps = result.get("timestamp") or []
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(stamps))

        frame = pl.DataFrame(
            {
                "symbol": [symbol] * len(stamps),
                "value_date": [datetime.fromtimestamp(t, tz=UTC).date() for t in stamps],
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "adj_close": adj,
                "volume": quote.get("volume", []),
            },
            schema_overrides={"volume": pl.Float64},
        )
        return (
            frame.drop_nulls("close")
            .with_columns(
                pl.col("volume").cast(pl.Int64),
                pl.lit(payload.knowledge_date).alias("knowledge_date"),
            )
            .sort("value_date")
        )

    def option_chain(
        self,
        symbol: str,
        expiry: date | None = None,
        knowledge_date: date | None = None,
    ) -> pl.DataFrame:
        """Fetch a listed option chain.

        Args:
            symbol: Underlying ticker.
            expiry: A specific expiry, or ``None`` for the nearest one.
            knowledge_date: Replay the archive as it stood on this date.

        Returns:
            One row per contract with strike, expiry, call/put, bid/ask/last,
            Yahoo's own implied volatility, volume and open interest, plus the
            spot price observed alongside the chain.

        Raises:
            ValueError: if the payload carries no chain for the symbol.
        """
        params = {"date": _epoch(expiry)} if expiry else {}
        payload = self._payload(
            "options", _OPTIONS_URL.format(symbol=symbol), params, knowledge_date=knowledge_date
        )
        body: Any = payload.json()
        results = body.get("optionChain", {}).get("result") or []
        if not results:
            raise ValueError(f"yahoo returned no option chain for {symbol}")

        result = results[0]
        spot = result.get("quote", {}).get("regularMarketPrice")
        rows: list[dict[str, object]] = []
        for chain in result.get("options", []):
            for right, contracts in (("C", chain.get("calls", [])), ("P", chain.get("puts", []))):
                for contract in contracts:
                    rows.append(
                        {
                            "underlying": symbol,
                            "option_right": right,
                            "expiry": datetime.fromtimestamp(contract["expiration"], tz=UTC).date(),
                            "strike": float(contract["strike"]),
                            "bid": _as_float(contract.get("bid")),
                            "ask": _as_float(contract.get("ask")),
                            "last": _as_float(contract.get("lastPrice")),
                            "provider_iv": _as_float(contract.get("impliedVolatility")),
                            "volume": int(contract.get("volume") or 0),
                            "open_interest": int(contract.get("openInterest") or 0),
                        }
                    )

        return pl.DataFrame(rows).with_columns(
            pl.lit(_as_float(spot)).alias("spot"),
            pl.lit(payload.knowledge_date).alias("value_date"),
            pl.lit(payload.knowledge_date).alias("knowledge_date"),
        )

    def expiries(self, symbol: str, knowledge_date: date | None = None) -> list[date]:
        """List the expiries Yahoo publishes for an underlying.

        Args:
            symbol: Underlying ticker.
            knowledge_date: Replay the archive as it stood on this date.

        Returns:
            The listed expiry dates, ascending.
        """
        payload = self._payload(
            "options", _OPTIONS_URL.format(symbol=symbol), {}, knowledge_date=knowledge_date
        )
        body: Any = payload.json()
        results = body.get("optionChain", {}).get("result") or []
        if not results:
            return []
        return sorted(
            datetime.fromtimestamp(t, tz=UTC).date() for t in results[0].get("expirationDates", [])
        )


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
