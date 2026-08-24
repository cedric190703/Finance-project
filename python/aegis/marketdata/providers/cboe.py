"""Cboe delayed quotes: listed option chains straight from the exchange.

Cboe publishes a full delayed chain per underlying as JSON, with no key and no
throttling: every listed strike and expiry, two-sided quotes, open interest,
volume, the exchange's own implied volatility and greeks, and a snapshot of the
underlying alongside it.

That last point matters more than it looks. An implied volatility is only
meaningful next to the spot it was struck against; taking the option quotes from
one source and the spot from another, minutes apart, is how a vol surface ends up
with a phantom skew.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, ClassVar

import polars as pl

from aegis.marketdata.providers.base import Provider

__all__ = ["CboeProvider", "parse_osi_symbol"]

_QUOTES_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"

# OSI contract symbol: root, YYMMDD, C or P, then the strike in thousandths.
_OSI = re.compile(r"^(?P<root>[A-Z0-9]+?)(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


def parse_osi_symbol(symbol: str) -> tuple[str, date, str, float]:
    """Split an OSI option symbol into its components.

    ``KO260918C00090000`` is a KO call expiring 18 September 2026 struck at 90.

    Args:
        symbol: The 21-character OSI contract symbol.

    Returns:
        A tuple of root, expiry date, right (``"C"`` or ``"P"``) and strike.

    Raises:
        ValueError: if the symbol is not in OSI form.
    """
    match = _OSI.match(symbol.strip())
    if match is None:
        raise ValueError(f"not an OSI option symbol: {symbol!r}")
    raw_expiry = match.group("expiry")
    expiry = date(2000 + int(raw_expiry[:2]), int(raw_expiry[2:4]), int(raw_expiry[4:6]))
    return match.group("root"), expiry, match.group("right"), int(match.group("strike")) / 1000.0


class CboeProvider(Provider):
    """Adapter for the Cboe delayed-quote endpoint."""

    name: ClassVar[str] = "cboe"

    def option_chain(self, symbol: str, knowledge_date: date | None = None) -> pl.DataFrame:
        """Fetch the full delayed option chain for an underlying.

        Args:
            symbol: Underlying ticker, e.g. ``"KO"``.
            knowledge_date: Replay the archive as it stood on this date.

        Returns:
            One row per contract: ``underlying``, ``value_date``, ``expiry``,
            ``strike``, ``option_right``, ``bid``, ``ask``, ``last``,
            ``provider_iv``, ``volume``, ``open_interest``, ``spot`` and
            ``knowledge_date``. Contracts the exchange could not mark (zero
            implied volatility on both sides of a stale quote) are kept: it is
            the surface builder's job to decide what is tradeable, not the
            adapter's.

        Raises:
            ValueError: if the payload contains no chain.
        """
        payload = self._payload(
            "chain",
            _QUOTES_URL.format(symbol=symbol.upper()),
            {"symbol": symbol.upper()},
            knowledge_date=knowledge_date,
        )
        body: Any = payload.json()
        data = body.get("data") or {}
        contracts = data.get("options") or []
        if not contracts:
            raise ValueError(f"cboe returned no option chain for {symbol}")

        as_of = datetime.fromisoformat(body["timestamp"]).date()
        spot = float(data["current_price"])

        parsed = [parse_osi_symbol(str(c["option"])) for c in contracts]
        return pl.DataFrame(
            {
                "underlying": [symbol.upper()] * len(contracts),
                "value_date": [as_of] * len(contracts),
                "expiry": [p[1] for p in parsed],
                "strike": [p[3] for p in parsed],
                "option_right": [p[2] for p in parsed],
                "bid": [_f(c.get("bid")) for c in contracts],
                "ask": [_f(c.get("ask")) for c in contracts],
                "last": [_f(c.get("last_trade_price")) for c in contracts],
                "provider_iv": [_f(c.get("iv")) for c in contracts],
                "volume": [int(c.get("volume") or 0) for c in contracts],
                "open_interest": [int(c.get("open_interest") or 0) for c in contracts],
                "spot": [spot] * len(contracts),
                "knowledge_date": [payload.knowledge_date] * len(contracts),
            }
        ).sort(["expiry", "strike", "option_right"])

    def underlying_quote(self, symbol: str, knowledge_date: date | None = None) -> pl.DataFrame:
        """Fetch the underlying snapshot published alongside the chain.

        Running this daily accumulates a genuine end-of-day series in the store,
        one knowledge date at a time.

        Args:
            symbol: Underlying ticker.
            knowledge_date: Replay the archive as it stood on this date.

        Returns:
            A single row shaped like the ``price_eod`` table.
        """
        payload = self._payload(
            "chain",
            _QUOTES_URL.format(symbol=symbol.upper()),
            {"symbol": symbol.upper()},
            knowledge_date=knowledge_date,
        )
        body: Any = payload.json()
        data = body["data"]
        return pl.DataFrame(
            {
                "symbol": [symbol.upper()],
                "value_date": [datetime.fromisoformat(body["timestamp"]).date()],
                "open": [_f(data.get("open"))],
                "high": [_f(data.get("high"))],
                "low": [_f(data.get("low"))],
                "close": [float(data["current_price"])],
                "adj_close": [float(data["current_price"])],
                "volume": [int(data.get("volume") or 0)],
                "knowledge_date": [payload.knowledge_date],
            }
        )


def _f(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
