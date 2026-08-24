"""Ingestion: pull from providers into the bitemporal store.

Each function is a single logical dataset. They are deliberately thin — fetch,
hand the frame to the store, report what landed — so that phase 10 can wrap them
in a task graph without any of this having to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from aegis.marketdata.providers import (
    CboeProvider,
    EcbProvider,
    FredProvider,
    YahooProvider,
)
from aegis.marketdata.store import MarketStore

__all__ = ["IngestResult", "ingest_fx", "ingest_option_chain", "ingest_prices", "ingest_treasury"]


@dataclass(frozen=True)
class IngestResult:
    """What one ingest call produced.

    Attributes:
        dataset: Logical dataset name.
        fetched: Rows returned by the provider.
        inserted: Rows that were new to the store; the rest were already known.
    """

    dataset: str
    fetched: int
    inserted: int

    def __str__(self) -> str:
        """Return a one-line human summary."""
        skipped = self.fetched - self.inserted
        return f"{self.dataset}: {self.inserted} new, {skipped} already known"


def ingest_prices(
    store: MarketStore,
    provider: YahooProvider,
    symbols: list[str],
    start: date,
    end: date,
    knowledge_date: date | None = None,
) -> list[IngestResult]:
    """Ingest daily equity bars for a list of symbols.

    Args:
        store: Destination store.
        provider: Yahoo adapter.
        symbols: Tickers to fetch.
        start: First date requested, inclusive.
        end: Last date requested, inclusive.
        knowledge_date: Replay the archive as it stood on this date.

    Returns:
        One result per symbol.
    """
    results = []
    for symbol in symbols:
        frame = provider.daily_bars(symbol, start, end, knowledge_date=knowledge_date)
        inserted = store.append("price_eod", frame, source=provider.name)
        results.append(IngestResult(f"price_eod/{symbol}", frame.height, inserted))
    return results


def ingest_treasury(
    store: MarketStore,
    provider: FredProvider,
    start: date,
    end: date,
    knowledge_date: date | None = None,
) -> IngestResult:
    """Ingest the US Treasury constant-maturity par curve.

    Args:
        store: Destination store.
        provider: FRED adapter.
        start: First date requested, inclusive.
        end: Last date requested, inclusive.
        knowledge_date: Replay the archive as it stood on this date.

    Returns:
        The ingest result for the curve.
    """
    frame = provider.treasury_curve(start, end, knowledge_date=knowledge_date)
    inserted = store.append("curve_point", frame, source=provider.name)
    return IngestResult("curve_point/UST", frame.height, inserted)


def ingest_fx(
    store: MarketStore,
    provider: EcbProvider,
    currencies: list[str],
    start: date,
    end: date,
    knowledge_date: date | None = None,
) -> list[IngestResult]:
    """Ingest ECB euro reference fixings.

    Args:
        store: Destination store.
        provider: ECB adapter.
        currencies: ISO codes quoted against the euro.
        start: First date requested, inclusive.
        end: Last date requested, inclusive.
        knowledge_date: Replay the archive as it stood on this date.

    Returns:
        One result per currency.
    """
    results = []
    for currency in currencies:
        frame = provider.fx_rates(currency, start, end, knowledge_date=knowledge_date)
        inserted = store.append("fx_rate", frame, source=provider.name)
        results.append(IngestResult(f"fx_rate/EUR{currency.upper()}", frame.height, inserted))
    return results


def ingest_option_chain(
    store: MarketStore,
    provider: CboeProvider,
    symbol: str,
    knowledge_date: date | None = None,
) -> list[IngestResult]:
    """Ingest a delayed option chain together with its underlying snapshot.

    The two land in the same call on purpose: an implied volatility is only
    interpretable next to the spot it was struck against, and taking them from
    separate requests minutes apart invents skew that was never quoted.

    Args:
        store: Destination store.
        provider: Cboe adapter.
        symbol: Underlying ticker.
        knowledge_date: Replay the archive as it stood on this date.

    Returns:
        One result for the chain and one for the underlying quote.
    """
    chain = provider.option_chain(symbol, knowledge_date=knowledge_date)
    quote = provider.underlying_quote(symbol, knowledge_date=knowledge_date)
    return [
        IngestResult(
            f"option_quote/{symbol.upper()}",
            chain.height,
            store.append("option_quote", chain, source=provider.name),
        ),
        IngestResult(
            f"price_eod/{symbol.upper()}",
            quote.height,
            store.append("price_eod", quote, source=provider.name),
        ),
    ]
