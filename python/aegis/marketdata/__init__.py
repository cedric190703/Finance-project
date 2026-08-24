"""Market data: provider adapters, the raw archive, and the bitemporal store."""

from aegis.marketdata.archive import ArchivedResponse, RawArchive, request_key
from aegis.marketdata.http import (
    BROWSER_USER_AGENT,
    DEFAULT_USER_AGENT,
    FetchError,
    HttpFetcher,
    RetryPolicy,
)
from aegis.marketdata.ingest import (
    IngestResult,
    ingest_fx,
    ingest_option_chain,
    ingest_prices,
    ingest_treasury,
)
from aegis.marketdata.providers import (
    CboeProvider,
    EcbProvider,
    FredProvider,
    Provider,
    YahooProvider,
)
from aegis.marketdata.store import MarketStore

__all__ = [
    "BROWSER_USER_AGENT",
    "DEFAULT_USER_AGENT",
    "ArchivedResponse",
    "CboeProvider",
    "EcbProvider",
    "FetchError",
    "FredProvider",
    "HttpFetcher",
    "IngestResult",
    "MarketStore",
    "Provider",
    "RawArchive",
    "RetryPolicy",
    "YahooProvider",
    "ingest_fx",
    "ingest_option_chain",
    "ingest_prices",
    "ingest_treasury",
    "request_key",
]
