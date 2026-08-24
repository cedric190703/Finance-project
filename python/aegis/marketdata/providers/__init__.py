"""Provider adapters for the free market-data sources the engine consumes."""

from aegis.marketdata.providers.base import Provider
from aegis.marketdata.providers.cboe import CboeProvider, parse_osi_symbol
from aegis.marketdata.providers.ecb import EcbProvider
from aegis.marketdata.providers.fred import INDEX_SERIES, TREASURY_SERIES, FredProvider
from aegis.marketdata.providers.yahoo import YahooProvider

__all__ = [
    "INDEX_SERIES",
    "TREASURY_SERIES",
    "CboeProvider",
    "EcbProvider",
    "FredProvider",
    "Provider",
    "YahooProvider",
    "parse_osi_symbol",
]
