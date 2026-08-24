"""The book: a named collection of positions, and what it is worth.

A portfolio here is a value object. It holds positions and nothing else — no
cached prices, no market data, no last-valued timestamp — so the same book can be
valued against a hundred scenarios concurrently without any of them interfering.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import yaml

from aegis.conventions import DayCount, Frequency
from aegis.instruments import (
    Cash,
    EquityOption,
    EquityPosition,
    FixedRateBond,
    Instrument,
    MarketSnapshot,
)
from aegis.pricing import OptionRight

__all__ = ["Portfolio", "PortfolioError"]

#: Greeks that are ratios rather than cash amounts. Duration is years and
#: convexity is years squared, so adding one bond's duration to another's is
#: arithmetic without meaning — the portfolio number would have to be weighted by
#: value, and even then it only describes the bonds. They stay on the positions
#: that own them and are left out of the book-level aggregate.
_NON_ADDITIVE = frozenset({"duration", "convexity"})


class PortfolioError(ValueError):
    """Raised when a book cannot be built from its definition."""


@dataclass(frozen=True)
class Portfolio:
    """A named set of positions.

    Attributes:
        name: Identifier used in reports.
        positions: The positions, in declaration order.
    """

    name: str
    positions: tuple[Instrument, ...]

    def __len__(self) -> int:
        """Return the number of positions."""
        return len(self.positions)

    def value(self, market: MarketSnapshot) -> float:
        """Return the book's value in the snapshot's base currency.

        Args:
            market: The market state.

        Returns:
            The total value.
        """
        return sum(
            position.present_value(market) * market.fx_rate(position.currency)
            for position in self.positions
        )

    def valuations(self, market: MarketSnapshot) -> pl.DataFrame:
        """Return a per-position valuation breakdown.

        Args:
            market: The market state.

        Returns:
            One row per position, with local and base-currency values.
        """
        rows = []
        for position in self.positions:
            local = position.present_value(market)
            rate = market.fx_rate(position.currency)
            rows.append(
                {
                    "id": position.id,
                    "type": type(position).__name__,
                    "currency": position.currency,
                    "local_value": local,
                    "fx_rate": rate,
                    "base_value": local * rate,
                }
            )
        return pl.DataFrame(rows)

    def sensitivities(self, market: MarketSnapshot) -> dict[str, float]:
        """Return the book's aggregated greeks, in base currency.

        Aggregating greeks across positions is only meaningful because they are
        all expressed in cash terms for a common shift size — a 1% move, a
        volatility point, one basis point. Adding dimensionless deltas across
        different underlyings would produce a number with no interpretation, and
        the ratio measures are excluded for the same reason.

        Args:
            market: The market state.

        Returns:
            A mapping of greek name to aggregated value, in base currency.
        """
        totals: dict[str, float] = {}
        for position in self.positions:
            rate = market.fx_rate(position.currency)
            for name, value in position.sensitivities(market).items():
                if name in _NON_ADDITIVE:
                    continue
                totals[name] = totals.get(name, 0.0) + value * rate
        return totals

    def risk_factors(self) -> tuple[str, ...]:
        """Return every market factor the book is exposed to, deduplicated."""
        seen: dict[str, None] = {}
        for position in self.positions:
            for factor in position.risk_factors():
                seen[factor] = None
            if position.currency:
                seen[f"FX:{position.currency}"] = None
        return tuple(seen)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Portfolio:
        """Load a book from a YAML definition.

        Args:
            path: Path to the definition file.

        Returns:
            The loaded portfolio.

        Raises:
            PortfolioError: if the file is malformed or names an unknown type.
        """
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict) or "positions" not in raw:
            raise PortfolioError(f"{path}: expected a mapping with a 'positions' key")
        positions = tuple(_build(entry, index) for index, entry in enumerate(raw["positions"]))
        return cls(name=str(raw.get("name", Path(path).stem)), positions=positions)


def _build(entry: dict[str, object], index: int) -> Instrument:
    """Turn one YAML entry into an instrument."""
    if not isinstance(entry, dict) or "type" not in entry:
        raise PortfolioError(f"position {index}: every entry needs a 'type'")

    kind = str(entry["type"]).lower()
    identifier = str(entry.get("id", f"POS-{index:03d}"))
    currency = str(entry.get("currency", "USD"))

    try:
        match kind:
            case "cash":
                return Cash(id=identifier, currency=currency, amount=float(entry["amount"]))  # type: ignore[arg-type]
            case "equity":
                return EquityPosition(
                    id=identifier,
                    currency=currency,
                    symbol=str(entry["symbol"]),
                    quantity=float(entry["quantity"]),  # type: ignore[arg-type]
                )
            case "option":
                return EquityOption(
                    id=identifier,
                    currency=currency,
                    underlying=str(entry["underlying"]),
                    strike=float(entry["strike"]),  # type: ignore[arg-type]
                    expiry=_as_date(entry["expiry"]),
                    right=OptionRight(str(entry.get("right", "C")).upper()[0]),
                    quantity=float(entry["quantity"]),  # type: ignore[arg-type]
                    contract_size=float(entry.get("contract_size", 100.0)),  # type: ignore[arg-type]
                )
            case "bond":
                return FixedRateBond(
                    id=identifier,
                    currency=currency,
                    face=float(entry["face"]),  # type: ignore[arg-type]
                    coupon=float(entry["coupon"]),  # type: ignore[arg-type]
                    maturity=_as_date(entry["maturity"]),
                    issue_date=_as_date(entry["issue_date"]),
                    frequency=Frequency[str(entry.get("frequency", "SEMI_ANNUAL")).upper()],
                    day_count=DayCount(str(entry.get("day_count", DayCount.ACT_ACT_ISDA.value))),
                )
            case other:
                raise PortfolioError(f"position {index}: unknown instrument type {other!r}")
    except KeyError as missing:
        raise PortfolioError(f"position {index} ({kind}): missing field {missing}") from None


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
