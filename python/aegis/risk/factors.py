"""Risk factors: what the book is exposed to, and how those things have moved.

A risk factor is a named market quantity the book's value depends on. Everything
downstream — historical VaR, the covariance matrix, stress scenarios, the P&L
explain — is expressed in terms of shocks to these:

``SPOT:KO``
    Log move in an equity price. Log rather than simple, because risk numbers get
    scaled: a ten-day horizon multiplies a daily move by √10, and a −35% daily
    move on a volatility index scaled that way becomes −110% — a negative
    volatility, which is not a stress scenario but a crash. In log space the same
    arithmetic always lands somewhere positive, and log moves add across days
    instead of compounding, which is what the √t rule assumes in the first place.
``RATE:USD:10Y``
    Absolute move, in decimal, of a point on a currency's curve. Rates are
    shocked additively because that is how they move and how they are quoted; a
    "10% rise" in a rate near zero means nothing.
``VOL:KO``
    Relative move in implied volatility, applied across the whole surface.
``FX:EUR``
    Relative move in the value of a currency against the base currency.

Two honest approximations are recorded here rather than buried:

* A curve is shocked by shifting the *zero* curve by the move observed in the
  *par* yield at that tenor. For the small daily moves that drive VaR the two
  are within a fraction of a basis point of each other, and the alternative — a
  full re-bootstrap inside every one of five hundred scenarios — costs far more
  than it corrects.
* Single-name implied volatility has no history in the free data, so it is
  proxied by the index volatility index. Desks do exactly this when a name has
  no option history; what matters is saying so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

from aegis.marketdata import MarketStore

__all__ = ["FactorHistory", "FactorMapping", "build_factor_history"]

FloatArray = npt.NDArray[np.float64]

#: How a factor's shock is interpreted.
RELATIVE = "relative"
ABSOLUTE = "absolute"


@dataclass(frozen=True)
class FactorMapping:
    """Where a factor's history comes from.

    Attributes:
        factor: The factor identifier, e.g. ``"SPOT:KO"``.
        table: Store table holding its history.
        where: SQL predicate selecting the right series.
        column: Column carrying the level.
        kind: ``"relative"`` for prices and volatilities, ``"absolute"`` for rates.
        note: Why this source was chosen, shown in the coverage report. Used to
            record proxies rather than let them pass unnoticed.
    """

    factor: str
    table: str
    where: str
    column: str
    kind: str = RELATIVE
    note: str = ""


@dataclass(frozen=True)
class FactorHistory:
    """Historical moves of a set of risk factors.

    Attributes:
        factors: Factor identifiers, in column order.
        dates: The observation date of each move.
        moves: One row per date, one column per factor. Relative factors carry
            log returns; absolute factors carry level changes.
        kinds: How each factor's move should be applied.
        notes: Any caveat attached to a factor's source.
    """

    factors: tuple[str, ...]
    dates: tuple[date, ...]
    moves: FloatArray
    kinds: dict[str, str]
    notes: dict[str, str]

    def __len__(self) -> int:
        """Return the number of historical observations."""
        return len(self.dates)

    def column(self, factor: str) -> FloatArray:
        """Return one factor's history.

        Args:
            factor: The factor identifier.

        Returns:
            Its column of moves.

        Raises:
            KeyError: if the factor is not in this history.
        """
        return self.moves[:, self.factors.index(factor)]

    def scenario(self, index: int) -> dict[str, float]:
        """Return one historical day as a shock dictionary.

        Args:
            index: Which observation to take.

        Returns:
            A mapping from factor to that day's move.
        """
        return dict(zip(self.factors, self.moves[index], strict=True))

    def covariance(self) -> FloatArray:
        """Return the sample covariance matrix of the factor moves.

        Returns:
            A square matrix in factor order.
        """
        covariance = np.cov(self.moves, rowvar=False, ddof=1)
        return np.asarray(covariance, dtype=np.float64).reshape(
            len(self.factors), len(self.factors)
        )

    def correlation(self) -> FloatArray:
        """Return the sample correlation matrix.

        Returns:
            A square matrix in factor order.
        """
        covariance = self.covariance()
        deviation = np.sqrt(np.diag(covariance))
        safe = np.where(deviation > 0, deviation, 1.0)
        return np.asarray(covariance / np.outer(safe, safe), dtype=np.float64)

    def summary(self) -> pl.DataFrame:
        """Return per-factor descriptive statistics.

        Returns:
            Volatility, worst and best move, skew and excess kurtosis. The last
            two are what justify not trusting a normal approximation.
        """
        rows = []
        for index, factor in enumerate(self.factors):
            column = self.moves[:, index]
            centred = column - column.mean()
            deviation = column.std(ddof=1)
            scaled = centred / deviation if deviation > 0 else centred
            rows.append(
                {
                    "factor": factor,
                    "kind": self.kinds[factor],
                    "observations": column.size,
                    "daily_vol": float(deviation),
                    "worst": float(column.min()),
                    "best": float(column.max()),
                    "skew": float(np.mean(scaled**3)),
                    "excess_kurtosis": float(np.mean(scaled**4) - 3.0),
                    "note": self.notes.get(factor, ""),
                }
            )
        return pl.DataFrame(rows)


def build_factor_history(
    store: MarketStore,
    mappings: list[FactorMapping],
    start: date,
    end: date,
    knowledge_date: date | None = None,
) -> FactorHistory:
    """Assemble a factor history from the store.

    Only dates on which *every* factor was observed are kept. Filling a gap in
    one series with its previous value would inject a zero move on a day the
    others moved, which quietly understates correlation exactly when it matters.

    Args:
        store: The market store to read from.
        mappings: Where each factor comes from.
        start: First value date, inclusive.
        end: Last value date, inclusive.
        knowledge_date: Read the store as it was known on this date.

    Returns:
        The aligned factor history.

    Raises:
        ValueError: if no mappings are given or no common dates survive.
    """
    if not mappings:
        raise ValueError("no factor mappings given")

    series: list[pl.DataFrame] = []
    for mapping in mappings:
        levels = store.as_of(
            mapping.table,
            knowledge_date=knowledge_date,
            start=start,
            end=end,
            where=mapping.where,
        )
        if levels.is_empty():
            raise ValueError(f"{mapping.factor}: no data in {mapping.table} for {mapping.where}")
        frame = (
            levels.select("value_date", pl.col(mapping.column).alias(mapping.factor))
            .drop_nulls()
            .sort("value_date")
        )
        move = (
            (pl.col(mapping.factor) / pl.col(mapping.factor).shift(1)).log()
            if mapping.kind == RELATIVE
            else pl.col(mapping.factor).diff()
        )
        series.append(frame.with_columns(move.alias(mapping.factor)).drop_nulls())

    aligned = series[0]
    for frame in series[1:]:
        aligned = aligned.join(frame, on="value_date", how="inner")
    aligned = aligned.sort("value_date")

    if aligned.height < 2:
        raise ValueError("factor histories do not overlap on enough dates")

    factors = tuple(m.factor for m in mappings)
    return FactorHistory(
        factors=factors,
        dates=tuple(aligned["value_date"].to_list()),
        moves=aligned.select(factors).to_numpy().astype(np.float64),
        kinds={m.factor: m.kind for m in mappings},
        notes={m.factor: m.note for m in mappings if m.note},
    )
