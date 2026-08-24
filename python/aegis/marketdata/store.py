"""The bitemporal market-data warehouse.

Every observation carries two dates:

``value_date``
    the date the observation *describes* — the session a close belongs to;
``knowledge_date``
    the date we *learned* it — when the payload arrived.

Keeping both is what lets the engine answer "what did the book look like on
14 March, using only what was known on 14 March?" months later. Without it, a
back-test silently uses restated prices — Yahoo rewrites every adjusted close
after a split — and reports a P&L nobody could have earned.

The tables are append-only. A revision does not overwrite its predecessor; it is
inserted with a later ``knowledge_date`` and wins only for queries that ask to
know about it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType

import duckdb
import polars as pl

__all__ = ["MarketStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_eod (
    symbol          VARCHAR   NOT NULL,
    value_date      DATE      NOT NULL,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE    NOT NULL,
    adj_close       DOUBLE,
    volume          BIGINT,
    source          VARCHAR   NOT NULL,
    knowledge_date  DATE      NOT NULL,
    ingested_at     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS curve_point (
    curve           VARCHAR   NOT NULL,
    value_date      DATE      NOT NULL,
    tenor           VARCHAR   NOT NULL,
    tenor_years     DOUBLE    NOT NULL,
    rate            DOUBLE    NOT NULL,
    source          VARCHAR   NOT NULL,
    knowledge_date  DATE      NOT NULL,
    ingested_at     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS fx_rate (
    pair            VARCHAR   NOT NULL,
    value_date      DATE      NOT NULL,
    rate            DOUBLE    NOT NULL,
    source          VARCHAR   NOT NULL,
    knowledge_date  DATE      NOT NULL,
    ingested_at     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS option_quote (
    underlying      VARCHAR   NOT NULL,
    value_date      DATE      NOT NULL,
    expiry          DATE      NOT NULL,
    strike          DOUBLE    NOT NULL,
    option_right    VARCHAR   NOT NULL,
    bid             DOUBLE,
    ask             DOUBLE,
    last            DOUBLE,
    provider_iv     DOUBLE,
    volume          BIGINT,
    open_interest   BIGINT,
    spot            DOUBLE,
    source          VARCHAR   NOT NULL,
    knowledge_date  DATE      NOT NULL,
    ingested_at     TIMESTAMP NOT NULL
);
"""

#: Table -> the columns that identify one observation, ignoring knowledge time.
_KEYS: dict[str, tuple[str, ...]] = {
    "price_eod": ("symbol", "value_date"),
    "curve_point": ("curve", "value_date", "tenor"),
    "fx_rate": ("pair", "value_date"),
    "option_quote": ("underlying", "value_date", "expiry", "strike", "option_right"),
}


class MarketStore:
    """A DuckDB-backed bitemporal store for market observations.

    Attributes:
        path: Database file, or ``":memory:"`` for an ephemeral store.
    """

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(self.path)
        self._con.execute(_SCHEMA)

    def __enter__(self) -> MarketStore:
        """Enter a context manager that closes the connection on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying connection."""
        self.close()

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._con.close()

    # ------------------------------------------------------------------ writes

    def append(self, table: str, frame: pl.DataFrame, source: str) -> int:
        """Append observations to a table without overwriting earlier ones.

        Rows whose ``(key..., knowledge_date)`` already exist are skipped, so
        re-running an ingest on the same day is idempotent, while a genuine
        revision arriving on a later day is retained alongside the original.

        Args:
            table: Target table name.
            frame: Observations to append; must carry a ``knowledge_date`` column.
            source: Provider that supplied the data.

        Returns:
            The number of rows actually inserted.

        Raises:
            ValueError: if the table is unknown or ``knowledge_date`` is missing.
        """
        if table not in _KEYS:
            raise ValueError(f"unknown table: {table}")
        if frame.is_empty():
            return 0
        if "knowledge_date" not in frame.columns:
            raise ValueError(f"{table} rows must carry a knowledge_date")

        columns = [c[0] for c in self._con.execute(f"DESCRIBE {table}").fetchall()]
        staged = frame.with_columns(
            pl.lit(source).alias("source"),
            pl.lit(datetime.now(UTC).replace(tzinfo=None)).alias("ingested_at"),
        )
        missing = [c for c in columns if c not in staged.columns]
        staged = staged.with_columns([pl.lit(None).alias(c) for c in missing]).select(columns)

        keys = [*_KEYS[table], "knowledge_date"]
        predicate = " AND ".join(f"t.{k} = s.{k}" for k in keys)
        self._con.register("staged", staged.to_arrow())
        before = self._row_count(table)
        self._con.execute(
            f"""
            INSERT INTO {table}
            SELECT * FROM staged s
            WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {predicate})
            """  # noqa: S608 - table name is validated against _KEYS above
        )
        self._con.unregister("staged")
        return self._row_count(table) - before

    # ------------------------------------------------------------------- reads

    def as_of(
        self,
        table: str,
        knowledge_date: date | None = None,
        start: date | None = None,
        end: date | None = None,
        where: str | None = None,
    ) -> pl.DataFrame:
        """Read a table as it was known on a given date.

        For each observation key, the row with the latest ``knowledge_date`` not
        after ``knowledge_date`` wins. This is the only read path the valuation
        engine uses, which is what makes a historical run reproducible.

        Args:
            table: Table to read.
            knowledge_date: Point in knowledge time; ``None`` means "everything
                known today".
            start: Earliest ``value_date`` to return, inclusive.
            end: Latest ``value_date`` to return, inclusive.
            where: Additional SQL predicate, e.g. ``"symbol = 'AAPL'"``.

        Returns:
            One row per observation key, with the winning revision.

        Raises:
            ValueError: if the table is unknown.
        """
        if table not in _KEYS:
            raise ValueError(f"unknown table: {table}")

        clauses: list[str] = []
        params: list[object] = []
        if knowledge_date is not None:
            clauses.append("knowledge_date <= ?")
            params.append(knowledge_date)
        if start is not None:
            clauses.append("value_date >= ?")
            params.append(start)
        if end is not None:
            clauses.append("value_date <= ?")
            params.append(end)
        if where:
            clauses.append(f"({where})")
        predicate = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        partition = ", ".join(_KEYS[table])

        sql = f"""
            SELECT * FROM {table}
            {predicate}
            QUALIFY row_number() OVER (
                PARTITION BY {partition}
                ORDER BY knowledge_date DESC, ingested_at DESC
            ) = 1
            ORDER BY value_date
        """  # noqa: S608 - table and partition come from _KEYS, values are bound
        return self._con.execute(sql, params).pl()

    def revisions(self, table: str, where: str | None = None) -> pl.DataFrame:
        """Return observations that were later restated.

        A non-empty result is the concrete argument for storing knowledge time:
        these are values that would differ between a live run and a naive
        historical rebuild.

        Args:
            table: Table to inspect.
            where: Additional SQL predicate.

        Returns:
            Every key with more than one distinct value across knowledge dates,
            ordered oldest first.

        Raises:
            ValueError: if the table is unknown.
        """
        if table not in _KEYS:
            raise ValueError(f"unknown table: {table}")
        keys = ", ".join(_KEYS[table])
        value = "close" if table == "price_eod" else "rate"
        predicate = f"WHERE {where}" if where else ""
        sql = f"""
            WITH history AS (
                SELECT {keys}, knowledge_date, {value} AS value FROM {table} {predicate}
            )
            SELECT * FROM history
            WHERE ({keys}) IN (
                SELECT {keys} FROM history GROUP BY {keys} HAVING count(DISTINCT value) > 1
            )
            ORDER BY {keys}, knowledge_date
        """  # noqa: S608 - identifiers come from _KEYS
        return self._con.execute(sql).pl()

    def coverage(self) -> pl.DataFrame:
        """Summarise what the store holds, per table.

        Returns:
            Row counts and the value-date and knowledge-date ranges per table.
        """
        parts = [
            f"""
            SELECT '{table}' AS table_name, count(*) AS rows,
                   min(value_date) AS first_value_date,
                   max(value_date) AS last_value_date,
                   min(knowledge_date) AS first_knowledge_date,
                   max(knowledge_date) AS last_knowledge_date
            FROM {table}
            """  # noqa: S608 - table names are literals from _KEYS
            for table in _KEYS
        ]
        return self._con.execute(" UNION ALL ".join(parts) + " ORDER BY table_name").pl()

    def sql(self, query: str) -> pl.DataFrame:
        """Run an arbitrary read-only query against the store.

        Args:
            query: SQL text.

        Returns:
            The result as a Polars frame.
        """
        return self._con.execute(query).pl()

    def _row_count(self, table: str) -> int:
        result = self._con.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
        return int(result[0]) if result else 0
