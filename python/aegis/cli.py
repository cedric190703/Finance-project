"""Command-line entry point for the Aegis engine."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from aegis import __version__
from aegis.marketdata import (
    CboeProvider,
    EcbProvider,
    FredProvider,
    MarketStore,
    RawArchive,
    YahooProvider,
    ingest_fx,
    ingest_option_chain,
    ingest_prices,
    ingest_treasury,
)

DEFAULT_DB = Path("data/warehouse/market.duckdb")
DEFAULT_ARCHIVE = Path("data/raw")

app = typer.Typer(
    name="aegis",
    help="Multi-asset risk and P&L engine.",
    no_args_is_help=True,
    add_completion=False,
)
fetch_app = typer.Typer(help="Pull market data into the bitemporal store.", no_args_is_help=True)
store_app = typer.Typer(help="Inspect the bitemporal store.", no_args_is_help=True)
app.add_typer(fetch_app, name="fetch")
app.add_typer(store_app, name="store")
console = Console()

DbOption = Annotated[Path, typer.Option("--db", help="DuckDB warehouse file.")]
ArchiveOption = Annotated[Path, typer.Option("--archive", help="Raw payload archive root.")]
StartOption = Annotated[str, typer.Option("--start", help="First value date (YYYY-MM-DD).")]
EndOption = Annotated[str, typer.Option("--end", help="Last value date (YYYY-MM-DD).")]


@app.callback()
def main() -> None:
    """Multi-asset risk and P&L engine."""


@app.command()
def version() -> None:
    """Print the Python package and compiled-core versions."""
    console.print(f"aegis        {__version__}")
    try:
        from aegis import core_version

        console.print(f"aegis._core  {core_version()}")
    except ImportError:  # pragma: no cover - only when the wheel is not built
        console.print("[yellow]aegis._core  not built (run `make build`)[/yellow]")


def _default_start() -> str:
    return (date.today() - timedelta(days=730)).isoformat()


@fetch_app.command("curve")
def fetch_curve(
    start: StartOption = "",
    end: EndOption = "",
    db: DbOption = DEFAULT_DB,
    archive: ArchiveOption = DEFAULT_ARCHIVE,
) -> None:
    """Fetch the US Treasury constant-maturity par curve from FRED."""
    with MarketStore(db) as store:
        provider = FredProvider(RawArchive(archive))
        result = ingest_treasury(store, provider, _day(start or _default_start()), _day(end))
        console.print(str(result))


@fetch_app.command("index")
def fetch_index(
    series: Annotated[list[str], typer.Argument(help="FRED series ids, e.g. SP500 VIXCLS.")],
    start: StartOption = "",
    end: EndOption = "",
    db: DbOption = DEFAULT_DB,
    archive: ArchiveOption = DEFAULT_ARCHIVE,
) -> None:
    """Fetch daily index levels from FRED into the price table."""
    with MarketStore(db) as store:
        provider = FredProvider(RawArchive(archive))
        frame = provider.series(list(series), _day(start or _default_start()), _day(end))
        inserted = store.append("price_eod", frame, source=provider.name)
        console.print(f"price_eod/{'+'.join(series)}: {inserted} new of {frame.height}")


@fetch_app.command("fx")
def fetch_fx(
    currencies: Annotated[list[str], typer.Argument(help="ISO codes against EUR, e.g. USD GBP.")],
    start: StartOption = "",
    end: EndOption = "",
    db: DbOption = DEFAULT_DB,
    archive: ArchiveOption = DEFAULT_ARCHIVE,
) -> None:
    """Fetch ECB euro reference fixings."""
    with MarketStore(db) as store:
        provider = EcbProvider(RawArchive(archive))
        for result in ingest_fx(
            store, provider, list(currencies), _day(start or _default_start()), _day(end)
        ):
            console.print(str(result))


@fetch_app.command("options")
def fetch_options(
    symbols: Annotated[list[str], typer.Argument(help="Underlying tickers, e.g. KO SPY.")],
    db: DbOption = DEFAULT_DB,
    archive: ArchiveOption = DEFAULT_ARCHIVE,
) -> None:
    """Fetch delayed option chains from Cboe, with the matching underlying quote."""
    with MarketStore(db) as store:
        provider = CboeProvider(RawArchive(archive))
        for symbol in symbols:
            for result in ingest_option_chain(store, provider, symbol):
                console.print(str(result))


@fetch_app.command("prices")
def fetch_prices(
    symbols: Annotated[list[str], typer.Argument(help="Yahoo tickers, e.g. AAPL MSFT.")],
    start: StartOption = "",
    end: EndOption = "",
    db: DbOption = DEFAULT_DB,
    archive: ArchiveOption = DEFAULT_ARCHIVE,
) -> None:
    """Fetch daily equity bars from Yahoo Finance."""
    with MarketStore(db) as store:
        provider = YahooProvider(RawArchive(archive))
        for result in ingest_prices(
            store, provider, list(symbols), _day(start or _default_start()), _day(end)
        ):
            console.print(str(result))


@store_app.command("coverage")
def store_coverage(db: DbOption = DEFAULT_DB) -> None:
    """Summarise what the store holds, per table."""
    with MarketStore(db) as store:
        _print_frame("Store coverage", store.coverage())


@store_app.command("revisions")
def store_revisions(
    table: Annotated[str, typer.Argument(help="Table to inspect, e.g. price_eod.")],
    where: Annotated[str, typer.Option("--where", help="Extra SQL predicate.")] = "",
) -> None:
    """Show observations that were restated after we first saw them."""
    with MarketStore(DEFAULT_DB) as store:
        frame = store.revisions(table, where=where or None)
        if frame.is_empty():
            console.print(f"[green]no restatements recorded in {table}[/green]")
            return
        _print_frame(f"Restatements in {table}", frame)


def _day(text: str) -> date:
    return date.fromisoformat(text) if text else date.today()


def _print_frame(title: str, frame: object) -> None:
    import polars as pl

    assert isinstance(frame, pl.DataFrame)  # noqa: S101 - internal helper contract
    table = Table(title=title, header_style="bold")
    for column in frame.columns:
        table.add_column(column)
    for row in frame.iter_rows():
        table.add_row(*(("" if v is None else str(v)) for v in row))
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
