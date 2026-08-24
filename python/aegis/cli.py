"""Command-line entry point for the Aegis engine."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from aegis import __version__
from aegis.curves import Interpolation, curve_from_store
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
from aegis.vol import build_surface

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
curve_app = typer.Typer(help="Build and inspect discount curves.", no_args_is_help=True)
vol_app = typer.Typer(help="Calibrate and inspect volatility surfaces.", no_args_is_help=True)
app.add_typer(fetch_app, name="fetch")
app.add_typer(store_app, name="store")
app.add_typer(curve_app, name="curve")
app.add_typer(vol_app, name="vol")
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


@curve_app.command("show")
def curve_show(
    value_date: Annotated[str, typer.Option("--date", help="Session to build (YYYY-MM-DD).")] = "",
    interpolation: Annotated[
        Interpolation, typer.Option("--interpolation", help="Interpolation scheme.")
    ] = Interpolation.LOG_LINEAR_DISCOUNT,
    knowledge_date: Annotated[
        str, typer.Option("--as-known-on", help="Rebuild using only what was known then.")
    ] = "",
    db: DbOption = DEFAULT_DB,
) -> None:
    """Bootstrap the Treasury curve for one session and print it."""
    import polars as pl

    session = _day(value_date)
    with MarketStore(db) as store:
        quotes = store.as_of(
            "curve_point",
            knowledge_date=_day(knowledge_date) if knowledge_date else None,
            start=session,
            end=session,
        )
    if quotes.is_empty():
        console.print(f"[red]no curve quotes stored for {session}[/red]")
        raise typer.Exit(code=1)

    curve = curve_from_store(quotes, session, interpolation=interpolation)
    table = Table(title=f"{curve.name} curve — {session} ({interpolation})", header_style="bold")
    for column in ("tenor", "years", "par yield", "zero rate", "discount factor"):
        table.add_column(column, justify="right")
    quoted = dict(zip(quotes["tenor"].to_list(), quotes["rate"].to_list(), strict=True))
    for label, years, zero, discount in curve.knot_table():
        table.add_row(
            label,
            f"{years:.4f}",
            f"{quoted.get(label, float('nan')):.4%}",
            f"{zero:.4%}",
            f"{discount:.6f}",
        )
    console.print(table)

    forwards = pl.DataFrame(
        {
            "period": ["0-1y", "1-2y", "2-5y", "5-10y", "10-30y"],
            "forward": [
                curve.forward_rate(a, b)
                for a, b in ((0.01, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0))
            ],
        }
    ).with_columns(pl.col("forward").map_elements(lambda r: f"{r:.4%}", return_dtype=pl.String))
    _print_frame("Implied forwards", forwards)


@vol_app.command("surface")
def vol_surface(
    symbol: Annotated[str, typer.Argument(help="Underlying ticker, e.g. KO.")],
    value_date: Annotated[str, typer.Option("--date", help="Chain date (YYYY-MM-DD).")] = "",
    db: DbOption = DEFAULT_DB,
) -> None:
    """Calibrate the SVI surface for one underlying and print it."""
    import polars as pl

    with MarketStore(db) as store:
        chain = store.as_of("option_quote", where=f"underlying = '{symbol.upper()}'")
        if value_date:
            chain = chain.filter(pl.col("value_date") == _day(value_date))
        if chain.is_empty():
            console.print(f"[red]no option quotes stored for {symbol.upper()}[/red]")
            raise typer.Exit(code=1)

        session = chain["value_date"].max()
        assert isinstance(session, date)  # noqa: S101 - column type is known
        chain = chain.filter(pl.col("value_date") == session)
        quotes = store.as_of("curve_point", start=session - timedelta(days=30), end=session)

    if quotes.is_empty():
        console.print("[red]no curve quotes stored near that date; run `aegis fetch curve`[/red]")
        raise typer.Exit(code=1)

    curve = curve_from_store(quotes, max(quotes["value_date"].to_list()))
    surface = build_surface(chain, curve, reference_date=session)

    table = Table(title=f"{surface.underlying} volatility surface — {session}", header_style="bold")
    columns = ("expiry", "years", "forward", "ATM vol", "skew 90-110", "RMSE", "quotes", "fit")
    for column in columns:
        table.add_column(column, justify="right")
    for calibrated in surface.slices:
        low = float(calibrated.implied_vol(calibrated.forward * 0.9)[0])
        high = float(calibrated.implied_vol(calibrated.forward * 1.1)[0])
        table.add_row(
            calibrated.expiry.isoformat(),
            f"{calibrated.time:.3f}",
            f"{calibrated.forward:.2f}",
            f"{calibrated.atm_vol:.2%}",
            f"{low - high:+.2%}",
            f"{calibrated.rmse_total_variance:.2e}",
            str(calibrated.quote_count),
            "[yellow]repaired[/yellow]" if calibrated.repaired else "clean",
        )
    console.print(table)
    console.print(f"spot {surface.spot:.2f} — {surface.check_arbitrage()}")


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
