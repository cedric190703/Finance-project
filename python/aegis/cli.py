"""Command-line entry point for the Aegis engine."""

from __future__ import annotations

import typer
from rich.console import Console

from aegis import __version__

app = typer.Typer(
    name="aegis",
    help="Multi-asset risk and P&L engine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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
    except ImportError:
        console.print("[yellow]aegis._core  not built (run `make build`)[/yellow]")


if __name__ == "__main__":  # pragma: no cover
    app()
