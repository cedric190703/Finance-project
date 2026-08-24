"""Aegis: a multi-asset risk and P&L engine.

The package is organised as a pipeline, and the subpackages follow the order in
which a valuation run touches them:

``conventions``
    Day-count fractions, business-day calendars and roll rules.
``marketdata``
    Provider adapters, the raw response archive and the bitemporal store.
``curves`` / ``vol``
    Discount curves bootstrapped from par yields, and arbitrage-checked
    implied-volatility surfaces.
``instruments`` / ``pricing``
    The trade representations and the engines that value them; the heavy
    numerical work is delegated to the compiled Rust core.
``risk`` / ``pnl``
    Greeks, VaR/ES, stress scenarios, model validation, and the daily P&L
    attribution report.
"""

from __future__ import annotations

__all__ = ["__version__", "core_version"]

__version__ = "0.1.0"


def core_version() -> str:
    """Return the version of the compiled Rust kernel library.

    Raises:
        ImportError: if the ``aegis._core`` extension module has not been built.
    """
    from aegis import _core

    return str(_core.core_version())
