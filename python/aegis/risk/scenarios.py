"""Applying shocks to a market snapshot.

One function does the work: given a snapshot and a set of factor moves, produce
the snapshot that would have obtained. Everything in the risk layer is built on
it — historical VaR replays five hundred of yesterday's moves through it, Monte
Carlo VaR pushes simulated ones, and a stress test pushes a scenario somebody
wrote down after 2008.

Full revaluation, not a Taylor expansion. A delta-gamma approximation of a book
with options in it is fine for a one percent move and badly wrong for the twenty
percent move that actually matters, which is precisely the move a stress test is
asking about.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import yaml

from aegis.conventions import Tenor
from aegis.curves import DiscountCurve
from aegis.instruments import MarketSnapshot

__all__ = ["Scenario", "ScenarioError", "apply_shocks", "load_scenarios"]

_BASIS_POINTS_PER_UNIT = 1e4


class ScenarioError(ValueError):
    """Raised when a scenario definition cannot be understood."""


@dataclass(frozen=True)
class Scenario:
    """A named set of factor shocks.

    Attributes:
        name: Short identifier.
        description: What the scenario represents and where it came from.
        shocks: Factor identifier to move, in engine units — log moves for
            prices, volatilities and FX, absolute decimals for rates.
    """

    name: str
    description: str
    shocks: dict[str, float]

    def apply(self, market: MarketSnapshot) -> MarketSnapshot:
        """Return the market as it would look under this scenario.

        Args:
            market: The starting market state.

        Returns:
            The shocked snapshot.
        """
        return apply_shocks(market, self.shocks)


def apply_shocks(market: MarketSnapshot, shocks: Mapping[str, float]) -> MarketSnapshot:
    """Return a snapshot with the given factor moves applied.

    Args:
        market: The starting market state.
        shocks: Factor identifier to move. ``SPOT:``, ``VOL:`` and ``FX:`` moves
            are *log* moves, applied as ``exp(move)``; ``RATE:`` moves are
            absolute, in decimal. Log moves are what make horizon scaling safe —
            see the note in `aegis.risk.factors`.

    Returns:
        A new snapshot. The original is never modified.

    Raises:
        ScenarioError: if a factor identifier is not recognised.
    """
    shocked = market
    spot_factors: dict[str, float] = {}
    vol_factors: dict[str, float] = {}
    fx_factors: dict[str, float] = {}
    curve_shifts: dict[str, list[tuple[str, float]]] = {}

    for factor, move in shocks.items():
        parts = factor.split(":")
        match parts:
            case ["SPOT", symbol]:
                spot_factors[symbol] = spot_factors.get(symbol, 1.0) * math.exp(move)
            case ["VOL", symbol]:
                vol_factors[symbol] = vol_factors.get(symbol, 1.0) * math.exp(move)
            case ["FX", currency]:
                fx_factors[currency] = fx_factors.get(currency, 1.0) * math.exp(move)
            case ["RATE", currency]:
                curve_shifts.setdefault(currency, []).append(("", move))
            case ["RATE", currency, tenor]:
                curve_shifts.setdefault(currency, []).append((tenor, move))
            case _:
                raise ScenarioError(f"unrecognised risk factor {factor!r}")

    if spot_factors:
        shocked = shocked.with_spots(**spot_factors)
    if vol_factors:
        shocked = shocked.with_vols_scaled(
            **{s: f for s, f in vol_factors.items() if s in shocked.surfaces}
        )
    if fx_factors:
        shocked = shocked.with_fx_scaled(**fx_factors)
    if curve_shifts:
        shocked = replace(shocked, curves=_shift_curves(shocked, curve_shifts))
    return shocked


def _shift_curves(
    market: MarketSnapshot, shifts: dict[str, list[tuple[str, float]]]
) -> dict[str, DiscountCurve]:
    """Apply parallel and key-rate shifts to the affected curves."""
    curves = dict(market.curves)
    for currency, moves in shifts.items():
        if currency not in curves:
            continue
        curve = curves[currency]
        for tenor, move in moves:
            basis_points = move * _BASIS_POINTS_PER_UNIT
            curve = (
                curve.shift_parallel(basis_points)
                if not tenor
                else curve.shift_key_rate(Tenor.parse(tenor).approximate_years, basis_points)
            )
        curves[currency] = curve
    return curves


def load_scenarios(path: Path | str) -> list[Scenario]:
    """Load stress scenarios from a YAML file.

    Args:
        path: Path to the definition file.

    Returns:
        The scenarios, in file order.

    Raises:
        ScenarioError: if the file is malformed.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "scenarios" not in raw:
        raise ScenarioError(f"{path}: expected a mapping with a 'scenarios' key")

    # Scenario files are written by people, so they are read in the units people
    # think in: "-0.25" means a twenty-five percent fall. The engine works in log
    # moves, so the conversion happens once, here, rather than being remembered
    # at every call site.
    scenarios = []
    for index, entry in enumerate(raw["scenarios"]):
        if not isinstance(entry, dict) or "name" not in entry:
            raise ScenarioError(f"{path}: scenario {index} needs a name")
        shocks = entry.get("shocks", {})
        if not isinstance(shocks, dict):
            raise ScenarioError(f"{path}: scenario {entry['name']} has malformed shocks")
        scenarios.append(
            Scenario(
                name=str(entry["name"]),
                description=str(entry.get("description", "")),
                shocks={str(k): _to_engine_units(str(k), float(v)) for k, v in shocks.items()},
            )
        )
    return scenarios


def _to_engine_units(factor: str, value: float) -> float:
    """Convert a human-written shock into the engine's units.

    Args:
        factor: The factor the shock applies to.
        value: The shock as written: a simple return for prices, volatilities
            and FX; an absolute decimal for rates.

    Returns:
        The shock in engine units.

    Raises:
        ScenarioError: if a relative shock wipes out more than the whole value.
    """
    if factor.startswith("RATE:"):
        return value
    if value <= -1.0:
        raise ScenarioError(f"{factor}: a relative shock of {value} would take the level to zero")
    return math.log1p(value)


def horizon_dates(value_date: date, days: int) -> tuple[date, date]:
    """Return the start and end of a risk horizon.

    Args:
        value_date: The starting date.
        days: Length of the horizon in calendar days.

    Returns:
        The pair of dates.
    """
    from datetime import timedelta

    return value_date, value_date + timedelta(days=days)
