"""Tenors: the ``3M`` / ``10Y`` shorthand that market data is quoted in."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

__all__ = ["Tenor", "TenorUnit"]

_PATTERN = re.compile(r"^\s*(-?\d+)\s*([DWMY])\s*$", re.IGNORECASE)


class TenorUnit(StrEnum):
    """The time unit of a tenor."""

    DAY = "D"
    WEEK = "W"
    MONTH = "M"
    YEAR = "Y"


@dataclass(frozen=True, order=True)
class Tenor:
    """A calendar offset such as ``3M`` or ``10Y``.

    Attributes:
        count: The number of units; may be negative to step backwards.
        unit: The unit the count is expressed in.
    """

    count: int
    unit: TenorUnit

    @classmethod
    def parse(cls, text: str) -> Tenor:
        """Parse a market tenor string.

        Args:
            text: A string such as ``"1D"``, ``"2W"``, ``"3M"`` or ``"10Y"``.

        Returns:
            The parsed tenor.

        Raises:
            ValueError: if the string is not a recognised tenor.
        """
        match = _PATTERN.match(text)
        if match is None:
            raise ValueError(f"not a tenor: {text!r}")
        return cls(int(match.group(1)), TenorUnit(match.group(2).upper()))

    @property
    def approximate_years(self) -> float:
        """Return a rough length in years, for sorting and interpolation only."""
        factors = {
            TenorUnit.DAY: 1.0 / 365.0,
            TenorUnit.WEEK: 7.0 / 365.0,
            TenorUnit.MONTH: 1.0 / 12.0,
            TenorUnit.YEAR: 1.0,
        }
        return self.count * factors[self.unit]

    def add_to(self, day: date) -> date:
        """Advance a date by this tenor, without any business-day adjustment.

        Month and year steps clamp to the end of the target month, so
        ``31 January + 1M`` is the last day of February rather than an error.

        Args:
            day: The date to advance.

        Returns:
            The unadjusted resulting date.
        """
        match self.unit:
            case TenorUnit.DAY:
                return day + timedelta(days=self.count)
            case TenorUnit.WEEK:
                return day + timedelta(weeks=self.count)
            case TenorUnit.MONTH:
                return _add_months(day, self.count)
            case TenorUnit.YEAR:
                return _add_months(day, 12 * self.count)
        raise AssertionError(f"unhandled tenor unit: {self.unit}")  # pragma: no cover

    def __str__(self) -> str:
        """Return the canonical string form, e.g. ``3M``."""
        return f"{self.count}{self.unit.value}"


def _add_months(day: date, months: int) -> date:
    total = day.month - 1 + months
    year = day.year + total // 12
    month = total % 12 + 1
    last_day = _days_in_month(year, month)
    return date(year, month, min(day.day, last_day))


def _days_in_month(year: int, month: int) -> int:
    from calendar import monthrange

    return monthrange(year, month)[1]
