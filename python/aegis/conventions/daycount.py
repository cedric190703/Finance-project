"""Day-count conventions.

A day-count convention answers a deceptively simple question: what fraction of a
year lies between two dates? Every discounting, accrual and theta calculation in
the engine goes through one of these, and picking the wrong one is a classic way
to be quietly wrong by a few basis points forever.

References:
    ISDA 2006 Definitions, Section 4.16 ("Day Count Fraction").
"""

from __future__ import annotations

from calendar import isleap
from datetime import date
from enum import StrEnum

__all__ = ["DayCount"]


def _days_in_year(year: int) -> int:
    return 366 if isleap(year) else 365


class DayCount(StrEnum):
    """The day-count conventions supported by the engine.

    Attributes:
        ACT_360: Actual days over 360. Money-market standard for USD/EUR deposits.
        ACT_365F: Actual days over a fixed 365. GBP money market, and the usual
            choice for equity-derivative time-to-expiry.
        THIRTY_360: 30/360 bond basis, ISDA 2006 4.16(f). US corporate and
            agency bonds.
        ACT_ACT_ISDA: Actual/Actual ISDA, which splits the period at year ends
            and divides each part by the length of its own year. Government
            bonds and IRS fixed legs.
    """

    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    THIRTY_360 = "30/360"
    ACT_ACT_ISDA = "ACT/ACT ISDA"

    def year_fraction(self, start: date, end: date) -> float:
        """Return the year fraction between two dates under this convention.

        The result is antisymmetric: swapping the arguments negates it, which
        keeps accrual arithmetic well behaved when a schedule runs backwards.

        Args:
            start: Period start date, inclusive.
            end: Period end date, exclusive.

        Returns:
            The accrual factor in years, negative when ``end`` precedes ``start``.
        """
        if end < start:
            return -self.year_fraction(end, start)
        if start == end:
            return 0.0

        match self:
            case DayCount.ACT_360:
                return (end - start).days / 360.0
            case DayCount.ACT_365F:
                return (end - start).days / 365.0
            case DayCount.THIRTY_360:
                return _thirty_360(start, end)
            case DayCount.ACT_ACT_ISDA:
                return _act_act_isda(start, end)

        raise AssertionError(f"unhandled day count: {self}")  # pragma: no cover


def _thirty_360(start: date, end: date) -> float:
    """30/360 bond basis with the ISDA 2006 4.16(f) date adjustments."""
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360.0


def _act_act_isda(start: date, end: date) -> float:
    """Actual/Actual ISDA: each calendar year contributes over its own length."""
    if start.year == end.year:
        return (end - start).days / _days_in_year(start.year)

    # Stub at the front, whole years in the middle, stub at the back.
    first_year_end = date(start.year + 1, 1, 1)
    last_year_start = date(end.year, 1, 1)
    head = (first_year_end - start).days / _days_in_year(start.year)
    tail = (end - last_year_start).days / _days_in_year(end.year)
    whole = end.year - start.year - 1
    return head + whole + tail
