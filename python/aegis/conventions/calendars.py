"""Business-day calendars.

Holidays are generated per year on demand and cached, rather than shipped as a
static list that silently expires. Two centres are implemented — TARGET (the
euro-area settlement calendar) and NYSE (the US equity market) — plus a joint
calendar for cross-currency trades, where a date must be good in both centres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

__all__ = ["NYSE", "TARGET", "WEEKENDS_ONLY", "BusinessCalendar", "JointCalendar"]

_SATURDAY = 5
_SUNDAY = 6
_JUNETEENTH_FIRST_YEAR = 2022  # first year the NYSE closed for Juneteenth


@lru_cache(maxsize=512)
def easter_sunday(year: int) -> date:
    """Return Gregorian Easter Sunday for a year (anonymous Gregorian algorithm).

    Args:
        year: Calendar year.

    Returns:
        The date of Easter Sunday.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th ``weekday`` (Mon=0) of a month; ``n = -1`` means the last."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    next_month = date(year + month // 12, month % 12 + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


@dataclass(frozen=True)
class BusinessCalendar:
    """A named business-day calendar backed by a per-year holiday generator.

    Attributes:
        name: Identifier used in reports and configuration files.
    """

    name: str
    _generator: str = field(repr=False)

    def holidays(self, year: int) -> frozenset[date]:
        """Return the public holidays observed in a given year.

        Args:
            year: Calendar year.

        Returns:
            The set of holiday dates, excluding weekends.
        """
        return _holidays_for(self._generator, year)

    def is_business_day(self, day: date) -> bool:
        """Return whether a date is a good business day in this centre.

        Args:
            day: The date to test.

        Returns:
            ``True`` when the date is neither a weekend nor a holiday.
        """
        if day.weekday() >= _SATURDAY:
            return False
        return day not in self.holidays(day.year)

    def next_business_day(self, day: date) -> date:
        """Return the first business day strictly after ``day``.

        Args:
            day: The reference date.

        Returns:
            The next good business day.
        """
        candidate = day + timedelta(days=1)
        while not self.is_business_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def previous_business_day(self, day: date) -> date:
        """Return the last business day strictly before ``day``.

        Args:
            day: The reference date.

        Returns:
            The previous good business day.
        """
        candidate = day - timedelta(days=1)
        while not self.is_business_day(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def add_business_days(self, day: date, count: int) -> date:
        """Shift a date by a whole number of business days.

        Used for settlement lags: ``T+2`` on a spot FX trade is
        ``add_business_days(trade_date, 2)``.

        Args:
            day: The reference date.
            count: Number of business days to add; may be negative.

        Returns:
            The shifted date, always a good business day.
        """
        step = self.next_business_day if count >= 0 else self.previous_business_day
        result = day
        for _ in range(abs(count)):
            result = step(result)
        if count == 0:
            while not self.is_business_day(result):
                result = self.next_business_day(result)
        return result

    def business_days_between(self, start: date, end: date) -> int:
        """Count business days in ``[start, end)``.

        Args:
            start: Period start, inclusive.
            end: Period end, exclusive.

        Returns:
            The number of good business days, negative when the period is reversed.
        """
        if end < start:
            return -self.business_days_between(end, start)
        count = 0
        day = start
        while day < end:
            if self.is_business_day(day):
                count += 1
            day += timedelta(days=1)
        return count


@dataclass(frozen=True)
class JointCalendar:
    """The intersection of several calendars: a date must be good in all of them.

    Attributes:
        calendars: The centres that must simultaneously be open.
    """

    calendars: tuple[BusinessCalendar, ...]

    @property
    def name(self) -> str:
        """Return the composite name, e.g. ``TARGET+NYSE``."""
        return "+".join(c.name for c in self.calendars)

    def is_business_day(self, day: date) -> bool:
        """Return whether the date is a business day in every constituent centre.

        Args:
            day: The date to test.

        Returns:
            ``True`` only when all calendars are open.
        """
        return all(c.is_business_day(day) for c in self.calendars)

    def next_business_day(self, day: date) -> date:
        """Return the first jointly good business day strictly after ``day``.

        Args:
            day: The reference date.

        Returns:
            The next date open in every centre.
        """
        candidate = day + timedelta(days=1)
        while not self.is_business_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def previous_business_day(self, day: date) -> date:
        """Return the last jointly good business day strictly before ``day``.

        Args:
            day: The reference date.

        Returns:
            The previous date open in every centre.
        """
        candidate = day - timedelta(days=1)
        while not self.is_business_day(candidate):
            candidate -= timedelta(days=1)
        return candidate


@lru_cache(maxsize=4096)
def _holidays_for(generator: str, year: int) -> frozenset[date]:
    if generator == "weekends":
        return frozenset()
    if generator == "target":
        return _target_holidays(year)
    if generator == "nyse":
        return _nyse_holidays(year)
    raise ValueError(f"unknown holiday generator: {generator}")


def _target_holidays(year: int) -> frozenset[date]:
    """TARGET2 closing days: fixed dates plus the Easter pair, weekends dropped."""
    easter = easter_sunday(year)
    days = {
        date(year, 1, 1),
        easter - timedelta(days=2),  # Good Friday
        easter + timedelta(days=1),  # Easter Monday
        date(year, 5, 1),  # Labour Day
        date(year, 12, 25),
        date(year, 12, 26),
    }
    return frozenset(d for d in days if d.weekday() < _SATURDAY)


def _nyse_holidays(year: int) -> frozenset[date]:
    """NYSE trading holidays, including the Saturday/Sunday observation rules."""
    easter = easter_sunday(year)
    days: set[date] = {
        _observed(date(year, 1, 1), new_year=True),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        easter - timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= _JUNETEENTH_FIRST_YEAR:
        days.add(_observed(date(year, 6, 19)))
    return frozenset(d for d in days if d.weekday() < _SATURDAY)


def _observed(day: date, *, new_year: bool = False) -> date:
    """Apply the US observation rule: Sunday rolls forward, Saturday back.

    New Year's Day is the exception the NYSE actually applies: when 1 January
    falls on a Saturday the market simply trades on the preceding Friday rather
    than closing for it.
    """
    if day.weekday() == _SUNDAY:
        return day + timedelta(days=1)
    if day.weekday() == _SATURDAY:
        return day if new_year else day - timedelta(days=1)
    return day


WEEKENDS_ONLY = BusinessCalendar("WEEKENDS", "weekends")
TARGET = BusinessCalendar("TARGET", "target")
NYSE = BusinessCalendar("NYSE", "nyse")
