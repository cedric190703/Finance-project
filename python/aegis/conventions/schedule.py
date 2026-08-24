"""Business-day adjustment and periodic schedule generation.

Coupon schedules are generated backwards from maturity, which is what the market
does: the maturity date is the anchor everybody agrees on, and any odd period
lands at the front as a short or long first coupon.
"""

from __future__ import annotations

from datetime import date
from enum import IntEnum, StrEnum
from typing import Protocol

from aegis.conventions.tenor import Tenor, TenorUnit

__all__ = [
    "BusinessDayConvention",
    "CalendarLike",
    "Frequency",
    "StubConvention",
    "adjust",
    "generate_schedule",
]


class CalendarLike(Protocol):
    """The slice of a calendar that date adjustment needs."""

    @property
    def name(self) -> str:
        """Return the calendar's identifier."""
        ...

    def is_business_day(self, day: date) -> bool:
        """Return whether the date is a good business day."""
        ...

    def next_business_day(self, day: date) -> date:
        """Return the first business day after the date."""
        ...

    def previous_business_day(self, day: date) -> date:
        """Return the last business day before the date."""
        ...


class BusinessDayConvention(StrEnum):
    """How a date that falls on a holiday is moved to a good business day.

    Attributes:
        UNADJUSTED: Leave the date where it is.
        FOLLOWING: Roll forward.
        MODIFIED_FOLLOWING: Roll forward, unless that crosses into the next
            month, in which case roll back. The market default for swaps.
        PRECEDING: Roll back.
        MODIFIED_PRECEDING: Roll back, unless that crosses into the previous
            month, in which case roll forward.
    """

    UNADJUSTED = "UNADJUSTED"
    FOLLOWING = "FOLLOWING"
    MODIFIED_FOLLOWING = "MODIFIED_FOLLOWING"
    PRECEDING = "PRECEDING"
    MODIFIED_PRECEDING = "MODIFIED_PRECEDING"


class Frequency(IntEnum):
    """Coupon frequency, expressed as payments per year."""

    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12

    @property
    def tenor(self) -> Tenor:
        """Return the period between two consecutive payments."""
        return Tenor(12 // self.value, TenorUnit.MONTH)


class StubConvention(StrEnum):
    """Where an odd period is placed when the schedule does not divide evenly.

    Attributes:
        SHORT_FRONT: A short first period. The market default.
        SHORT_BACK: A short final period.
    """

    SHORT_FRONT = "SHORT_FRONT"
    SHORT_BACK = "SHORT_BACK"


def adjust(
    day: date,
    calendar: CalendarLike,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> date:
    """Move a date to a good business day under a roll convention.

    Args:
        day: The unadjusted date.
        calendar: The centre whose holidays apply.
        convention: The roll rule to apply.

    Returns:
        An adjusted date, guaranteed to be a business day unless the convention
        is ``UNADJUSTED``.
    """
    if convention is BusinessDayConvention.UNADJUSTED or calendar.is_business_day(day):
        return day

    match convention:
        case BusinessDayConvention.FOLLOWING:
            return calendar.next_business_day(day)
        case BusinessDayConvention.PRECEDING:
            return calendar.previous_business_day(day)
        case BusinessDayConvention.MODIFIED_FOLLOWING:
            rolled = calendar.next_business_day(day)
            return rolled if rolled.month == day.month else calendar.previous_business_day(day)
        case BusinessDayConvention.MODIFIED_PRECEDING:
            rolled = calendar.previous_business_day(day)
            return rolled if rolled.month == day.month else calendar.next_business_day(day)

    raise AssertionError(f"unhandled convention: {convention}")  # pragma: no cover


def generate_schedule(
    effective: date,
    maturity: date,
    frequency: Frequency,
    calendar: CalendarLike,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    stub: StubConvention = StubConvention.SHORT_FRONT,
) -> list[date]:
    """Generate an adjusted periodic schedule from effective date to maturity.

    Args:
        effective: The start of the first accrual period.
        maturity: The final payment date.
        frequency: How many periods there are per year.
        calendar: The centre whose holidays apply.
        convention: The roll rule applied to every generated date.
        stub: Whether an odd period lands at the front or the back.

    Returns:
        Adjusted dates in ascending order, starting with the effective date and
        ending with maturity. Consecutive entries bound one accrual period.

    Raises:
        ValueError: if maturity does not lie strictly after the effective date.
    """
    if maturity <= effective:
        raise ValueError(f"maturity {maturity} must fall after effective {effective}")

    period = frequency.tenor
    unadjusted: list[date] = []

    if stub is StubConvention.SHORT_FRONT:
        cursor = maturity
        while cursor > effective:
            unadjusted.append(cursor)
            cursor = Tenor(-period.count, period.unit).add_to(cursor)
        unadjusted.append(effective)
        unadjusted.reverse()
    else:
        cursor = effective
        while cursor < maturity:
            unadjusted.append(cursor)
            cursor = period.add_to(cursor)
        unadjusted.append(maturity)

    # Deduplicate the degenerate case where the stub collapses onto an anchor.
    seen: list[date] = []
    for day in unadjusted:
        if not seen or day != seen[-1]:
            seen.append(day)

    adjusted = [adjust(day, calendar, convention) for day in seen]
    return sorted(dict.fromkeys(adjusted))
