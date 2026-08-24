"""Business-day calendars: known market holidays and adjustment invariants."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aegis.conventions import (
    NYSE,
    TARGET,
    WEEKENDS_ONLY,
    BusinessDayConvention,
    JointCalendar,
    adjust,
    easter_sunday,
)

dates = st.dates(min_value=date(2000, 1, 1), max_value=date(2050, 12, 31))


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5))],
)
def test_easter_matches_published_dates(year: int, expected: date) -> None:
    assert easter_sunday(year) == expected


@pytest.mark.parametrize(
    "holiday",
    [
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Washington's Birthday
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
    ],
)
def test_nyse_2024_holiday_calendar(holiday: date) -> None:
    assert not NYSE.is_business_day(holiday)


def test_nyse_traded_on_juneteenth_before_2022() -> None:
    assert NYSE.is_business_day(date(2019, 6, 19))
    assert not NYSE.is_business_day(date(2022, 6, 20))  # 19 June was a Sunday


def test_nyse_does_not_close_when_new_year_falls_on_a_saturday() -> None:
    # 1 January 2022 was a Saturday; the NYSE traded normally on Friday 31 Dec.
    assert NYSE.is_business_day(date(2021, 12, 31))


def test_independence_day_observed_on_the_friday_when_it_falls_on_a_saturday() -> None:
    assert date(2020, 7, 4).weekday() == 5
    assert not NYSE.is_business_day(date(2020, 7, 3))


@pytest.mark.parametrize(
    "holiday",
    [
        date(2024, 1, 1),
        date(2024, 3, 29),  # Good Friday
        date(2024, 4, 1),  # Easter Monday
        date(2024, 5, 1),  # Labour Day
        date(2024, 12, 25),
        date(2024, 12, 26),
    ],
)
def test_target_2024_holiday_calendar(holiday: date) -> None:
    assert not TARGET.is_business_day(holiday)


def test_target_and_nyse_differ_where_the_market_says_they_should() -> None:
    assert TARGET.is_business_day(date(2024, 7, 4))  # not a euro-area holiday
    assert NYSE.is_business_day(date(2024, 5, 1))  # not a US market holiday


@given(day=dates)
def test_weekends_only_calendar_is_open_on_every_weekday(day: date) -> None:
    assert WEEKENDS_ONLY.is_business_day(day) == (day.weekday() < 5)


@given(day=dates, count=st.integers(-40, 40))
def test_add_business_days_always_lands_on_a_business_day(day: date, count: int) -> None:
    assert NYSE.is_business_day(NYSE.add_business_days(day, count))


@given(day=dates, count=st.integers(1, 30))
def test_add_business_days_round_trips(day: date, count: int) -> None:
    start = NYSE.add_business_days(day, 0)
    assert NYSE.add_business_days(NYSE.add_business_days(start, count), -count) == start


@given(day=dates)
def test_next_and_previous_business_day_bracket_the_date(day: date) -> None:
    assert NYSE.previous_business_day(day) < day < NYSE.next_business_day(day)


@given(start=dates, gap=st.integers(0, 400))
def test_business_days_between_is_antisymmetric(start: date, gap: int) -> None:
    end = start + timedelta(days=gap)
    assert TARGET.business_days_between(start, end) == -TARGET.business_days_between(end, start)


@pytest.mark.parametrize(
    "convention",
    [c for c in BusinessDayConvention if c is not BusinessDayConvention.UNADJUSTED],
)
@given(day=dates)
def test_adjustment_always_produces_a_business_day(
    convention: BusinessDayConvention, day: date
) -> None:
    assert TARGET.is_business_day(adjust(day, TARGET, convention))


@given(day=dates)
def test_modified_following_never_leaves_the_month(day: date) -> None:
    assert adjust(day, TARGET, BusinessDayConvention.MODIFIED_FOLLOWING).month == day.month


@given(day=dates)
def test_unadjusted_is_the_identity(day: date) -> None:
    assert adjust(day, TARGET, BusinessDayConvention.UNADJUSTED) == day


def test_modified_following_rolls_back_at_a_month_end() -> None:
    # 31 May 2020 was a Sunday, so following would cross into June.
    assert adjust(date(2020, 5, 31), NYSE, BusinessDayConvention.MODIFIED_FOLLOWING) == date(
        2020, 5, 29
    )


def test_modified_preceding_rolls_forward_at_a_month_start() -> None:
    # 1 March 2020 was a Sunday, so preceding would cross into February.
    assert adjust(date(2020, 3, 1), NYSE, BusinessDayConvention.MODIFIED_PRECEDING) == date(
        2020, 3, 2
    )


def test_joint_calendar_closes_when_either_centre_closes() -> None:
    joint = JointCalendar((TARGET, NYSE))
    assert joint.name == "TARGET+NYSE"
    assert not joint.is_business_day(date(2024, 7, 4))  # US only
    assert not joint.is_business_day(date(2024, 5, 1))  # euro area only
    assert joint.is_business_day(date(2024, 7, 5))


@given(day=dates)
def test_joint_calendar_brackets_the_date(day: date) -> None:
    joint = JointCalendar((TARGET, NYSE))
    assert joint.previous_business_day(day) < day < joint.next_business_day(day)
