"""Day-count conventions: ISDA reference values and structural invariants."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aegis.conventions import DayCount

dates = st.dates(min_value=date(1990, 1, 1), max_value=date(2070, 12, 31))


@pytest.mark.parametrize(
    ("convention", "start", "end", "expected"),
    [
        # ISDA 2006 examples: a short money-market period.
        (DayCount.ACT_360, date(2024, 1, 1), date(2024, 7, 1), 182 / 360),
        (DayCount.ACT_365F, date(2024, 1, 1), date(2024, 7, 1), 182 / 365),
        # 30/360 treats every month as 30 days: exactly half a year.
        (DayCount.THIRTY_360, date(2024, 1, 1), date(2024, 7, 1), 0.5),
        # 31 January to 28 February under 30/360: d1 clamps to 30.
        (DayCount.THIRTY_360, date(2024, 1, 31), date(2024, 2, 28), 28 / 360),
        # A whole non-leap year under ACT/ACT ISDA is exactly one.
        (DayCount.ACT_ACT_ISDA, date(2023, 1, 1), date(2024, 1, 1), 1.0),
        # A whole leap year is also exactly one: 366 days over a 366-day year.
        (DayCount.ACT_ACT_ISDA, date(2024, 1, 1), date(2025, 1, 1), 1.0),
    ],
)
def test_known_year_fractions(
    convention: DayCount, start: date, end: date, expected: float
) -> None:
    assert convention.year_fraction(start, end) == pytest.approx(expected, abs=1e-12)


def test_act_act_isda_splits_a_period_across_the_year_end() -> None:
    # 15 Dec 2023 to 15 Jan 2024: 17 days in a 365-day year, then 14 in a 366-day one.
    got = DayCount.ACT_ACT_ISDA.year_fraction(date(2023, 12, 15), date(2024, 1, 15))
    assert got == pytest.approx(17 / 365 + 14 / 366, abs=1e-12)


@pytest.mark.parametrize("convention", list(DayCount))
@given(start=dates, end=dates)
def test_year_fraction_is_antisymmetric(convention: DayCount, start: date, end: date) -> None:
    assert convention.year_fraction(start, end) == pytest.approx(
        -convention.year_fraction(end, start), abs=1e-12
    )


@pytest.mark.parametrize("convention", list(DayCount))
@given(start=dates, extra=st.integers(min_value=0, max_value=4000))
def test_year_fraction_is_monotone_in_the_end_date(
    convention: DayCount, start: date, extra: int
) -> None:
    assert convention.year_fraction(start, start + timedelta(days=extra)) >= -1e-12


@pytest.mark.parametrize("convention", list(DayCount))
@given(start=dates, mid_gap=st.integers(0, 900), end_gap=st.integers(0, 900))
def test_year_fraction_is_additive_across_adjacent_periods(
    convention: DayCount, start: date, mid_gap: int, end_gap: int
) -> None:
    mid = start + timedelta(days=mid_gap)
    end = mid + timedelta(days=end_gap)
    whole = convention.year_fraction(start, end)
    split = convention.year_fraction(start, mid) + convention.year_fraction(mid, end)
    # 30/360 is only additive up to its date clamping, which can shift a day.
    tolerance = 2 / 360 if convention is DayCount.THIRTY_360 else 1e-12
    assert whole == pytest.approx(split, abs=tolerance)


@given(start=dates, days=st.integers(1, 3000))
def test_act_360_is_literally_days_over_360(start: date, days: int) -> None:
    end = start + timedelta(days=days)
    assert DayCount.ACT_360.year_fraction(start, end) == pytest.approx(days / 360)


def test_str_returns_the_market_label() -> None:
    assert str(DayCount.ACT_365F) == "ACT/365F"
