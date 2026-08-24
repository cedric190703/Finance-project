"""Tenors and coupon schedule generation."""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aegis.conventions import (
    NYSE,
    TARGET,
    WEEKENDS_ONLY,
    BusinessDayConvention,
    Frequency,
    StubConvention,
    Tenor,
    TenorUnit,
    generate_schedule,
)

dates = st.dates(min_value=date(2000, 1, 1), max_value=date(2040, 12, 31))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1D", Tenor(1, TenorUnit.DAY)),
        ("2w", Tenor(2, TenorUnit.WEEK)),
        (" 3M ", Tenor(3, TenorUnit.MONTH)),
        ("10Y", Tenor(10, TenorUnit.YEAR)),
        ("-1M", Tenor(-1, TenorUnit.MONTH)),
    ],
)
def test_tenor_parsing(text: str, expected: Tenor) -> None:
    assert Tenor.parse(text) == expected


@pytest.mark.parametrize("text", ["", "M3", "3X", "three months", "3.5Y"])
def test_tenor_parsing_rejects_nonsense(text: str) -> None:
    with pytest.raises(ValueError, match="not a tenor"):
        Tenor.parse(text)


def test_tenor_round_trips_through_its_string_form() -> None:
    assert Tenor.parse(str(Tenor(6, TenorUnit.MONTH))) == Tenor(6, TenorUnit.MONTH)


def test_month_addition_clamps_at_a_short_month_end() -> None:
    assert Tenor.parse("1M").add_to(date(2024, 1, 31)) == date(2024, 2, 29)
    assert Tenor.parse("1M").add_to(date(2023, 1, 31)) == date(2023, 2, 28)


def test_year_addition_clamps_on_a_leap_day() -> None:
    assert Tenor.parse("1Y").add_to(date(2024, 2, 29)) == date(2025, 2, 28)


def test_tenors_sort_by_length() -> None:
    tenors = sorted(
        [Tenor.parse("10Y"), Tenor.parse("3M"), Tenor.parse("1D")],
        key=lambda t: t.approximate_years,
    )
    assert [str(t) for t in tenors] == ["1D", "3M", "10Y"]


def test_semiannual_schedule_matches_a_hand_built_bond() -> None:
    schedule = generate_schedule(
        effective=date(2024, 2, 15),
        maturity=date(2026, 2, 15),
        frequency=Frequency.SEMI_ANNUAL,
        calendar=WEEKENDS_ONLY,
        convention=BusinessDayConvention.UNADJUSTED,
    )
    assert schedule == [
        date(2024, 2, 15),
        date(2024, 8, 15),
        date(2025, 2, 15),
        date(2025, 8, 15),
        date(2026, 2, 15),
    ]


def test_short_front_stub_lands_at_the_start() -> None:
    schedule = generate_schedule(
        effective=date(2024, 1, 10),
        maturity=date(2025, 2, 15),
        frequency=Frequency.SEMI_ANNUAL,
        calendar=WEEKENDS_ONLY,
        convention=BusinessDayConvention.UNADJUSTED,
        stub=StubConvention.SHORT_FRONT,
    )
    assert schedule[0] == date(2024, 1, 10)
    assert schedule[1] == date(2024, 2, 15)  # the stub closes on the anchor
    assert schedule[-1] == date(2025, 2, 15)


def test_short_back_stub_lands_at_the_end() -> None:
    schedule = generate_schedule(
        effective=date(2024, 2, 15),
        maturity=date(2025, 4, 1),
        frequency=Frequency.SEMI_ANNUAL,
        calendar=WEEKENDS_ONLY,
        convention=BusinessDayConvention.UNADJUSTED,
        stub=StubConvention.SHORT_BACK,
    )
    assert schedule[-2] == date(2025, 2, 15)
    assert schedule[-1] == date(2025, 4, 1)


def test_schedule_rejects_a_maturity_before_the_effective_date() -> None:
    with pytest.raises(ValueError, match="must fall after"):
        generate_schedule(
            effective=date(2024, 5, 1),
            maturity=date(2024, 1, 1),
            frequency=Frequency.ANNUAL,
            calendar=TARGET,
        )


def test_frequency_exposes_its_period() -> None:
    assert str(Frequency.QUARTERLY.tenor) == "3M"
    assert str(Frequency.MONTHLY.tenor) == "1M"


@given(
    effective=dates,
    years=st.integers(1, 30),
    frequency=st.sampled_from(list(Frequency)),
    stub=st.sampled_from(list(StubConvention)),
)
def test_generated_schedules_are_sorted_business_days(
    effective: date, years: int, frequency: Frequency, stub: StubConvention
) -> None:
    maturity = Tenor(years, TenorUnit.YEAR).add_to(effective)
    schedule = generate_schedule(effective, maturity, frequency, NYSE, stub=stub)

    assert schedule == sorted(schedule)
    assert len(set(schedule)) == len(schedule)
    assert all(NYSE.is_business_day(day) for day in schedule)
    assert len(schedule) >= 2


@given(effective=dates, years=st.integers(1, 20))
def test_schedule_length_matches_the_payment_count(effective: date, years: int) -> None:
    maturity = Tenor(years, TenorUnit.YEAR).add_to(effective)
    schedule = generate_schedule(
        effective,
        maturity,
        Frequency.SEMI_ANNUAL,
        WEEKENDS_ONLY,
        convention=BusinessDayConvention.UNADJUSTED,
    )
    assert len(schedule) == 2 * years + 1
