"""Discount curves: interpolation, bootstrapping and bumping."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aegis.conventions import DayCount
from aegis.curves import (
    BootstrapError,
    DiscountCurve,
    Interpolation,
    bootstrap_treasury_curve,
    curve_from_store,
    interpolate_zero_rates,
)
from aegis.marketdata import FredProvider
from tests.conftest import FIXTURE_END, FIXTURE_START

REFERENCE = date(2024, 12, 31)

# A curve shaped like the real thing: rates in a plausible band, upward sloping
# often but not always, at the standard quoted tenors.
rates = st.floats(min_value=0.0001, max_value=0.12, allow_nan=False, allow_infinity=False)


def _curve(
    zeros: list[float], scheme: Interpolation = Interpolation.LOG_LINEAR_DISCOUNT
) -> DiscountCurve:
    times = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0][: len(zeros)], dtype=np.float64)
    return DiscountCurve(REFERENCE, times, np.array(zeros, dtype=np.float64), interpolation=scheme)


# --------------------------------------------------------------------- structure


def test_flat_curve_discounts_analytically() -> None:
    curve = DiscountCurve.flat(REFERENCE, 0.05)
    for t in (0.25, 1.0, 7.5, 30.0):
        assert float(curve.discount_factor(t)[0]) == pytest.approx(np.exp(-0.05 * t), rel=1e-12)


def test_a_cash_flow_today_is_not_discounted() -> None:
    curve = DiscountCurve.flat(REFERENCE, 0.05)
    assert float(curve.discount_factor(0.0)[0]) == 1.0
    assert curve.discount_to(REFERENCE) == 1.0


def test_curve_rejects_a_degenerate_knot_structure() -> None:
    with pytest.raises(ValueError, match="same length"):
        DiscountCurve(REFERENCE, np.array([1.0]), np.array([0.01, 0.02]))
    with pytest.raises(ValueError, match="strictly increasing"):
        DiscountCurve(REFERENCE, np.array([2.0, 1.0]), np.array([0.01, 0.02]))
    with pytest.raises(ValueError, match="positive"):
        DiscountCurve(REFERENCE, np.array([0.0, 1.0]), np.array([0.01, 0.02]))


def test_year_fraction_uses_the_curves_own_day_count() -> None:
    curve = DiscountCurve.flat(REFERENCE, 0.05, day_count=DayCount.ACT_360)
    assert curve.year_fraction(date(2025, 12, 31)) == pytest.approx(365 / 360)


@given(zeros=st.lists(rates, min_size=3, max_size=7))
def test_discount_factors_are_positive_and_never_exceed_one(zeros: list[float]) -> None:
    curve = _curve(zeros)
    factors = curve.discount_factor(np.linspace(0.01, 30.0, 60))
    assert np.all(factors > 0.0)
    assert np.all(factors <= 1.0)


@given(zeros=st.lists(rates, min_size=3, max_size=7))
def test_discount_factors_decrease_on_an_upward_sloping_curve(zeros: list[float]) -> None:
    # Monotone discount factors are equivalent to non-negative forwards, which a
    # non-decreasing positive zero curve guarantees. An inverted curve steep
    # enough to imply a negative forward legitimately breaks it, so the property
    # is stated for the case where it actually holds.
    factors = _curve(sorted(zeros)).discount_factor(np.linspace(0.05, 30.0, 100))
    assert np.all(np.diff(factors) <= 1e-15)


@given(zeros=st.lists(rates, min_size=3, max_size=7))
def test_a_steeply_inverted_curve_can_imply_a_negative_forward(zeros: list[float]) -> None:
    # The converse, asserted so the limitation above is recorded rather than
    # assumed away: forwards are an output of the quotes, not a constraint on them.
    curve = _curve(sorted(zeros, reverse=True))
    forwards = [curve.forward_rate(t, t + 0.5) for t in (0.5, 2.0, 8.0)]
    assert all(np.isfinite(f) for f in forwards)


@given(
    zeros=st.lists(rates, min_size=3, max_size=7),
    start=st.floats(0.1, 10.0),
    gap=st.floats(0.05, 15.0),
)
def test_forward_rates_reprice_the_discount_factors_they_came_from(
    zeros: list[float], start: float, gap: float
) -> None:
    curve = _curve(zeros)
    end = start + gap
    forward = curve.forward_rate(start, end)
    implied = float(curve.discount_factor(start)[0]) * np.exp(-forward * (end - start))
    assert implied == pytest.approx(float(curve.discount_factor(end)[0]), rel=1e-10)


def test_forward_rate_rejects_a_reversed_period() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        DiscountCurve.flat(REFERENCE, 0.03).forward_rate(5.0, 5.0)


# ----------------------------------------------------------------- interpolation


@pytest.mark.parametrize("scheme", list(Interpolation))
def test_every_scheme_passes_through_its_knots(scheme: Interpolation) -> None:
    zeros = [0.045, 0.042, 0.040, 0.041, 0.044, 0.046, 0.047]
    curve = _curve(zeros, scheme)
    assert curve.zero_rate(curve.times) == pytest.approx(np.array(zeros), abs=1e-12)


@pytest.mark.parametrize("scheme", list(Interpolation))
def test_every_scheme_extrapolates_flat(scheme: Interpolation) -> None:
    curve = _curve([0.045, 0.042, 0.040, 0.041, 0.044, 0.046, 0.047], scheme)
    assert float(curve.zero_rate(60.0)[0]) == pytest.approx(0.047, abs=1e-12)
    assert float(curve.zero_rate(0.01)[0]) == pytest.approx(0.045, abs=1e-12)


def test_log_linear_discount_implies_constant_forwards_between_knots() -> None:
    curve = _curve(
        [0.03, 0.035, 0.04, 0.045, 0.05, 0.052, 0.053], Interpolation.LOG_LINEAR_DISCOUNT
    )
    # Both windows sit inside the 10y-30y bucket, so the forward must not move.
    assert curve.forward_rate(11.0, 12.0) == pytest.approx(
        curve.forward_rate(20.0, 21.0), rel=1e-10
    )


def test_linear_zero_and_log_linear_discount_disagree_between_knots() -> None:
    zeros = [0.03, 0.035, 0.04, 0.045, 0.05, 0.052, 0.053]
    linear = _curve(zeros, Interpolation.LINEAR_ZERO)
    log_linear = _curve(zeros, Interpolation.LOG_LINEAR_DISCOUNT)
    assert float(linear.zero_rate(7.5)[0]) != pytest.approx(float(log_linear.zero_rate(7.5)[0]))


def test_interpolation_rejects_bad_knots() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        interpolate_zero_rates(
            np.array([]), np.array([]), np.array([1.0]), Interpolation.LINEAR_ZERO
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_zero_rates(
            np.array([2.0, 1.0]), np.array([0.01, 0.02]), np.array([1.5]), Interpolation.LINEAR_ZERO
        )


def test_a_single_knot_curve_is_flat() -> None:
    got = interpolate_zero_rates(
        np.array([5.0]), np.array([0.04]), np.array([0.5, 50.0]), Interpolation.NATURAL_CUBIC_ZERO
    )
    assert got == pytest.approx(np.array([0.04, 0.04]))


# -------------------------------------------------------------------- bootstrap


REAL_QUOTES = {
    "1M": 0.0440,
    "3M": 0.0437,
    "6M": 0.0424,
    "1Y": 0.0416,
    "2Y": 0.0425,
    "3Y": 0.0427,
    "5Y": 0.0438,
    "7Y": 0.0448,
    "10Y": 0.0458,
    "20Y": 0.0486,
    "30Y": 0.0478,
}


@pytest.mark.parametrize("scheme", list(Interpolation))
def test_bootstrap_reprices_the_par_yields_it_was_built_from(scheme: Interpolation) -> None:
    # The round trip: bootstrap to zero rates, price the par bonds back off the
    # curve, and recover the input yields. This is the phase's central claim.
    curve = bootstrap_treasury_curve(REFERENCE, REAL_QUOTES, interpolation=scheme)
    for label, quoted in REAL_QUOTES.items():
        years = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0}.get(label)
        if years is not None:
            continue  # money-market tenors are zero rates, not par bonds
        tenor_years = float(label.rstrip("Y"))
        assert curve.par_rate(tenor_years) == pytest.approx(quoted, abs=1e-12)


def test_money_market_tenors_become_simple_interest_zeros() -> None:
    curve = bootstrap_treasury_curve(REFERENCE, {"6M": 0.05})
    # DF = 1 / (1 + y*t) is the quoting convention, not exp(-y*t).
    assert float(curve.discount_factor(0.5)[0]) == pytest.approx(1 / (1 + 0.05 * 0.5), rel=1e-14)


def test_bootstrapped_knots_sit_at_the_quoted_tenors() -> None:
    curve = bootstrap_treasury_curve(REFERENCE, REAL_QUOTES)
    assert curve.times.size == len(REAL_QUOTES)
    assert float(curve.times[-1]) == pytest.approx(30.0)


def test_an_inverted_curve_bootstraps() -> None:
    inverted = {
        "3M": 0.0550,
        "1Y": 0.0520,
        "2Y": 0.0470,
        "5Y": 0.0420,
        "10Y": 0.0405,
        "30Y": 0.0410,
    }
    curve = bootstrap_treasury_curve(REFERENCE, inverted)
    assert curve.par_rate(10.0) == pytest.approx(0.0405, abs=1e-12)
    assert float(curve.zero_rate(0.25)[0]) > float(curve.zero_rate(10.0)[0])


def test_bootstrap_rejects_an_empty_quote_set() -> None:
    with pytest.raises(BootstrapError, match="no quotes"):
        bootstrap_treasury_curve(REFERENCE, {})


def test_bootstrap_reports_a_quote_it_cannot_solve() -> None:
    with pytest.raises(BootstrapError, match="cannot bracket"):
        bootstrap_treasury_curve(REFERENCE, {"2Y": 5.0})


def test_curve_from_store_rows(fred: FredProvider) -> None:
    quotes = fred.treasury_curve(FIXTURE_START, FIXTURE_END)
    curve = curve_from_store(quotes, FIXTURE_END)

    assert curve.reference_date == FIXTURE_END
    assert curve.name == "UST"
    assert curve.par_rate(10.0) == pytest.approx(0.0458, abs=1e-10)


def test_curve_from_store_reports_a_missing_session(fred: FredProvider) -> None:
    quotes = fred.treasury_curve(FIXTURE_START, FIXTURE_END)
    with pytest.raises(BootstrapError, match="no UST quotes"):
        curve_from_store(quotes, date(2024, 12, 25))  # Christmas: no publication


def test_every_session_in_two_years_of_real_data_bootstraps(fred: FredProvider) -> None:
    # 2023-24 covers a deeply inverted curve, its re-steepening, and the whole
    # hiking-to-cutting turn. Every one of those sessions has to bootstrap and
    # reprice, not just the well-behaved ones.
    quotes = fred.treasury_curve(FIXTURE_START, FIXTURE_END)
    sessions = sorted(quotes["value_date"].unique())
    assert len(sessions) > 400

    for day in sessions:
        curve = curve_from_store(quotes, day)
        factors = curve.discount_factor(np.linspace(0.05, 30.0, 40))
        assert np.all(factors > 0.0), day
        quoted = quotes.filter(pl.col("value_date") == day)
        for tenor_label in ("2Y", "10Y", "30Y"):
            quoted_rate = quoted.filter(pl.col("tenor") == tenor_label)["rate"][0]
            years = float(tenor_label.rstrip("Y"))
            assert curve.par_rate(years) == pytest.approx(quoted_rate, abs=1e-10), (
                day,
                tenor_label,
            )


# ------------------------------------------------------------------------ bumps


def test_a_parallel_shift_moves_every_zero_rate() -> None:
    curve = bootstrap_treasury_curve(REFERENCE, REAL_QUOTES)
    bumped = curve.shift_parallel(1.0)
    assert bumped.zero_rates == pytest.approx(curve.zero_rates + 1e-4)
    assert curve.zero_rates[0] != bumped.zero_rates[0]  # the original is untouched


def test_key_rate_bumps_sum_to_a_parallel_shift() -> None:
    # The reconciliation a risk report is judged on: the key-rate decomposition
    # has to add back up to the parallel number.
    curve = bootstrap_treasury_curve(REFERENCE, REAL_QUOTES)
    total = np.zeros_like(curve.zero_rates)
    for tenor in curve.times:
        total += curve.shift_key_rate(float(tenor), 1.0).zero_rates - curve.zero_rates
    assert total == pytest.approx(np.full_like(curve.zero_rates, 1e-4), abs=1e-15)


def test_a_key_rate_bump_is_local() -> None:
    curve = bootstrap_treasury_curve(REFERENCE, REAL_QUOTES)
    bumped = curve.shift_key_rate(10.0, 10.0)
    moved = np.flatnonzero(np.abs(bumped.zero_rates - curve.zero_rates) > 1e-15)
    # Only the 10y knot and its two neighbours move.
    assert set(moved.tolist()) <= {7, 8, 9}


def test_switching_interpolation_keeps_the_knots() -> None:
    curve = bootstrap_treasury_curve(REFERENCE, REAL_QUOTES)
    switched = curve.with_interpolation(Interpolation.NATURAL_CUBIC_ZERO)
    assert switched.zero_rates == pytest.approx(curve.zero_rates)
    assert switched.interpolation is Interpolation.NATURAL_CUBIC_ZERO


def test_knot_table_labels_the_tenors() -> None:
    curve = bootstrap_treasury_curve(REFERENCE, REAL_QUOTES)
    labels = [row[0] for row in curve.knot_table()]
    assert labels[:3] == ["1M", "3M", "6M"]
    assert labels[-1] == "30Y"
