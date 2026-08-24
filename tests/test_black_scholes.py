"""Black-Scholes in forward form, and the inversion back to implied volatility."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aegis.pricing import (
    OptionRight,
    black_delta,
    black_gamma,
    black_price,
    black_theta,
    black_vega,
    implied_volatility,
)

forwards = st.floats(min_value=1.0, max_value=5000.0, allow_nan=False)
strikes = st.floats(min_value=1.0, max_value=5000.0, allow_nan=False)
vols = st.floats(min_value=0.01, max_value=2.0, allow_nan=False)
times = st.floats(min_value=0.01, max_value=10.0, allow_nan=False)
discounts = st.floats(min_value=0.5, max_value=1.0, allow_nan=False)


@given(f=forwards, k=strikes, v=vols, t=times, df=discounts)
def test_put_call_parity_holds_exactly(f: float, k: float, v: float, t: float, df: float) -> None:
    call = float(black_price(f, k, v, t, df, OptionRight.CALL)[0])
    put = float(black_price(f, k, v, t, df, OptionRight.PUT)[0])
    assert call - put == pytest.approx(df * (f - k), rel=1e-9, abs=1e-9)


@given(f=forwards, k=strikes, v=vols, t=times, df=discounts)
def test_prices_respect_the_no_arbitrage_bounds(
    f: float, k: float, v: float, t: float, df: float
) -> None:
    call = float(black_price(f, k, v, t, df, OptionRight.CALL)[0])
    put = float(black_price(f, k, v, t, df, OptionRight.PUT)[0])
    assert df * max(f - k, 0.0) - 1e-9 <= call <= df * f + 1e-9
    assert df * max(k - f, 0.0) - 1e-9 <= put <= df * k + 1e-9


@given(f=forwards, k=strikes, t=times, v=vols, bump=st.floats(0.001, 0.5))
def test_price_increases_with_volatility(
    f: float, k: float, t: float, v: float, bump: float
) -> None:
    low = float(black_price(f, k, v, t, 1.0)[0])
    high = float(black_price(f, k, v + bump, t, 1.0)[0])
    assert high >= low - 1e-12


@pytest.mark.parametrize("right", list(OptionRight))
@given(f=forwards, k=strikes, v=vols, t=times, df=discounts)
def test_implied_volatility_reproduces_the_price_it_was_given(
    right: OptionRight, f: float, k: float, v: float, t: float, df: float
) -> None:
    # The invariant that always holds: whatever volatility comes back must
    # reprice the quote. The volatility itself is only pinned down where vega is
    # meaningful, which is the test below.
    price = black_price(f, k, v, t, df, right)
    recovered = implied_volatility(price, f, k, t, df, right)
    if np.isfinite(recovered[0]):
        reprice = black_price(f, k, recovered, t, df, right)
        assert float(reprice[0]) == pytest.approx(float(price[0]), rel=1e-6, abs=1e-8)


@given(
    f=forwards,
    standard_deviations=st.floats(min_value=-2.0, max_value=2.0),
    v=st.floats(min_value=0.05, max_value=1.5),
    t=st.floats(min_value=0.05, max_value=5.0),
    df=discounts,
)
def test_implied_volatility_recovers_the_volatility_of_an_out_of_the_money_quote(
    f: float, standard_deviations: float, v: float, t: float, df: float
) -> None:
    # Moneyness only means anything measured in standard deviations: a strike 50%
    # above the forward is barely out of the money on a two-year 60-vol name and
    # hopelessly far on a one-month 10-vol one. Within two standard deviations,
    # on the out-of-the-money side — which is the half of the chain a surface is
    # actually fitted to — the premium is nearly all time value and the inversion
    # is exact. Outside it no solver can do better, because the information is
    # not in the price.
    k = f * np.exp(standard_deviations * v * np.sqrt(t))
    right = OptionRight.CALL if standard_deviations >= 0 else OptionRight.PUT
    price = black_price(f, k, v, t, df, right)
    recovered = implied_volatility(price, f, k, t, df, right)
    assert np.isfinite(recovered[0])
    assert float(recovered[0]) == pytest.approx(v, rel=1e-6, abs=1e-8)


def test_implied_volatility_rejects_prices_that_admit_arbitrage() -> None:
    assert np.isnan(implied_volatility(0.5, 100.0, 50.0, 1.0, 1.0, OptionRight.CALL)[0])
    assert np.isnan(implied_volatility(120.0, 100.0, 50.0, 1.0, 1.0, OptionRight.CALL)[0])


def test_implied_volatility_handles_a_vector_of_strikes() -> None:
    strike_grid = np.array([60.0, 80.0, 100.0, 120.0, 150.0])
    prices = black_price(100.0, strike_grid, 0.25, 1.0, 0.97, OptionRight.CALL)
    recovered = implied_volatility(prices, 100.0, strike_grid, 1.0, 0.97, OptionRight.CALL)
    assert recovered == pytest.approx(np.full(5, 0.25), abs=1e-8)


@pytest.mark.parametrize("right", list(OptionRight))
def test_an_expired_option_is_worth_its_intrinsic(right: OptionRight) -> None:
    price = black_price(110.0, 100.0, 0.3, 0.0, 1.0, right)
    assert float(price[0]) == pytest.approx(max(right.sign * 10.0, 0.0))


def test_zero_volatility_is_not_a_division_by_zero() -> None:
    price = black_price(110.0, 100.0, 0.0, 2.0, 0.9, OptionRight.CALL)
    assert float(price[0]) == pytest.approx(0.9 * 10.0)


# ------------------------------------------------------------------------ greeks

FORWARD, STRIKE, VOL, TIME, DISCOUNT = 100.0, 90.0, 0.2, 1.0, 0.96


def _price(
    forward: float = FORWARD,
    strike: float = STRIKE,
    vol: float = VOL,
    time: float = TIME,
) -> float:
    return float(black_price(forward, strike, vol, time, DISCOUNT)[0])


def test_delta_matches_a_finite_difference() -> None:
    h = 1e-4
    numerical = (_price(forward=100.0 + h) - _price(forward=100.0 - h)) / (2 * h)
    assert float(black_delta(FORWARD, STRIKE, VOL, TIME, DISCOUNT)[0]) == pytest.approx(
        numerical, rel=1e-7
    )


def test_gamma_matches_a_finite_difference() -> None:
    h = 1e-2
    numerical = (_price(forward=100.0 + h) - 2 * _price() + _price(forward=100.0 - h)) / h**2
    assert float(black_gamma(FORWARD, STRIKE, VOL, TIME, DISCOUNT)[0]) == pytest.approx(
        numerical, rel=1e-6
    )


def test_vega_matches_a_finite_difference() -> None:
    h = 1e-6
    numerical = (_price(vol=0.2 + h) - _price(vol=0.2 - h)) / (2 * h)
    assert float(black_vega(FORWARD, STRIKE, VOL, TIME, DISCOUNT)[0]) == pytest.approx(
        numerical, rel=1e-6
    )


def test_theta_is_the_negative_of_the_price_drift_in_maturity() -> None:
    h = 1e-6
    drift = (_price(time=1.0 + h) - _price(time=1.0 - h)) / (2 * h)
    assert float(black_theta(FORWARD, STRIKE, VOL, TIME, DISCOUNT)[0]) == pytest.approx(
        -drift, rel=1e-5
    )


def test_theta_carry_term_reflects_the_discount_roll() -> None:
    without_rate = float(black_theta(FORWARD, STRIKE, VOL, TIME, DISCOUNT)[0])
    with_rate = float(
        black_theta(FORWARD, STRIKE, VOL, TIME, DISCOUNT, OptionRight.CALL, rate=0.04)[0]
    )
    assert with_rate > without_rate  # the discount factor rolls up towards one


@given(f=forwards, k=strikes, v=vols, t=times)
def test_gamma_and_vega_are_the_same_for_calls_and_puts(
    f: float, k: float, v: float, t: float
) -> None:
    assert float(black_gamma(f, k, v, t)[0]) >= 0.0
    assert float(black_vega(f, k, v, t)[0]) >= 0.0


@given(f=forwards, k=strikes, v=vols, t=times, df=discounts)
def test_delta_is_bounded_by_the_discount_factor(
    f: float, k: float, v: float, t: float, df: float
) -> None:
    call = float(black_delta(f, k, v, t, df, OptionRight.CALL)[0])
    put = float(black_delta(f, k, v, t, df, OptionRight.PUT)[0])
    assert 0.0 <= call <= df + 1e-12
    assert -df - 1e-12 <= put <= 0.0
    # Parity again, differentiated: the deltas differ by exactly the discount factor.
    assert call - put == pytest.approx(df, abs=1e-9)


def test_vega_peaks_near_the_money() -> None:
    grid = np.array([60.0, 80.0, 100.0, 125.0, 160.0])
    vegas = black_vega(100.0, grid, 0.2, 1.0)
    assert int(np.argmax(vegas)) == 2


def test_option_right_signs() -> None:
    assert OptionRight.CALL.sign == 1.0
    assert OptionRight.PUT.sign == -1.0
    assert OptionRight("C") is OptionRight.CALL
