"""Instruments, the market snapshot, and the greeks they report."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from aegis.conventions import DayCount, Frequency
from aegis.curves import DiscountCurve, bootstrap_treasury_curve, curve_from_store
from aegis.instruments import (
    Cash,
    EquityOption,
    EquityPosition,
    FixedRateBond,
    MarketSnapshot,
    MissingMarketDataError,
    total_value,
)
from aegis.marketdata import CboeProvider, FredProvider
from aegis.pricing import OptionRight, black_price
from aegis.vol import VolSurface, build_surface

TODAY = date(2024, 12, 31)
QUOTES = {
    "1M": 0.0440,
    "3M": 0.0437,
    "6M": 0.0424,
    "1Y": 0.0416,
    "2Y": 0.0425,
    "5Y": 0.0438,
    "10Y": 0.0458,
    "30Y": 0.0478,
}


@pytest.fixture(scope="module")
def curve() -> DiscountCurve:
    """The real 31 December 2024 Treasury curve."""
    return bootstrap_treasury_curve(TODAY, QUOTES)


@pytest.fixture
def market(curve: DiscountCurve) -> MarketSnapshot:
    """A snapshot with one equity, one curve and one FX pair."""
    return MarketSnapshot(
        value_date=TODAY,
        base_currency="USD",
        spots={"KO": 60.0, "SAP": 200.0},
        curves={"USD": curve, "EUR": DiscountCurve.flat(TODAY, 0.025, name="EUR")},
        fx_rates={"EURUSD": 1.0353},
        dividend_yields={"KO": 0.03},
    )


@pytest.fixture(scope="module")
def option_market(cboe: CboeProvider, fred: FredProvider) -> MarketSnapshot:
    """A snapshot carrying the calibrated KO surface."""
    chain = cboe.option_chain("KO")
    as_of = chain["value_date"][0]
    assert isinstance(as_of, date)
    quotes = fred.treasury_curve(as_of - timedelta(days=30), as_of)
    rates = curve_from_store(quotes, max(quotes["value_date"].to_list()))
    surface = build_surface(chain, rates)
    return MarketSnapshot(
        value_date=as_of,
        base_currency="USD",
        spots={"KO": surface.spot},
        curves={"USD": rates},
        surfaces={"KO": surface},
    )


def _surface(market: MarketSnapshot) -> VolSurface:
    return market.surface("KO")


# ------------------------------------------------------------------- snapshot


def test_the_snapshot_refuses_to_invent_missing_data(market: MarketSnapshot) -> None:
    # Defaulting a missing vol to 20% turns an outage into a plausible number,
    # which is far worse than a failed run.
    with pytest.raises(MissingMarketDataError, match="no spot price"):
        market.spot("NVDA")
    with pytest.raises(MissingMarketDataError, match="no discount curve"):
        market.curve("JPY")
    with pytest.raises(MissingMarketDataError, match="no volatility surface"):
        market.surface("KO")
    with pytest.raises(MissingMarketDataError, match="no FX rate"):
        market.fx_rate("JPY")


def test_fx_conversion_works_in_both_directions(market: MarketSnapshot) -> None:
    assert market.fx_rate("USD") == 1.0
    assert market.fx_rate("EUR") == pytest.approx(1.0353)

    inverted = MarketSnapshot(value_date=TODAY, base_currency="EUR", fx_rates={"EURUSD": 1.0353})
    assert inverted.fx_rate("USD") == pytest.approx(1 / 1.0353)


def test_scenarios_leave_the_original_snapshot_alone(market: MarketSnapshot) -> None:
    shocked = market.with_spots(KO=0.9).with_curves_shifted(100.0)

    assert market.spot("KO") == 60.0
    assert shocked.spot("KO") == pytest.approx(54.0)
    assert shocked.curve("USD").zero_rates[0] == pytest.approx(
        market.curve("USD").zero_rates[0] + 0.01
    )


def test_moving_the_value_date_leaves_the_market_data_alone(market: MarketSnapshot) -> None:
    tomorrow = market.with_value_date(TODAY + timedelta(days=1))
    assert tomorrow.value_date == TODAY + timedelta(days=1)
    assert tomorrow.spot("KO") == market.spot("KO")


def test_the_forward_follows_carry_when_there_is_no_surface(market: MarketSnapshot) -> None:
    expiry = date(2025, 12, 31)
    forward = market.forward("KO", expiry, "USD")
    rate = float(market.curve("USD").zero_rate(1.0)[0])
    expected = 60.0 * pow(2.718281828459045, (rate - 0.03) * 1.0)
    assert forward == pytest.approx(expected, rel=1e-6)
    assert forward > 60.0  # funding above the dividend yield


def test_the_forward_comes_from_the_option_market_when_a_surface_exists(
    option_market: MarketSnapshot,
) -> None:
    calibrated = _surface(option_market).slices[5]
    assert option_market.forward("KO", calibrated.expiry, "USD") == pytest.approx(
        calibrated.forward, rel=1e-9
    )


def test_the_forward_moves_with_spot(option_market: MarketSnapshot) -> None:
    # The bug this guards against is silent and total: a forward read straight
    # off the surface does not move when spot does, so every option delta in the
    # book comes out at zero and the risk report looks perfectly hedged.
    expiry = _surface(option_market).slices[5].expiry
    base = option_market.forward("KO", expiry, "USD")
    bumped = option_market.with_spots(KO=1.02).forward("KO", expiry, "USD")
    assert bumped == pytest.approx(base * 1.02, rel=1e-12)


# ----------------------------------------------------------------------- cash


def test_cash_is_worth_its_balance(market: MarketSnapshot) -> None:
    assert Cash(id="C1", currency="USD", amount=25_000.0).present_value(market) == 25_000.0


# --------------------------------------------------------------------- equity


def test_an_equity_position_is_quantity_times_spot(market: MarketSnapshot) -> None:
    position = EquityPosition(id="E1", currency="USD", symbol="KO", quantity=10_000)
    assert position.present_value(market) == 600_000.0
    assert position.risk_factors() == ("SPOT:KO",)


def test_equity_delta_is_the_cash_move_for_one_percent(market: MarketSnapshot) -> None:
    position = EquityPosition(id="E1", currency="USD", symbol="KO", quantity=10_000)
    delta = position.sensitivities(market)["delta"]
    moved = position.present_value(market.with_spots(KO=1.01)) - position.present_value(market)
    assert delta == pytest.approx(moved, rel=1e-12)


def test_a_short_equity_position_has_negative_value(market: MarketSnapshot) -> None:
    position = EquityPosition(id="E1", currency="USD", symbol="KO", quantity=-5_000)
    assert position.present_value(market) == -300_000.0


# ----------------------------------------------------------------------- bond


def _bond(coupon: float, years: int = 10, face: float = 1_000_000.0) -> FixedRateBond:
    return FixedRateBond(
        id=f"BOND-{years}Y",
        currency="USD",
        face=face,
        coupon=coupon,
        maturity=date(2024 + years, 11, 15),
        issue_date=date(2024, 11, 15),
    )


def test_a_par_coupon_prices_at_par(curve: DiscountCurve, market: MarketSnapshot) -> None:
    # The curve was bootstrapped from par yields, so a bond paying the 10-year par
    # rate has to come back at 100. This is the same round trip as phase 3, seen
    # from the instrument side.
    par = curve.par_rate(10.0)
    bond = FixedRateBond(
        id="PAR",
        currency="USD",
        face=1_000_000.0,
        coupon=par,
        maturity=date(2034, 12, 31),
        issue_date=TODAY,
        day_count=DayCount.ACT_365F,
    )
    assert bond.clean_price(market) == pytest.approx(1.0, abs=2e-3)


def test_a_zero_coupon_bond_is_worth_its_discount_factor(market: MarketSnapshot) -> None:
    maturity = date(2029, 12, 31)
    bond = FixedRateBond(
        id="ZCB",
        currency="USD",
        face=1_000_000.0,
        coupon=0.0,
        maturity=maturity,
        issue_date=TODAY,
        frequency=Frequency.ANNUAL,
    )
    expected = market.curve("USD").discount_to(maturity)
    assert bond.dirty_price(market) == pytest.approx(expected, rel=1e-9)


def test_dirty_price_is_clean_plus_accrued(market: MarketSnapshot) -> None:
    bond = _bond(0.04)
    assert bond.dirty_price(market) == pytest.approx(
        bond.clean_price(market) + bond.accrued_interest(TODAY), rel=1e-12
    )


def test_accrued_interest_is_zero_on_a_coupon_date() -> None:
    bond = _bond(0.04)
    coupon_date = bond.schedule()[1]
    assert bond.accrued_interest(coupon_date) == pytest.approx(0.0, abs=1e-12)


def test_accrued_interest_grows_through_the_period() -> None:
    bond = _bond(0.04)
    start, end = bond.schedule()[0], bond.schedule()[1]
    middle = start + (end - start) / 2
    accrued = bond.accrued_interest(middle)
    assert 0.0 < accrued < bond.coupon / 2
    assert accrued == pytest.approx(bond.coupon / 4, rel=0.05)


def test_a_bond_loses_value_when_rates_rise(market: MarketSnapshot) -> None:
    bond = _bond(0.04)
    assert bond.present_value(market.with_curves_shifted(50.0)) < bond.present_value(market)


def test_dv01_predicts_the_move_for_one_basis_point(market: MarketSnapshot) -> None:
    bond = _bond(0.04)
    dv01 = bond.sensitivities(market)["dv01"]
    actual = bond.present_value(market.with_curves_shifted(1.0)) - bond.present_value(market)
    assert dv01 == pytest.approx(-actual, rel=1e-3)
    assert dv01 > 0  # a long bond gains when yields fall


def test_duration_lengthens_with_maturity(market: MarketSnapshot) -> None:
    short = _bond(0.04, years=2).sensitivities(market)["duration"]
    long = _bond(0.04, years=10).sensitivities(market)["duration"]
    assert 1.5 < short < 2.0
    assert 7.5 < long < 8.5
    assert long > short


def test_convexity_is_positive_for_a_plain_bond(market: MarketSnapshot) -> None:
    assert _bond(0.04).sensitivities(market)["convexity"] > 0


def test_yield_to_maturity_reprices_the_bond(market: MarketSnapshot) -> None:
    bond = _bond(0.04)
    ytm = bond.yield_to_maturity(market)
    flows = bond.remaining_cash_flows(TODAY)
    frequency = float(bond.frequency.value)
    reprice = sum(
        amount / (1 + ytm / frequency) ** (frequency * bond.day_count.year_fraction(TODAY, day))
        for day, amount in flows
    )
    assert reprice == pytest.approx(bond.dirty_price(market), rel=1e-10)


def test_a_discount_bond_yields_more_than_its_coupon(market: MarketSnapshot) -> None:
    bond = _bond(0.04)  # below the ~4.6% ten-year yield, so it trades below par
    assert bond.clean_price(market) < 1.0
    assert bond.yield_to_maturity(market) > bond.coupon


def test_a_matured_bond_has_no_remaining_flows() -> None:
    bond = _bond(0.04, years=1)
    assert bond.remaining_cash_flows(date(2030, 1, 1)) == []
    assert bond.accrued_interest(date(2030, 1, 1)) == 0.0


# --------------------------------------------------------------------- option


def test_an_option_prices_to_the_black_formula(option_market: MarketSnapshot) -> None:
    calibrated = _surface(option_market).slices[8]
    option = EquityOption(
        id="O1",
        currency="USD",
        underlying="KO",
        strike=95.0,
        expiry=calibrated.expiry,
        right=OptionRight.CALL,
        quantity=10,
    )
    time = option.time_to_expiry(option_market)
    vol = option_market.implied_vol("KO", 95.0, calibrated.expiry)
    discount = float(option_market.curve("USD").discount_factor(time)[0])
    expected = float(black_price(calibrated.forward, 95.0, vol, time, discount)[0])
    assert option.present_value(option_market) == pytest.approx(1000 * expected, rel=1e-9)


def test_put_call_parity_holds_at_the_position_level(option_market: MarketSnapshot) -> None:
    expiry = _surface(option_market).slices[8].expiry
    call = EquityOption(
        id="C",
        currency="USD",
        underlying="KO",
        strike=95.0,
        expiry=expiry,
        right=OptionRight.CALL,
        quantity=10,
    )
    put = EquityOption(
        id="P",
        currency="USD",
        underlying="KO",
        strike=95.0,
        expiry=expiry,
        right=OptionRight.PUT,
        quantity=10,
    )

    time = call.time_to_expiry(option_market)
    discount = float(option_market.curve("USD").discount_factor(time)[0])
    forward = option_market.forward("KO", expiry, "USD")
    assert call.present_value(option_market) - put.present_value(option_market) == pytest.approx(
        1000 * discount * (forward - 95.0), rel=1e-9
    )


def test_delta_and_gamma_together_reproduce_a_one_percent_move(
    option_market: MarketSnapshot,
) -> None:
    # The greeks are quoted for a 1% move, so a second-order Taylor expansion
    # should land on the actual revaluation. This is the same arithmetic the P&L
    # explain does in phase 9, checked here in miniature.
    expiry = _surface(option_market).slices[8].expiry
    option = EquityOption(
        id="O1",
        currency="USD",
        underlying="KO",
        strike=95.0,
        expiry=expiry,
        right=OptionRight.CALL,
        quantity=10,
    )
    greeks = option.sensitivities(option_market)
    actual = option.present_value(option_market.with_spots(KO=1.01)) - option.present_value(
        option_market
    )
    assert greeks["delta"] + greeks["gamma"] == pytest.approx(actual, rel=1e-3)


def test_a_long_call_has_positive_delta_vega_and_negative_theta(
    option_market: MarketSnapshot,
) -> None:
    expiry = _surface(option_market).slices[8].expiry
    greeks = EquityOption(
        id="O1",
        currency="USD",
        underlying="KO",
        strike=95.0,
        expiry=expiry,
        right=OptionRight.CALL,
        quantity=10,
    ).sensitivities(option_market)

    assert greeks["delta"] > 0
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0
    assert greeks["theta"] < 0


def test_a_long_put_has_negative_delta_but_still_positive_vega(
    option_market: MarketSnapshot,
) -> None:
    expiry = _surface(option_market).slices[8].expiry
    greeks = EquityOption(
        id="O1",
        currency="USD",
        underlying="KO",
        strike=95.0,
        expiry=expiry,
        right=OptionRight.PUT,
        quantity=10,
    ).sensitivities(option_market)

    assert greeks["delta"] < 0
    assert greeks["vega"] > 0


def test_an_expired_option_is_worth_its_intrinsic_and_has_no_greeks(
    option_market: MarketSnapshot,
) -> None:
    expired = EquityOption(
        id="O1",
        currency="USD",
        underlying="KO",
        strike=50.0,
        expiry=option_market.value_date - timedelta(days=1),
        right=OptionRight.CALL,
        quantity=1,
    )
    intrinsic = 100 * (option_market.spot("KO") - 50.0)
    assert expired.present_value(option_market) == pytest.approx(intrinsic, rel=1e-6)
    assert all(value == 0.0 for value in expired.sensitivities(option_market).values())


def test_an_option_declares_every_factor_it_depends_on() -> None:
    option = EquityOption(
        id="O1",
        currency="USD",
        underlying="KO",
        strike=95.0,
        expiry=date(2027, 1, 15),
        right=OptionRight.CALL,
        quantity=10,
    )
    assert set(option.risk_factors()) == {"SPOT:KO", "VOL:KO", "RATE:USD"}


# ------------------------------------------------------------------ portfolio


def test_positions_aggregate_into_the_base_currency(market: MarketSnapshot) -> None:
    book = [
        Cash(id="C", currency="USD", amount=10_000.0),
        EquityPosition(id="E", currency="USD", symbol="KO", quantity=1_000),
        EquityPosition(id="F", currency="EUR", symbol="SAP", quantity=100),
    ]
    expected = 10_000.0 + 60_000.0 + 20_000.0 * 1.0353
    assert total_value(book, market) == pytest.approx(expected, rel=1e-12)
