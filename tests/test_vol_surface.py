"""SVI slices and the surface built from a real option chain."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aegis.curves import DiscountCurve, curve_from_store
from aegis.marketdata import CboeProvider, FredProvider
from aegis.pricing import OptionRight, black_price
from aegis.vol import (
    SurfaceError,
    SviFitError,
    SviParameters,
    VolSurface,
    build_surface,
    fit_svi_slice,
    implied_forward,
)

# A well-behaved slice: modest level, negative skew, rounded floor.
TYPICAL = SviParameters(a=0.02, b=0.10, rho=-0.4, m=0.0, sigma=0.15)

log_moneyness = st.floats(min_value=-1.5, max_value=1.5)


# ----------------------------------------------------------------------- the form


@given(k=log_moneyness)
def test_total_variance_is_never_negative(k: float) -> None:
    assert float(TYPICAL.total_variance(k)[0]) >= 0.0


def test_the_wings_are_asymptotically_linear() -> None:
    # Roger Lee's moment formula caps the wing growth at linear in log-moneyness;
    # SVI is built to sit exactly on that bound. The approach is asymptotic, so
    # the far wing is where the slope should be read.
    far = np.array([2_000.0, 4_000.0])
    w = TYPICAL.total_variance(far)
    right_slope = (w[1] - w[0]) / (far[1] - far[0])
    assert right_slope == pytest.approx(TYPICAL.b * (1 + TYPICAL.rho), rel=1e-6)

    w_left = TYPICAL.total_variance(-far)
    left_slope = (w_left[1] - w_left[0]) / (far[1] - far[0])
    assert left_slope == pytest.approx(TYPICAL.b * (1 - TYPICAL.rho), rel=1e-6)


def test_negative_rho_produces_the_equity_skew() -> None:
    downside = float(TYPICAL.implied_vol(-0.2, 1.0)[0])
    upside = float(TYPICAL.implied_vol(0.2, 1.0)[0])
    assert downside > upside


def test_implied_vol_needs_a_positive_maturity() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TYPICAL.implied_vol(0.0, 0.0)


def test_a_well_behaved_slice_has_a_non_negative_density() -> None:
    assert TYPICAL.is_butterfly_free()
    assert float(TYPICAL.butterfly_g(np.array([0.0]))[0]) > 0.0


def test_an_extreme_slice_is_caught_by_the_butterfly_check() -> None:
    # A huge skew with almost no curvature bends the smile hard enough to imply a
    # negative density; that is exactly what g < 0 detects.
    pathological = SviParameters(a=0.001, b=0.9, rho=-0.99, m=0.0, sigma=0.01)
    assert not pathological.is_butterfly_free()


def test_derivatives_match_finite_differences() -> None:
    h = 1e-5
    k = np.array([0.15])
    w, first, second = TYPICAL.derivatives(k)
    numerical_first = (TYPICAL.total_variance(k + h) - TYPICAL.total_variance(k - h)) / (2 * h)
    numerical_second = (
        TYPICAL.total_variance(k + h)
        - 2 * TYPICAL.total_variance(k)
        + TYPICAL.total_variance(k - h)
    ) / h**2
    assert float(w[0]) == pytest.approx(float(TYPICAL.total_variance(k)[0]))
    assert float(first[0]) == pytest.approx(float(numerical_first[0]), rel=1e-6)
    assert float(second[0]) == pytest.approx(float(numerical_second[0]), rel=1e-4)


def test_as_tuple_round_trips() -> None:
    assert SviParameters(*TYPICAL.as_tuple()) == TYPICAL


# ---------------------------------------------------------------------- the fit


def test_calibration_recovers_parameters_from_clean_data() -> None:
    k = np.linspace(-0.6, 0.6, 25)
    w = TYPICAL.total_variance(k)
    fitted, rmse = fit_svi_slice(k, w)

    assert rmse < 1e-6
    assert fitted.total_variance(k) == pytest.approx(w, abs=1e-6)


def test_calibration_is_robust_to_quote_noise() -> None:
    rng = np.random.default_rng(20240315)
    k = np.linspace(-0.5, 0.5, 30)
    w = TYPICAL.total_variance(k) * (1.0 + rng.normal(0.0, 0.01, k.size))
    fitted, rmse = fit_svi_slice(k, w)

    # 1% noise on a total variance of ~0.04 is ~4e-4 of scatter, so a fit that
    # lands there has absorbed the noise rather than chased it.
    assert rmse < 1e-3
    assert float(fitted.implied_vol(0.0, 1.0)[0]) == pytest.approx(
        float(TYPICAL.implied_vol(0.0, 1.0)[0]), rel=0.02
    )


def test_calibration_refuses_too_few_quotes() -> None:
    with pytest.raises(SviFitError, match="at least"):
        fit_svi_slice(np.array([0.0, 0.1]), np.array([0.04, 0.041]))


def test_calibration_rejects_mismatched_inputs() -> None:
    with pytest.raises(SviFitError, match="same length"):
        fit_svi_slice(np.linspace(-0.5, 0.5, 10), np.zeros(9))


def test_the_calendar_floor_lifts_a_fit_that_would_otherwise_dip_below_it() -> None:
    k = np.linspace(-0.5, 0.5, 21)
    quotes = TYPICAL.total_variance(k)
    grid = np.linspace(-1.0, 1.0, 41)
    floor = TYPICAL.total_variance(grid) * 1.3  # a later expiry cannot be cheaper

    unconstrained, _ = fit_svi_slice(k, quotes, floor_grid=grid)
    constrained, _ = fit_svi_slice(k, quotes, floor_grid=grid, variance_floor=floor)

    assert np.any(unconstrained.total_variance(grid) < floor - 1e-6)
    below = floor - constrained.total_variance(grid)
    assert float(below.max()) < float((floor - unconstrained.total_variance(grid)).max())


# -------------------------------------------------------------- implied forward


def test_the_forward_is_recovered_from_parity_consistent_quotes() -> None:
    forward, discount, vol, time = 4321.0, 0.97, 0.2, 0.75
    strikes = np.array([3800.0, 4000.0, 4200.0, 4400.0, 4600.0])
    calls = black_price(forward, strikes, vol, time, discount, OptionRight.CALL)
    puts = black_price(forward, strikes, vol, time, discount, OptionRight.PUT)

    implied, implied_discount = implied_forward(strikes, calls, puts)
    assert implied == pytest.approx(forward, rel=1e-9)
    assert implied_discount == pytest.approx(discount, rel=1e-9)


def test_the_forward_can_be_implied_against_a_given_discount_factor() -> None:
    forward, discount = 4321.0, 0.97
    strikes = np.array([4000.0, 4200.0, 4400.0, 4600.0])
    calls = black_price(forward, strikes, 0.2, 0.75, discount, OptionRight.CALL)
    puts = black_price(forward, strikes, 0.2, 0.75, discount, OptionRight.PUT)

    implied, used = implied_forward(strikes, calls, puts, discount=discount)
    assert implied == pytest.approx(forward, rel=1e-9)
    assert used == discount


def test_implied_forward_needs_enough_strikes() -> None:
    with pytest.raises(SurfaceError, match="at least three"):
        implied_forward(np.array([100.0]), np.array([5.0]), np.array([3.0]))


def test_implied_forward_rejects_an_impossible_discount() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    with pytest.raises(SurfaceError, match="discount factor must lie"):
        implied_forward(strikes, strikes * 0, strikes * 0, discount=1.5)


def test_implied_forward_rejects_quotes_that_imply_a_negative_discount() -> None:
    # Parity slope of the wrong sign: C - P rising with strike.
    strikes = np.array([90.0, 100.0, 110.0])
    with pytest.raises(SurfaceError, match="impossible discount factor"):
        implied_forward(strikes, strikes * 0.1, np.zeros(3))


# ------------------------------------------------------- the surface, real data


@pytest.fixture(scope="module")
def ko_surface(cboe: CboeProvider, fred: FredProvider) -> VolSurface:
    """A surface calibrated to the committed Cboe chain for KO."""
    chain = cboe.option_chain("KO")
    as_of = chain["value_date"][0]
    assert isinstance(as_of, date)
    quotes = fred.treasury_curve(as_of - timedelta(days=30), as_of)
    sessions = sorted(quotes["value_date"].unique())
    curve = curve_from_store(quotes, sessions[-1])
    return build_surface(chain, curve)


def test_the_real_chain_calibrates_across_the_term_structure(ko_surface: VolSurface) -> None:
    assert ko_surface.underlying == "KO"
    assert len(ko_surface.slices) >= 10
    assert all(s.quote_count >= 6 for s in ko_surface.slices)
    assert [s.expiry for s in ko_surface.slices] == sorted(s.expiry for s in ko_surface.slices)


def test_atm_volatilities_are_plausible_for_a_defensive_large_cap(
    ko_surface: VolSurface,
) -> None:
    atm = [s.atm_vol for s in ko_surface.slices]
    assert all(0.08 < v < 0.45 for v in atm)
    # A staples name is not a meme stock: the term structure is gently upward.
    assert atm[-1] > atm[0]


def test_forwards_sit_near_spot_and_drift_with_carry(ko_surface: VolSurface) -> None:
    spot = ko_surface.spot
    front, back = ko_surface.slices[0], ko_surface.slices[-1]
    assert abs(front.forward - spot) / spot < 0.02
    # Rates above the dividend yield mean the forward rises with maturity.
    assert back.forward > front.forward


def test_the_regressed_discount_is_kept_as_a_diagnostic(ko_surface: VolSurface) -> None:
    # It is exactly the number the code refuses to trust for short expiries, so
    # it should be present, and visibly worse than the curve at the front.
    assert all(np.isfinite(s.regressed_discount) for s in ko_surface.slices)
    front = ko_surface.slices[0]
    assert front.discount < 1.0


def test_slices_record_the_range_they_were_quoted_over(ko_surface: VolSurface) -> None:
    # The range matters: outside it a slice is extrapolating, and the calendar
    # condition is only enforced and only checked where two expiries overlap.
    for slice_ in ko_surface.slices:
        low, high = slice_.quoted_range
        assert low < 0.0 < high
        assert high - low < 2 * 1.5


def test_the_smile_shows_the_equity_skew(ko_surface: VolSurface) -> None:
    slice_ = ko_surface.slices[len(ko_surface.slices) // 2]
    downside = float(slice_.implied_vol(slice_.forward * 0.85)[0])
    upside = float(slice_.implied_vol(slice_.forward * 1.15)[0])
    assert downside > upside


def test_the_calibrated_surface_is_free_of_arbitrage(ko_surface: VolSurface) -> None:
    report = ko_surface.check_arbitrage(tolerance=1e-6)
    assert report.is_clean, str(report)
    assert report.checked_slices == len(ko_surface.slices)


def test_slices_fit_the_market_closely(ko_surface: VolSurface) -> None:
    # RMSE is in total variance; 1e-2 there is a wide miss, 1e-3 is a good fit.
    assert all(s.rmse_total_variance < 1e-2 for s in ko_surface.slices)


def test_the_surface_interpolates_between_calibrated_expiries(
    ko_surface: VolSurface,
) -> None:
    early, late = ko_surface.slices[2], ko_surface.slices[3]
    midpoint = early.expiry + (late.expiry - early.expiry) / 2
    strike = ko_surface.spot

    interpolated = ko_surface.implied_vol(strike, midpoint)
    bracketing = sorted([float(early.implied_vol(strike)[0]), float(late.implied_vol(strike)[0])])
    assert bracketing[0] - 0.02 <= interpolated <= bracketing[1] + 0.02


def test_interpolated_total_variance_never_decreases_with_maturity(
    ko_surface: VolSurface,
) -> None:
    strike = ko_surface.spot
    horizon = [ko_surface.reference_date + timedelta(days=d) for d in range(30, 700, 15)]
    total_variance = [
        ko_surface.implied_vol(strike, day) ** 2 * ((day - ko_surface.reference_date).days / 365.0)
        for day in horizon
    ]
    assert np.all(np.diff(total_variance) > -1e-8)


def test_looking_up_a_calibrated_expiry_returns_that_slice(ko_surface: VolSurface) -> None:
    target = ko_surface.slices[1]
    assert ko_surface.slice_for(target.expiry) is target
    with pytest.raises(KeyError, match="no calibrated slice"):
        ko_surface.slice_for(date(1999, 1, 1))


def test_the_surface_refuses_an_expiry_in_the_past(ko_surface: VolSurface) -> None:
    with pytest.raises(SurfaceError, match="not in the future"):
        ko_surface.implied_vol(90.0, ko_surface.reference_date)


def test_the_surface_frame_reports_per_slice_diagnostics(ko_surface: VolSurface) -> None:
    frame = ko_surface.to_frame()
    assert frame.height == len(ko_surface.slices)
    assert set(frame.columns) >= {"expiry", "forward", "atm_vol", "quotes", "butterfly_free"}
    assert frame["butterfly_free"].all()


def test_building_from_an_empty_chain_is_an_error() -> None:
    import polars as pl

    with pytest.raises(SurfaceError, match="empty option chain"):
        build_surface(pl.DataFrame(), DiscountCurve.flat(date(2024, 1, 2), 0.04))


def test_a_chain_with_no_tradeable_quotes_is_an_error(cboe: CboeProvider) -> None:
    chain = cboe.option_chain("KO").with_columns(bid=0.0)
    with pytest.raises(SurfaceError, match="no quotes survived"):
        build_surface(chain, DiscountCurve.flat(date(2026, 8, 24), 0.04))
