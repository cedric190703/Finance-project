"""Portfolios, scenarios, and Value at Risk."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from aegis.curves import DiscountCurve, curve_from_store
from aegis.instruments import EquityPosition, MarketSnapshot
from aegis.marketdata import CboeProvider, FredProvider, MarketStore
from aegis.portfolio import Portfolio, PortfolioError
from aegis.risk import (
    BaselZone,
    FactorHistory,
    FactorMapping,
    ScenarioError,
    VarMethod,
    apply_shocks,
    backtest_series,
    basel_zone,
    build_factor_history,
    christoffersen_independence,
    conditional_coverage,
    contribution_report,
    expected_shortfall,
    factor_exposures,
    historical_pnl,
    historical_var,
    kupiec_pof,
    load_scenarios,
    monte_carlo_var,
    parametric_var,
    rolling_backtest,
    value_at_risk,
)
from aegis.risk.factors import ABSOLUTE, RELATIVE
from aegis.vol import build_surface

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def market(cboe: CboeProvider, fred: FredProvider, risk_store: MarketStore) -> MarketSnapshot:
    """Today's market: the real KO surface, Treasury curve, and index level."""
    chain = cboe.option_chain("KO")
    as_of = chain["value_date"][0]
    assert isinstance(as_of, date)
    quotes = fred.treasury_curve(as_of - timedelta(days=30), as_of)
    curve = curve_from_store(quotes, max(quotes["value_date"].to_list()))
    surface = build_surface(chain, curve)
    sp500 = risk_store.as_of("price_eod", where="symbol = 'SP500'").sort("value_date")
    return MarketSnapshot(
        value_date=as_of,
        base_currency="USD",
        spots={"KO": surface.spot, "SP500": float(sp500["close"][-1])},
        curves={"USD": curve, "EUR": DiscountCurve.flat(as_of, 0.022, name="EUR")},
        surfaces={"KO": surface},
        fx_rates={"EURUSD": 1.16},
    )


@pytest.fixture(scope="module")
def book() -> Portfolio:
    """The demo book shipped with the repository."""
    return Portfolio.from_yaml(REPO / "config" / "portfolio.yaml")


@pytest.fixture(scope="module")
def history(risk_store: MarketStore) -> FactorHistory:
    """Two years of real factor moves."""
    mappings = [
        FactorMapping("SPOT:SP500", "price_eod", "symbol = 'SP500'", "close", RELATIVE),
        FactorMapping(
            "SPOT:KO",
            "price_eod",
            "symbol = 'SP500'",
            "close",
            RELATIVE,
            note="proxied by the index",
        ),
        FactorMapping(
            "VOL:KO", "price_eod", "symbol = 'VIXCLS'", "close", RELATIVE, note="proxied by the VIX"
        ),
        FactorMapping("RATE:USD:2Y", "curve_point", "tenor = '2Y'", "rate", ABSOLUTE),
        FactorMapping("RATE:USD:10Y", "curve_point", "tenor = '10Y'", "rate", ABSOLUTE),
        FactorMapping("FX:EUR", "fx_rate", "pair = 'EURUSD'", "rate", RELATIVE),
    ]
    return build_factor_history(risk_store, mappings, date(2024, 8, 1), date(2026, 8, 24))


# ------------------------------------------------------------------ portfolio


def test_the_demo_book_loads(book: Portfolio) -> None:
    assert book.name == "demo-book"
    assert len(book) == 7
    assert {p.id for p in book.positions} >= {"EQ-KO", "OPT-KO-CALL", "UST-10Y", "BUND-5Y"}


def test_the_book_declares_every_factor_it_touches(book: Portfolio) -> None:
    factors = set(book.risk_factors())
    assert {"SPOT:KO", "SPOT:SP500", "VOL:KO", "RATE:USD", "RATE:EUR", "FX:EUR"} <= factors


def test_valuations_reconcile_to_the_book_total(book: Portfolio, market: MarketSnapshot) -> None:
    breakdown = book.valuations(market)
    assert breakdown.height == len(book)
    assert breakdown["base_value"].sum() == pytest.approx(book.value(market), rel=1e-12)


def test_the_euro_leg_is_converted(book: Portfolio, market: MarketSnapshot) -> None:
    bund = book.valuations(market).filter(pl.col("id") == "BUND-5Y")
    assert bund["currency"][0] == "EUR"
    assert bund["base_value"][0] == pytest.approx(bund["local_value"][0] * 1.16, rel=1e-12)


def test_the_book_greeks_tell_the_story_the_positions_imply(
    book: Portfolio, market: MarketSnapshot
) -> None:
    greeks = book.sensitivities(market)
    # Long equity, short a covered call: long delta, short gamma, short vega,
    # and collecting time decay.
    assert greeks["delta"] > 0
    assert greeks["gamma"] < 0
    assert greeks["vega"] < 0
    assert greeks["theta"] > 0
    assert greeks["dv01"] > 0


def test_ratio_greeks_are_not_summed_across_positions(
    book: Portfolio, market: MarketSnapshot
) -> None:
    # Adding one bond's duration to another's is arithmetic without meaning.
    assert "duration" not in book.sensitivities(market)
    assert "convexity" not in book.sensitivities(market)


def test_a_malformed_book_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    with pytest.raises(PortfolioError, match="positions"):
        Portfolio.from_yaml(bad)

    bad.write_text("positions:\n  - type: unicorn\n")
    with pytest.raises(PortfolioError, match="unknown instrument type"):
        Portfolio.from_yaml(bad)

    bad.write_text("positions:\n  - type: equity\n    symbol: KO\n")
    with pytest.raises(PortfolioError, match="missing field"):
        Portfolio.from_yaml(bad)


# ------------------------------------------------------------------ scenarios


def test_shocks_are_applied_in_log_space(market: MarketSnapshot) -> None:
    shocked = apply_shocks(market, {"SPOT:KO": math.log(0.9)})
    assert shocked.spot("KO") == pytest.approx(market.spot("KO") * 0.9, rel=1e-12)


def test_a_large_negative_shock_cannot_produce_a_negative_level(
    market: MarketSnapshot,
) -> None:
    # This is the bug log moves exist to prevent: a -35% daily volatility move
    # scaled to a ten-day horizon is -110% in simple terms, which would imply a
    # negative volatility.
    scaled = math.log(0.65) * math.sqrt(10)
    shocked = apply_shocks(market, {"VOL:KO": scaled})
    assert shocked.surface("KO").slices[0].atm_vol > 0


def test_rate_shocks_are_absolute(market: MarketSnapshot) -> None:
    shocked = apply_shocks(market, {"RATE:USD": 0.0100})
    assert shocked.curve("USD").zero_rates[0] == pytest.approx(
        market.curve("USD").zero_rates[0] + 0.01, abs=1e-12
    )


def test_a_key_rate_shock_is_local(market: MarketSnapshot) -> None:
    shocked = apply_shocks(market, {"RATE:USD:10Y": 0.0010})
    moved = np.flatnonzero(
        np.abs(shocked.curve("USD").zero_rates - market.curve("USD").zero_rates) > 1e-15
    )
    assert 0 < moved.size < market.curve("USD").times.size


def test_a_volatility_shock_scales_the_whole_surface(market: MarketSnapshot) -> None:
    shocked = apply_shocks(market, {"VOL:KO": math.log(1.25)})
    for before, after in zip(
        market.surface("KO").slices, shocked.surface("KO").slices, strict=True
    ):
        assert after.atm_vol == pytest.approx(before.atm_vol * 1.25, rel=1e-9)


def test_scaling_volatility_leaves_the_skew_alone(market: MarketSnapshot) -> None:
    slice_ = market.surface("KO").slices[8]
    scaled = market.surface("KO").scaled(1.3).slices[8]
    before = slice_.implied_vol(slice_.forward * 0.9)[0] / slice_.atm_vol
    after = scaled.implied_vol(scaled.forward * 0.9)[0] / scaled.atm_vol
    assert after == pytest.approx(before, rel=1e-9)


def test_a_scenario_leaves_the_original_market_alone(market: MarketSnapshot) -> None:
    spot_before = market.spot("KO")
    apply_shocks(market, {"SPOT:KO": -0.5, "RATE:USD": 0.02, "VOL:KO": 1.0})
    assert market.spot("KO") == spot_before


def test_unknown_factors_are_rejected(market: MarketSnapshot) -> None:
    with pytest.raises(ScenarioError, match="unrecognised risk factor"):
        apply_shocks(market, {"WEATHER:LONDON": 0.1})


def test_scenario_files_are_read_in_human_units() -> None:
    scenarios = load_scenarios(REPO / "config" / "scenarios" / "stress.yaml")
    lehman = next(s for s in scenarios if s.name == "lehman-2008")
    # Written as -0.25 in the file; stored as the log move the engine wants.
    assert lehman.shocks["SPOT:KO"] == pytest.approx(math.log(0.75), rel=1e-12)
    assert lehman.shocks["RATE:USD:2Y"] == pytest.approx(-0.0125)


def test_a_shock_that_wipes_out_the_level_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("scenarios:\n  - name: apocalypse\n    shocks:\n      SPOT:KO: -1.0\n")
    with pytest.raises(ScenarioError, match="would take the level to zero"):
        load_scenarios(path)


def test_the_stress_scenarios_all_hurt_this_book(book: Portfolio, market: MarketSnapshot) -> None:
    base = book.value(market)
    for scenario in load_scenarios(REPO / "config" / "scenarios" / "stress.yaml"):
        pnl = book.value(scenario.apply(market)) - base
        assert pnl < 0, scenario.name


def test_the_crisis_scenarios_are_the_worst_ones(book: Portfolio, market: MarketSnapshot) -> None:
    base = book.value(market)
    losses = {
        s.name: base - book.value(s.apply(market))
        for s in load_scenarios(REPO / "config" / "scenarios" / "stress.yaml")
    }
    assert losses["covid-march-2020"] > losses["lehman-2008"] > losses["rates-shock-2022"]
    assert losses["vol-spike-only"] < losses["rates-shock-2022"]


# ------------------------------------------------------------ factor history


def test_the_factor_history_covers_two_years_of_sessions(history: FactorHistory) -> None:
    assert len(history) > 450
    assert history.moves.shape == (len(history), len(history.factors))
    assert history.dates == tuple(sorted(history.dates))


def test_only_dates_where_every_factor_traded_are_kept(history: FactorHistory) -> None:
    # Filling a gap with the previous level would inject a zero move on a day the
    # others moved, quietly understating correlation exactly when it matters.
    assert not np.any(np.isnan(history.moves))


def test_proxied_factors_are_labelled_as_such(history: FactorHistory) -> None:
    assert "proxied" in history.notes["VOL:KO"]
    assert "proxied" in history.summary().filter(pl.col("factor") == "VOL:KO")["note"][0]


def test_the_summary_shows_why_a_normal_assumption_is_optimistic(
    history: FactorHistory,
) -> None:
    equities = history.summary().filter(pl.col("factor") == "SPOT:SP500")
    assert equities["excess_kurtosis"][0] > 1.0  # fat tails, in real data
    assert 0.005 < equities["daily_vol"][0] < 0.03


def test_the_covariance_matrix_is_symmetric_and_positive_semi_definite(
    history: FactorHistory,
) -> None:
    covariance = history.covariance()
    assert covariance == pytest.approx(covariance.T, abs=1e-18)
    assert np.all(np.linalg.eigvalsh(covariance) > -1e-18)


def test_correlations_are_bounded_and_unit_on_the_diagonal(history: FactorHistory) -> None:
    correlation = history.correlation()
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.all(np.abs(correlation) <= 1.0 + 1e-12)


def test_equity_and_volatility_are_strongly_negatively_correlated(
    history: FactorHistory,
) -> None:
    # The single most reliable fact in equity risk: the VIX rises when the index
    # falls. If this comes out positive, something is wired backwards.
    index = history.factors.index("SPOT:SP500")
    vol = history.factors.index("VOL:KO")
    assert history.correlation()[index, vol] < -0.5


def test_an_empty_mapping_list_is_rejected(risk_store: MarketStore) -> None:
    with pytest.raises(ValueError, match="no factor mappings"):
        build_factor_history(risk_store, [], date(2024, 1, 1), date(2026, 1, 1))


def test_a_factor_with_no_data_is_reported(risk_store: MarketStore) -> None:
    with pytest.raises(ValueError, match="no data"):
        build_factor_history(
            risk_store,
            [FactorMapping("SPOT:NVDA", "price_eod", "symbol = 'NVDA'", "close")],
            date(2024, 1, 1),
            date(2026, 1, 1),
        )


# ------------------------------------------------------------------- the tails


def test_var_is_reported_as_a_positive_loss() -> None:
    losses = np.linspace(-100.0, 100.0, 1001)
    assert value_at_risk(losses, 0.99) == pytest.approx(98.0, abs=0.5)


def test_var_of_a_profitable_distribution_is_zero() -> None:
    assert value_at_risk(np.linspace(10.0, 100.0, 500), 0.99) == 0.0


def test_expected_shortfall_is_never_below_the_var_at_the_same_level() -> None:
    rng = np.random.default_rng(0)
    pnl = rng.standard_t(df=4, size=5000) * 1000.0
    assert expected_shortfall(pnl, 0.99) >= value_at_risk(pnl, 0.99)


def test_expected_shortfall_sees_deeper_into_a_fat_tail() -> None:
    rng = np.random.default_rng(1)
    normal = rng.standard_normal(20_000)
    student = rng.standard_t(df=3, size=20_000)
    student = student / student.std() * normal.std()  # same volatility, fatter tail
    normal_gap = expected_shortfall(normal, 0.99) - value_at_risk(normal, 0.99)
    student_gap = expected_shortfall(student, 0.99) - value_at_risk(student, 0.99)
    assert student_gap > normal_gap


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
def test_impossible_confidence_levels_are_rejected(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must lie"):
        value_at_risk(np.zeros(100), confidence)


def test_a_tail_quantile_needs_enough_observations() -> None:
    with pytest.raises(ValueError, match="at least"):
        value_at_risk(np.zeros(5), 0.99)


# ------------------------------------------------------------------ the report


def test_historical_var_is_a_material_but_survivable_fraction_of_the_book(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    result = historical_var(book, market, history, 0.99)
    assert result.method is VarMethod.HISTORICAL
    assert 0.5 < result.var_percent < 10.0
    assert result.expected_shortfall >= result.var * 0.9
    assert result.observations == len(history)


def test_the_ten_day_number_scales_roughly_with_the_square_root_of_time(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    one_day = historical_var(book, market, history, 0.99).var
    ten_day = historical_var(book, market, history, 0.99, horizon_days=10).var
    assert ten_day == pytest.approx(one_day * math.sqrt(10), rel=0.15)


def test_historical_var_exceeds_the_normal_one_because_the_tails_are_fat(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    historical = historical_var(book, market, history, 0.99).var
    parametric = parametric_var(book, market, history, 0.99).var
    assert historical > parametric


def test_monte_carlo_lands_near_the_parametric_number_for_a_nearly_linear_book(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    # Both draw from the same covariance matrix; the gap between them is the
    # book's optionality, which at a one-day horizon is small.
    parametric = parametric_var(book, market, history, 0.99).var
    simulated = monte_carlo_var(book, market, history, 0.99, scenarios=4000, seed=3).var
    assert simulated == pytest.approx(parametric, rel=0.15)


def test_monte_carlo_var_is_reproducible(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    first = monte_carlo_var(book, market, history, 0.99, scenarios=1000, seed=42)
    second = monte_carlo_var(book, market, history, 0.99, scenarios=1000, seed=42)
    assert first.var == second.var


def test_cornish_fisher_stays_finite_on_a_wildly_fat_tailed_sample(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    # Unshrunk, the expansion on this history returns a "99% VaR" of 250k against
    # a historical 111k — not conservative, arithmetically meaningless.
    plain = parametric_var(book, market, history, 0.99).var
    adjusted = parametric_var(book, market, history, 0.99, cornish_fisher=True).var
    assert adjusted > plain
    assert adjusted < 3.0 * plain


def test_factor_exposures_recover_a_linear_position(market: MarketSnapshot) -> None:
    linear = Portfolio(
        name="linear",
        positions=(EquityPosition(id="E", currency="USD", symbol="KO", quantity=1_000),),
    )
    history = FactorHistory(
        factors=("SPOT:KO",),
        dates=(date(2026, 1, 2),),
        moves=np.array([[0.01]]),
        kinds={"SPOT:KO": RELATIVE},
        notes={},
    )
    exposure = factor_exposures(linear, market, history)[0]
    assert exposure == pytest.approx(1_000 * market.spot("KO"), rel=1e-2)


def test_component_var_adds_up_to_the_total(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    report = contribution_report(book, market, history)
    assert report["share"].sum() == pytest.approx(1.0, rel=1e-9)
    # The equity leg dominates, which is what a book this shaped should show.
    assert report["factor"][0].startswith("SPOT:")


def test_the_pnl_distribution_has_one_entry_per_historical_day(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    pnl = historical_pnl(book, market, history)
    assert pnl.size == len(history)
    assert pnl.min() < 0 < pnl.max()


def test_the_result_prints_the_way_a_risk_report_reads(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    text = str(historical_var(book, market, history, 0.99))
    assert "99%" in text
    assert "1-day" in text
    assert "ES" in text


# ------------------------------------------------------------- backtesting


def test_kupiec_accepts_a_plausible_number_of_exceptions() -> None:
    statistic, p_value = kupiec_pof(exceptions=3, observations=250, confidence=0.99)
    assert statistic >= 0.0
    assert p_value > 0.05


def test_kupiec_rejects_a_model_with_too_many_exceptions() -> None:
    statistic, p_value = kupiec_pof(exceptions=20, observations=250, confidence=0.99)
    assert statistic > 10.0
    assert p_value < 0.01


def test_christoffersen_detects_clustered_exceptions() -> None:
    clustered = np.zeros(250, dtype=bool)
    clustered[100:110] = True
    independent = np.zeros(250, dtype=bool)
    independent[[20, 80, 140, 200]] = True
    _, clustered_p = christoffersen_independence(clustered)
    _, independent_p = christoffersen_independence(independent)
    assert clustered_p < 0.05
    assert independent_p > 0.05


def test_conditional_coverage_combines_both_tests() -> None:
    breaches = np.zeros(250, dtype=bool)
    breaches[50:60] = True
    statistic, p_value = conditional_coverage(breaches, 0.99)
    assert statistic > 10.0
    assert p_value < 0.01


@pytest.mark.parametrize(
    ("exceptions", "expected_zone", "multiplier"),
    [
        (4, BaselZone.GREEN, 3.0),
        (5, BaselZone.AMBER, 3.4),
        (9, BaselZone.AMBER, 3.85),
        (10, BaselZone.RED, 4.0),
    ],
)
def test_basel_traffic_light_matches_the_250_day_table(
    exceptions: int, expected_zone: BaselZone, multiplier: float
) -> None:
    zone, actual_multiplier = basel_zone(exceptions)
    assert zone is expected_zone
    assert actual_multiplier == multiplier


def test_backtest_reports_breaches_and_worst_shortfall() -> None:
    pnl = np.full(250, 10.0)
    pnl[[10, 70, 180]] = [-120.0, -130.0, -180.0]
    result = backtest_series(pnl, np.full(250, 100.0))
    assert result.exceptions == 3
    assert result.zone is BaselZone.GREEN
    assert result.worst_breach == 80.0
    assert result.summary().filter(pl.col("test").str.contains("Kupiec"))["verdict"][0] == "pass"


def test_backtest_rejects_misaligned_series() -> None:
    with pytest.raises(ValueError, match="same length"):
        backtest_series(np.zeros(10), np.zeros(9))


def test_rolling_backtest_has_no_lookahead(
    book: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> None:
    result, series = rolling_backtest(book, market, history, window=250)
    assert result.observations == len(history) - 250
    assert series.height == result.observations
    assert series["value_date"][0] == history.dates[250]
    minimum_forecast = series["var_forecast"].min()
    assert isinstance(minimum_forecast, (int, float))
    assert minimum_forecast >= 0.0
