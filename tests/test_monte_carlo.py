"""The compiled Monte Carlo core, seen from Python."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from aegis import _core
from aegis.pricing import (
    McPrice,
    OptionRight,
    black_price,
    monte_carlo_european,
    monte_carlo_european_batch,
)
from aegis.pricing.reference import numpy_european, python_loop_european

PATHS = 1 << 18


def _exact(
    forward: float, strike: float, vol: float, time: float, discount: float, right: OptionRight
) -> float:
    return float(black_price(forward, strike, vol, time, discount, right)[0])


# ------------------------------------------------------------- the kernels agree


def test_the_two_black_implementations_agree_across_the_ffi_boundary() -> None:
    # The same formula, written twice in two languages, is only reassuring if
    # somebody checks the two answers actually match.
    for strike in (60.0, 90.0, 100.0, 130.0, 200.0):
        rust = _core.black_price(100.0, strike, 0.22, 1.4, 0.95, "C")
        python = _exact(100.0, strike, 0.22, 1.4, 0.95, OptionRight.CALL)
        assert rust == pytest.approx(python, rel=1e-12)


@given(x=st.floats(min_value=-8.0, max_value=8.0))
def test_the_rust_normal_cdf_matches_scipy(x: float) -> None:
    assert _core.normal_cdf(x) == pytest.approx(float(norm.cdf(x)), abs=1e-13)


@given(p=st.floats(min_value=1e-9, max_value=1 - 1e-9))
def test_the_rust_inverse_normal_matches_scipy(p: float) -> None:
    assert _core.normal_inverse_cdf(p) == pytest.approx(float(norm.ppf(p)), rel=1e-9, abs=1e-9)


def test_the_sobol_sequence_is_the_van_der_corput_sequence() -> None:
    points = _core.sobol_points(7, 0)
    assert points == pytest.approx(np.array([0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125]))


def test_sobol_points_fill_every_bucket_evenly() -> None:
    points = _core.sobol_points(1024, 12345)
    counts = np.bincount((points * 32).astype(int), minlength=32)
    assert np.all(counts == 32)


# ------------------------------------------------------------------ convergence


@pytest.mark.parametrize("right", list(OptionRight))
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_simulation_agrees_with_the_closed_form(right: OptionRight, strike: float) -> None:
    result = monte_carlo_european(100.0, strike, 0.25, 1.0, 0.96, right, PATHS, seed=7)
    exact = _exact(100.0, strike, 0.25, 1.0, 0.96, right)
    assert result.agrees_with(exact), f"{result} against {exact:.6f}"


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    strike=st.floats(min_value=50.0, max_value=200.0),
    vol=st.floats(min_value=0.05, max_value=0.9),
    time=st.floats(min_value=0.1, max_value=3.0),
    right=st.sampled_from(list(OptionRight)),
)
def test_simulation_agrees_with_the_closed_form_across_the_grid(
    strike: float, vol: float, time: float, right: OptionRight
) -> None:
    result = monte_carlo_european(100.0, strike, vol, time, 0.97, right, 1 << 16, seed=3)
    exact = _exact(100.0, strike, vol, time, 0.97, right)
    # One tenth of a basis point on a notional of 100 is the materiality bar; in
    # the deep wings the option itself is worth less than that, and plain Monte
    # Carlo cannot price it to a useful relative accuracy at any path count.
    assert result.agrees_with(exact, sigmas=5.0, absolute=1e-3), f"{result} vs {exact:.6f}"


def test_the_error_bar_shrinks_with_the_path_count() -> None:
    # Stated for pseudo-random sampling, where the 1/sqrt(n) rate applies to every
    # step. The randomised quasi-random estimator gets its error bar from the
    # spread between independently shifted replicas, so at small path counts
    # there are only a handful of replicas and the estimate is itself noisy —
    # monotone convergence there is a claim about the mean, not about one run.
    errors = [
        monte_carlo_european(
            100.0, 100.0, 0.25, 1.0, 0.96, paths=paths, seed=5, sampler="pseudo"
        ).standard_error
        for paths in (1 << 14, 1 << 16, 1 << 18, 1 << 20)
    ]
    assert errors == sorted(errors, reverse=True)

    # Four times the paths should roughly halve the error.
    assert errors[0] / errors[-1] == pytest.approx(8.0, rel=0.25)


def test_quasi_random_sampling_is_far_more_accurate_than_pseudo_random() -> None:
    pseudo = monte_carlo_european(
        100.0, 100.0, 0.25, 1.0, 0.96, paths=PATHS, seed=5, sampler="pseudo"
    )
    quasi = monte_carlo_european(
        100.0, 100.0, 0.25, 1.0, 0.96, paths=PATHS, seed=5, sampler="quasi"
    )
    assert quasi.standard_error * 10 < pseudo.standard_error


# ---------------------------------------------------------------- reproducibility


def test_the_same_seed_reproduces_the_price_exactly() -> None:
    first = monte_carlo_european(100.0, 95.0, 0.3, 2.0, 0.9, paths=PATHS, seed=1234)
    second = monte_carlo_european(100.0, 95.0, 0.3, 2.0, 0.9, paths=PATHS, seed=1234)
    assert first == second


def test_different_seeds_produce_different_draws() -> None:
    first = monte_carlo_european(100.0, 95.0, 0.3, 2.0, 0.9, paths=PATHS, seed=1)
    second = monte_carlo_european(100.0, 95.0, 0.3, 2.0, 0.9, paths=PATHS, seed=2)
    assert first.price != second.price


def test_an_interval_from_few_replicas_is_wider_than_the_normal_one() -> None:
    # Eight replicas earn a Student-t interval, not a normal one.
    few = McPrice(price=10.0, standard_error=0.01, paths=1 << 16, replicas=8)
    many = McPrice(price=10.0, standard_error=0.01, paths=1 << 16, replicas=10_000)
    assert few.confidence_95 > many.confidence_95 * 1.15
    assert many.confidence_95 == pytest.approx(0.0196, abs=1e-4)


def test_a_quasi_random_run_reports_one_replica_per_chunk(request: pytest.FixtureRequest) -> None:
    result = monte_carlo_european(
        100.0, 100.0, 0.25, 1.0, 0.96, paths=8 * _core.CHUNK_PATHS, sampler="quasi"
    )
    assert result.replicas == 8


def test_the_path_count_is_rounded_up_to_whole_chunks() -> None:
    result = monte_carlo_european(100.0, 100.0, 0.2, 1.0, paths=1)
    assert result.paths == _core.CHUNK_PATHS


# ------------------------------------------------------------------------ books


def test_a_book_prices_in_one_call() -> None:
    strikes = np.linspace(70.0, 130.0, 25)
    n = strikes.size
    prices, errors = monte_carlo_european_batch(
        np.full(n, 100.0),
        strikes,
        np.full(n, 0.25),
        np.full(n, 1.0),
        np.full(n, 0.96),
        ["C"] * n,
        paths=1 << 18,
        seed=11,
    )

    assert prices.shape == (n,)
    assert np.all(errors > 0)
    for strike, price, error in zip(strikes, prices, errors, strict=True):
        exact = _exact(100.0, float(strike), 0.25, 1.0, 0.96, OptionRight.CALL)
        # The floor matters for the in-the-money strikes, where the control
        # variate drives the sampling error to almost nothing; see McPrice.agrees_with.
        assert abs(price - exact) < max(5 * error, 1e-6 * exact), f"strike {strike}"
    # Calls are worth less the further out of the money they are.
    assert np.all(np.diff(prices) < 0)


def test_a_book_can_mix_calls_and_puts() -> None:
    prices, _ = monte_carlo_european_batch(
        np.array([100.0, 100.0]),
        np.array([100.0, 100.0]),
        np.array([0.25, 0.25]),
        np.array([1.0, 1.0]),
        np.array([0.96, 0.96]),
        ["C", "P"],
        paths=1 << 16,
        seed=1,
    )
    # Struck at the forward, a call and a put are worth the same.
    assert prices[0] == pytest.approx(prices[1], rel=1e-3)


def test_mismatched_book_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        monte_carlo_european_batch(
            np.array([100.0, 100.0]),
            np.array([100.0]),
            np.array([0.25, 0.25]),
            np.array([1.0, 1.0]),
            np.array([0.96, 0.96]),
            ["C", "P"],
        )


# ------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"forward": -1.0}, "positive"),
        ({"strike": 0.0}, "positive"),
        ({"vol": -0.1}, "negative"),
        ({"time": -1.0}, "negative"),
        ({"discount": 1.5}, "discount factor"),
        ({"paths": 0}, "path count"),
    ],
)
def test_nonsensical_terms_are_rejected(kwargs: dict[str, float], message: str) -> None:
    terms: dict[str, float] = {
        "forward": 100.0,
        "strike": 100.0,
        "vol": 0.2,
        "time": 1.0,
        "discount": 0.96,
    }
    terms.update(kwargs)
    with pytest.raises(ValueError, match=message):
        monte_carlo_european(**terms)  # type: ignore[arg-type]


def test_an_unknown_sampler_is_rejected() -> None:
    with pytest.raises(ValueError, match="sampler must be"):
        monte_carlo_european(100.0, 100.0, 0.2, 1.0, sampler="dice")


def test_an_unknown_option_right_is_rejected() -> None:
    with pytest.raises(ValueError, match="option right"):
        _core.mc_european(100.0, 100.0, 0.2, 1.0, 0.96, "X")


# --------------------------------------------------------------------- the type


def test_the_result_reports_a_confidence_interval() -> None:
    result = McPrice(price=10.0, standard_error=0.01, paths=1 << 20, replicas=1 << 20)
    assert result.confidence_95 == pytest.approx(0.0196, abs=1e-4)
    assert result.agrees_with(10.01)
    assert not result.agrees_with(10.5)
    assert "1,048,576 paths" in str(result)


# -------------------------------------------------- against the reference builds


def test_the_compiled_core_agrees_with_the_numpy_reference() -> None:
    # Same algorithm, same control variate, different implementation: they must
    # land on the same answer within their combined error.
    reference = numpy_european(100.0, 100.0, 0.25, 1.0, 0.96, OptionRight.CALL, 1 << 20, seed=2)
    compiled = monte_carlo_european(100.0, 100.0, 0.25, 1.0, 0.96, paths=1 << 20, seed=2)
    assert abs(reference - compiled.price) < 0.05


def test_the_python_loop_reference_agrees_too() -> None:
    reference = python_loop_european(
        100.0, 100.0, 0.25, 1.0, 0.96, OptionRight.CALL, 20_000, seed=4
    )
    exact = _exact(100.0, 100.0, 0.25, 1.0, 0.96, OptionRight.CALL)
    assert abs(reference - exact) < 0.25
