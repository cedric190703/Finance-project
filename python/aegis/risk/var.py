"""Value at Risk and Expected Shortfall.

Three methods, because they disagree, and where they disagree is informative:

**Historical** replays each of the last few hundred daily factor moves through a
full revaluation of the book. It makes no distributional assumption at all, so
it captures fat tails and the way correlations tighten in a sell-off — but it
can only produce losses that have already happened, and its tail rests on the
handful of worst days in the window.

**Parametric** (delta-normal) takes the book's sensitivities and the factor
covariance matrix and reads the quantile off a normal distribution. It is fast
and it is smooth, which is why it is still used for limits monitoring, and it is
wrong in exactly two ways: markets are not normal, and a book with options in it
is not linear. The Cornish-Fisher variant repairs the first by adjusting the
quantile for the observed skew and excess kurtosis of the *portfolio's* P&L.

**Monte Carlo** draws correlated normal factor moves from the same covariance
matrix and fully revalues under each. It keeps the parametric method's smooth
tail while dropping its linearity assumption, so the gap between the two is a
direct measure of how much the book's optionality matters.

A note on sign convention, because it causes endless confusion: VaR is reported
here as a **positive number representing a loss**. A 99% one-day VaR of 42,000
means "on the worst day in a hundred, expect to lose at least 42,000".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import polars as pl

from aegis.instruments import MarketSnapshot
from aegis.portfolio import Portfolio
from aegis.risk.factors import FactorHistory
from aegis.risk.scenarios import apply_shocks

__all__ = [
    "VarMethod",
    "VarResult",
    "contribution_report",
    "expected_shortfall",
    "factor_exposures",
    "historical_pnl",
    "historical_var",
    "monte_carlo_var",
    "parametric_var",
    "value_at_risk",
]

FloatArray = npt.NDArray[np.float64]

#: Below this many observations a tail quantile is not worth reporting.
_MIN_OBSERVATIONS = 30
#: Range of standard deviations over which the Cornish-Fisher expansion must stay
#: monotone to be usable. Covers everything from the median out past 99.9%.
_CORNISH_FISHER_RANGE = 3.5


class VarMethod(StrEnum):
    """How a VaR number was produced."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    CORNISH_FISHER = "cornish-fisher"
    MONTE_CARLO = "monte-carlo"


@dataclass(frozen=True)
class VarResult:
    """A VaR and ES pair, with everything needed to reproduce it.

    Attributes:
        method: How it was computed.
        confidence: Confidence level, e.g. 0.99.
        horizon_days: Holding period the number applies to.
        var: The loss threshold, reported positive.
        expected_shortfall: Mean loss conditional on breaching it, positive.
        observations: How many scenarios the estimate rests on.
        portfolio_value: The book's value in the base scenario.
    """

    method: VarMethod
    confidence: float
    horizon_days: int
    var: float
    expected_shortfall: float
    observations: int
    portfolio_value: float

    @property
    def var_percent(self) -> float:
        """Return the VaR as a percentage of portfolio value."""
        return 100.0 * self.var / abs(self.portfolio_value) if self.portfolio_value else 0.0

    def __str__(self) -> str:
        """Return a one-line summary in the form a risk report uses."""
        return (
            f"{self.confidence:.0%} {self.horizon_days}-day {self.method} VaR "
            f"{self.var:,.0f} ({self.var_percent:.2f}%), ES {self.expected_shortfall:,.0f}"
        )


def historical_pnl(
    portfolio: Portfolio,
    market: MarketSnapshot,
    history: FactorHistory,
    horizon_days: int = 1,
) -> FloatArray:
    """Revalue the book under every historical day's factor moves.

    Each scenario asks a counterfactual: what would this book have made if
    *today's* market had moved the way it did on that day? The positions and the
    starting market are today's; only the moves are historical.

    Args:
        portfolio: The book to revalue.
        market: Today's market state.
        history: The factor moves to replay.
        horizon_days: Holding period. Moves are scaled by its square root, which
            is the standard convention and assumes independence across days —
            true enough for equities, optimistic in a trending market.

    Returns:
        One P&L per historical observation, in base currency.
    """
    base = portfolio.value(market)
    scale = float(np.sqrt(horizon_days))
    pnl = np.empty(len(history), dtype=np.float64)
    for index in range(len(history)):
        shocks = {k: v * scale for k, v in history.scenario(index).items()}
        pnl[index] = portfolio.value(apply_shocks(market, shocks)) - base
    return pnl


def value_at_risk(pnl: FloatArray, confidence: float = 0.99) -> float:
    """Return the VaR of a P&L distribution, as a positive loss.

    The quantile is taken with linear interpolation between order statistics
    rather than by picking the nearest observation. With 500 days and 99%
    confidence the threshold falls between the fifth and sixth worst day, and
    rounding to one or the other moves the number by a noticeable amount.

    Args:
        pnl: Simulated or historical profit and loss.
        confidence: Confidence level, e.g. 0.99.

    Returns:
        The loss threshold, positive by convention. Zero if the tail is a profit.

    Raises:
        ValueError: if the confidence level is not in (0, 1) or there are too
            few observations for the quantile to mean anything.
    """
    _validate(pnl, confidence)
    return float(max(-np.quantile(pnl, 1.0 - confidence, method="linear"), 0.0))


def expected_shortfall(pnl: FloatArray, confidence: float = 0.975) -> float:
    """Return the Expected Shortfall — the mean loss beyond the VaR threshold.

    ES is the regulatory measure under the Fundamental Review of the Trading
    Book, and for a good reason: VaR says nothing about how bad the bad days
    are, and it is not sub-additive, so it can claim that splitting a book into
    two desks reduces total risk. ES has neither problem. Basel pairs 97.5% ES
    with 99% VaR because for a normal distribution the two land in a similar
    place, while ES keeps looking further into a fat tail.

    Args:
        pnl: Simulated or historical profit and loss.
        confidence: Confidence level, e.g. 0.975.

    Returns:
        The mean loss in the tail, positive by convention.

    Raises:
        ValueError: if the confidence level or sample size is unusable.
    """
    _validate(pnl, confidence)
    threshold = np.quantile(pnl, 1.0 - confidence, method="linear")
    tail = pnl[pnl <= threshold]
    if tail.size == 0:  # pragma: no cover - quantile guarantees at least one
        return float(max(-threshold, 0.0))
    return float(max(-tail.mean(), 0.0))


def historical_var(
    portfolio: Portfolio,
    market: MarketSnapshot,
    history: FactorHistory,
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> VarResult:
    """Compute historical-simulation VaR and ES.

    Args:
        portfolio: The book.
        market: Today's market state.
        history: Historical factor moves to replay.
        confidence: Confidence level for the VaR.
        horizon_days: Holding period.

    Returns:
        The result, with ES taken at 97.5% alongside.
    """
    pnl = historical_pnl(portfolio, market, history, horizon_days)
    return VarResult(
        method=VarMethod.HISTORICAL,
        confidence=confidence,
        horizon_days=horizon_days,
        var=value_at_risk(pnl, confidence),
        expected_shortfall=expected_shortfall(pnl, 0.975),
        observations=pnl.size,
        portfolio_value=portfolio.value(market),
    )


def parametric_var(
    portfolio: Portfolio,
    market: MarketSnapshot,
    history: FactorHistory,
    confidence: float = 0.99,
    horizon_days: int = 1,
    cornish_fisher: bool = False,
) -> VarResult:
    """Compute delta-normal VaR from the factor covariance matrix.

    The book's exposure to each factor is measured by a one-unit bump and full
    revaluation, which keeps the method honest about non-linear positions at the
    margin even though the distributional assumption remains linear.

    Args:
        portfolio: The book.
        market: Today's market state.
        history: Factor history, for its covariance matrix.
        confidence: Confidence level.
        horizon_days: Holding period.
        cornish_fisher: Adjust the normal quantile for the P&L distribution's
            observed skew and excess kurtosis.

    Returns:
        The result.
    """
    from scipy.stats import norm

    exposures = factor_exposures(portfolio, market, history)
    covariance = history.covariance()
    variance = float(exposures @ covariance @ exposures) * horizon_days
    deviation = float(np.sqrt(max(variance, 0.0)))

    quantile = float(norm.ppf(confidence))
    method = VarMethod.PARAMETRIC
    if cornish_fisher:
        pnl = historical_pnl(portfolio, market, history, horizon_days)
        quantile = _cornish_fisher_quantile(quantile, pnl)
        method = VarMethod.CORNISH_FISHER

    es_quantile = float(norm.pdf(norm.ppf(0.975)) / (1.0 - 0.975))
    return VarResult(
        method=method,
        confidence=confidence,
        horizon_days=horizon_days,
        var=max(quantile * deviation, 0.0),
        expected_shortfall=max(es_quantile * deviation, 0.0),
        observations=len(history),
        portfolio_value=portfolio.value(market),
    )


def monte_carlo_var(
    portfolio: Portfolio,
    market: MarketSnapshot,
    history: FactorHistory,
    confidence: float = 0.99,
    horizon_days: int = 1,
    scenarios: int = 10_000,
    seed: int = 0,
) -> VarResult:
    """Compute VaR by simulating correlated factor moves and revaluing in full.

    The covariance matrix is factorised with an eigenvalue decomposition rather
    than a Cholesky. A sample covariance estimated from fewer days than it has
    factors is singular, and one estimated from barely more is numerically close
    to it; Cholesky fails outright on the first and produces nonsense on the
    second. Clipping negative eigenvalues to zero yields the nearest positive
    semi-definite matrix, which is the standard repair.

    Args:
        portfolio: The book.
        market: Today's market state.
        history: Factor history, for its covariance matrix.
        confidence: Confidence level.
        horizon_days: Holding period.
        scenarios: How many paths to draw.
        seed: Seed, so the number is reproducible.

    Returns:
        The result.
    """
    covariance = history.covariance() * horizon_days
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor_loadings = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))

    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((scenarios, len(history.factors)))
    moves = draws @ factor_loadings.T

    base = portfolio.value(market)
    pnl = np.empty(scenarios, dtype=np.float64)
    for index in range(scenarios):
        shocks = dict(zip(history.factors, moves[index], strict=True))
        pnl[index] = portfolio.value(apply_shocks(market, shocks)) - base

    return VarResult(
        method=VarMethod.MONTE_CARLO,
        confidence=confidence,
        horizon_days=horizon_days,
        var=value_at_risk(pnl, confidence),
        expected_shortfall=expected_shortfall(pnl, 0.975),
        observations=scenarios,
        portfolio_value=base,
    )


def factor_exposures(
    portfolio: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> FloatArray:
    """Return the book's P&L per unit move in each factor.

    Args:
        portfolio: The book.
        market: Today's market state.
        history: Supplies the factor list and how each is shocked.

    Returns:
        One exposure per factor, in factor order.
    """
    base = portfolio.value(market)
    exposures = np.zeros(len(history.factors), dtype=np.float64)
    for index, factor in enumerate(history.factors):
        bump = 0.01 if history.kinds[factor] != "absolute" else 1e-4
        shocked = portfolio.value(apply_shocks(market, {factor: bump}))
        exposures[index] = (shocked - base) / bump
    return exposures


def contribution_report(
    portfolio: Portfolio, market: MarketSnapshot, history: FactorHistory
) -> pl.DataFrame:
    """Break the parametric VaR down by factor.

    Component VaR is what a risk report is actually read for: not "the book risks
    two million" but "and eighty percent of it is the equity leg". The components
    sum to the total, because each is the factor's marginal contribution
    multiplied by its own exposure.

    Args:
        portfolio: The book.
        market: Today's market state.
        history: Factor history.

    Returns:
        Per-factor exposure, standalone volatility and component contribution.
    """
    exposures = factor_exposures(portfolio, market, history)
    covariance = history.covariance()
    variance = float(exposures @ covariance @ exposures)
    deviation = np.sqrt(max(variance, 0.0))
    marginal = (covariance @ exposures) / deviation if deviation > 0 else np.zeros_like(exposures)

    return pl.DataFrame(
        {
            "factor": list(history.factors),
            "exposure": exposures,
            "standalone_vol": np.sqrt(np.diag(covariance)) * np.abs(exposures),
            "component": marginal * exposures,
            "share": (marginal * exposures) / deviation if deviation > 0 else exposures * 0.0,
        }
    ).sort("component", descending=True)


def _cornish_fisher_quantile(normal_quantile: float, pnl: FloatArray) -> float:
    """Adjust a normal quantile for the sample's skew and excess kurtosis.

    The expansion is a third-order correction, and like every truncated series it
    is only trustworthy near the point it was expanded about. Fed a sample with
    excess kurtosis of sixteen — which two years of index returns containing one
    ten-percent day genuinely produce — the correction stops being monotone in
    the underlying quantile, and the "99% VaR" it returns can exceed the 99.9%
    one. On the demo book that failure inflated the number by 130%: not a
    conservative estimate, an arithmetically meaningless one.

    So the moments are shrunk towards zero by the largest factor that keeps the
    transform monotone across the working range. Where the sample is mildly
    non-normal nothing is shrunk and the correction applies in full; where it is
    wild the estimate degrades gracefully towards the normal quantile instead of
    exploding. The alternative — reporting it anyway — is how a risk system
    loses its readers.

    Args:
        normal_quantile: The Gaussian quantile being corrected.
        pnl: The P&L sample supplying the moments.

    Returns:
        The adjusted quantile.
    """
    deviation = pnl.std(ddof=1)
    if deviation <= 0:
        return normal_quantile
    scaled = (pnl - pnl.mean()) / deviation
    skew = float(np.mean(scaled**3))
    excess_kurtosis = float(np.mean(scaled**4) - 3.0)

    shrink = _monotone_shrinkage(skew, excess_kurtosis)
    return _expansion(normal_quantile, skew * shrink, excess_kurtosis * shrink)


def _expansion(z: float, skew: float, excess_kurtosis: float) -> float:
    """Evaluate the Cornish-Fisher polynomial."""
    return float(
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )


def _monotone_shrinkage(skew: float, excess_kurtosis: float) -> float:
    """Return the largest shrinkage in [0, 1] keeping the expansion monotone."""
    grid = np.linspace(-_CORNISH_FISHER_RANGE, _CORNISH_FISHER_RANGE, 241)

    def monotone(factor: float) -> bool:
        values = np.array(
            [_expansion(float(z), skew * factor, excess_kurtosis * factor) for z in grid]
        )
        return bool(np.all(np.diff(values) > 0.0))

    if monotone(1.0):
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        if monotone(middle):
            low = middle
        else:
            high = middle
    return low


def _validate(pnl: FloatArray, confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if pnl.size < _MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {_MIN_OBSERVATIONS} observations for a tail quantile, got {pnl.size}"
        )
