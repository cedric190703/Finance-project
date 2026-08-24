"""Backtesting a VaR model.

A VaR number is a falsifiable claim: at 99% confidence, losses should exceed it
on about one day in a hundred. Backtesting is where that claim gets checked, and
it is the part of a risk system that separates a model somebody built from a
model somebody trusts.

Three tests, each answering a different question:

**Kupiec's proportion-of-failures test** asks whether there were the right
*number* of exceptions. Too many and the model is understating risk; too few and
it is wasting capital. It is a likelihood ratio against the binomial, distributed
chi-squared with one degree of freedom.

**Christoffersen's independence test** asks whether the exceptions were *spread
out*. A model can have exactly the right count and still be useless if all of
them fell in the same fortnight, because that means it does not react to
volatility. It compares the probability of an exception following an exception
against the unconditional rate.

**Conditional coverage** is the two together, chi-squared with two degrees of
freedom. A model has to pass both: the right number, in the right places.

Then the supervisory view. The **Basel traffic light** maps the exception count
onto green, amber and red zones and, in the amber zone, onto a capital multiplier
that rises with the count. It is not a statistical test — it is what the number
costs, which is why it is the one the desk hears about.

A note on which P&L is used. The tests here run on *hypothetical* P&L: the
position is held fixed and only the market moves. That is the regulatory
definition and it is the right one, because a backtest is meant to test the
model, not the trading. A day when the desk closed a losing position at noon
tells you nothing about whether the morning's VaR was well calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import binom, chi2

from aegis.instruments import MarketSnapshot
from aegis.portfolio import Portfolio
from aegis.risk.factors import FactorHistory
from aegis.risk.var import historical_pnl, value_at_risk

__all__ = [
    "BacktestResult",
    "BaselZone",
    "basel_zone",
    "christoffersen_independence",
    "conditional_coverage",
    "kupiec_pof",
    "rolling_backtest",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Significance level the tests are reported against.
_ALPHA = 0.05
#: Basel's zone boundaries, as cumulative binomial probabilities.
_AMBER_FROM = 0.95
_RED_FROM = 0.9999
#: Capital multipliers by exception count in the amber zone, for a 250-day year.
_AMBER_MULTIPLIERS = {5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}
_GREEN_MULTIPLIER = 3.00
_RED_MULTIPLIER = 4.00


class BaselZone(StrEnum):
    """The supervisory zone an exception count falls in."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"


@dataclass(frozen=True)
class BacktestResult:
    """The outcome of backtesting one VaR series.

    Attributes:
        observations: Days tested.
        exceptions: Days the loss exceeded the VaR forecast.
        expected: Exceptions the confidence level implies.
        confidence: The VaR confidence level tested.
        kupiec_statistic: Likelihood ratio for unconditional coverage.
        kupiec_p_value: Its p-value under chi-squared with one degree of freedom.
        independence_statistic: Likelihood ratio for independence of exceptions.
        independence_p_value: Its p-value, one degree of freedom.
        conditional_statistic: The two combined.
        conditional_p_value: Its p-value, two degrees of freedom.
        zone: Basel traffic-light zone.
        capital_multiplier: The multiplier that zone implies.
        worst_breach: Largest amount by which a loss exceeded its forecast.
    """

    observations: int
    exceptions: int
    expected: float
    confidence: float
    kupiec_statistic: float
    kupiec_p_value: float
    independence_statistic: float
    independence_p_value: float
    conditional_statistic: float
    conditional_p_value: float
    zone: BaselZone
    capital_multiplier: float
    worst_breach: float

    @property
    def exception_rate(self) -> float:
        """Return the realised exception rate."""
        return self.exceptions / self.observations if self.observations else 0.0

    @property
    def passes_kupiec(self) -> bool:
        """Return whether the exception count is consistent with the model."""
        return self.kupiec_p_value >= _ALPHA

    @property
    def passes_independence(self) -> bool:
        """Return whether exceptions were spread out rather than clustered."""
        return self.independence_p_value >= _ALPHA

    @property
    def passes(self) -> bool:
        """Return whether the model passes conditional coverage."""
        return self.conditional_p_value >= _ALPHA

    def summary(self) -> pl.DataFrame:
        """Return the results as a table for reporting.

        Returns:
            One row per test, with statistic, p-value and verdict.
        """
        return pl.DataFrame(
            {
                "test": [
                    "Kupiec (unconditional coverage)",
                    "Christoffersen (independence)",
                    "Conditional coverage",
                ],
                "statistic": [
                    self.kupiec_statistic,
                    self.independence_statistic,
                    self.conditional_statistic,
                ],
                "p_value": [
                    self.kupiec_p_value,
                    self.independence_p_value,
                    self.conditional_p_value,
                ],
                "verdict": [
                    "pass" if self.passes_kupiec else "reject",
                    "pass" if self.passes_independence else "reject",
                    "pass" if self.passes else "reject",
                ],
            }
        )

    def __str__(self) -> str:
        """Return a one-line summary."""
        return (
            f"{self.exceptions} exceptions in {self.observations} days "
            f"(expected {self.expected:.1f}), {self.zone} zone, "
            f"multiplier {self.capital_multiplier:.2f}"
        )


def kupiec_pof(exceptions: int, observations: int, confidence: float) -> tuple[float, float]:
    """Run Kupiec's proportion-of-failures test.

    Args:
        exceptions: Number of days the loss exceeded the forecast.
        observations: Number of days tested.
        confidence: VaR confidence level, e.g. 0.99.

    Returns:
        The likelihood ratio statistic and its p-value.

    Raises:
        ValueError: if there are no observations.
    """
    if observations <= 0:
        raise ValueError("cannot test an empty backtest")

    p = 1.0 - confidence
    x, n = exceptions, observations
    if x == 0:
        statistic = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        statistic = -2.0 * n * np.log(p)
    else:
        observed = x / n
        statistic = -2.0 * (
            (n - x) * np.log(1.0 - p)
            + x * np.log(p)
            - (n - x) * np.log(1.0 - observed)
            - x * np.log(observed)
        )
    return float(statistic), float(chi2.sf(statistic, df=1))


def christoffersen_independence(breaches: BoolArray) -> tuple[float, float]:
    """Test whether exceptions cluster.

    An exception today should say nothing about an exception tomorrow. When it
    does, the model is not reacting to changing volatility — it takes the right
    number of hits but takes them all in the same week, which is precisely the
    week the capital was needed.

    Args:
        breaches: One boolean per day: did the loss exceed the forecast?

    Returns:
        The likelihood ratio statistic and its p-value.

    Raises:
        ValueError: if there are fewer than two days.
    """
    if breaches.size < 2:
        raise ValueError("independence needs at least two observations")

    previous, current = breaches[:-1], breaches[1:]
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))

    # With no exceptions, or none following one another, there is nothing to
    # distinguish the two states and the test has no power. Reporting a
    # statistic of zero is honest: it says "found no evidence of clustering".
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0, 1.0

    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def log_likelihood(*terms: tuple[int, float]) -> float:
        return float(sum(count * np.log(probability) for count, probability in terms if count > 0))

    restricted = log_likelihood((n00 + n10, 1.0 - pi), (n01 + n11, pi))
    unrestricted = log_likelihood(
        (n00, 1.0 - pi_01), (n01, pi_01), (n10, 1.0 - pi_11), (n11, pi_11)
    )
    statistic = -2.0 * (restricted - unrestricted)
    return float(statistic), float(chi2.sf(statistic, df=1))


def conditional_coverage(breaches: BoolArray, confidence: float) -> tuple[float, float]:
    """Combine the coverage and independence tests.

    Args:
        breaches: One boolean per day.
        confidence: VaR confidence level.

    Returns:
        The combined statistic and its p-value, on two degrees of freedom.
    """
    coverage, _ = kupiec_pof(int(np.sum(breaches)), breaches.size, confidence)
    independence, _ = christoffersen_independence(breaches)
    statistic = coverage + independence
    return float(statistic), float(chi2.sf(statistic, df=2))


def basel_zone(
    exceptions: int, observations: int = 250, confidence: float = 0.99
) -> tuple[BaselZone, float]:
    """Map an exception count onto a supervisory zone and capital multiplier.

    The published Basel table is defined for a 250-day year at 99%: green up to
    four exceptions, amber from five to nine, red at ten. Those boundaries are
    not arbitrary — they are where the cumulative binomial probability of seeing
    that many or fewer crosses 95% and 99.99%. Computing them that way rather
    than hard-coding the table means the zones stay meaningful for a window that
    is not exactly 250 days long, and it reproduces the published table exactly
    when it is.

    Args:
        exceptions: Number of exceptions observed.
        observations: Number of days tested.
        confidence: VaR confidence level.

    Returns:
        The zone and the capital multiplier it implies.
    """
    cumulative = float(binom.cdf(exceptions, observations, 1.0 - confidence))
    if cumulative < _AMBER_FROM:
        return BaselZone.GREEN, _GREEN_MULTIPLIER
    if cumulative < _RED_FROM:
        scaled = round(exceptions * 250 / observations) if observations else exceptions
        return BaselZone.AMBER, _AMBER_MULTIPLIERS.get(scaled, _AMBER_MULTIPLIERS[9])
    return BaselZone.RED, _RED_MULTIPLIER


def backtest_series(
    pnl: FloatArray, var_forecast: FloatArray, confidence: float = 0.99
) -> BacktestResult:
    """Backtest a series of VaR forecasts against realised P&L.

    Args:
        pnl: Realised profit and loss, one per day.
        var_forecast: The VaR forecast made *before* each day, as a positive loss.
        confidence: The confidence level those forecasts were made at.

    Returns:
        The full set of test results.

    Raises:
        ValueError: if the two series are not the same length.
    """
    if pnl.size != var_forecast.size:
        raise ValueError("P&L and VaR series must be the same length")

    breaches = pnl < -var_forecast
    exceptions = int(np.sum(breaches))
    kupiec_statistic, kupiec_p = kupiec_pof(exceptions, pnl.size, confidence)
    independence_statistic, independence_p = christoffersen_independence(breaches)
    conditional_statistic, conditional_p = conditional_coverage(breaches, confidence)
    zone, multiplier = basel_zone(exceptions, pnl.size, confidence)
    shortfall = -(pnl + var_forecast)

    return BacktestResult(
        observations=pnl.size,
        exceptions=exceptions,
        expected=pnl.size * (1.0 - confidence),
        confidence=confidence,
        kupiec_statistic=kupiec_statistic,
        kupiec_p_value=kupiec_p,
        independence_statistic=independence_statistic,
        independence_p_value=independence_p,
        conditional_statistic=conditional_statistic,
        conditional_p_value=conditional_p,
        zone=zone,
        capital_multiplier=multiplier,
        worst_breach=float(shortfall.max()) if exceptions else 0.0,
    )


def rolling_backtest(
    portfolio: Portfolio,
    market: MarketSnapshot,
    history: FactorHistory,
    confidence: float = 0.99,
    window: int = 250,
) -> tuple[BacktestResult, pl.DataFrame]:
    """Walk a VaR model forward through history and test what it produced.

    On each day the model sees only the ``window`` days before it, forecasts a
    VaR, and is then judged against the next day's move. No day's forecast uses
    information from its own outcome, which is the property a backtest lives or
    dies by.

    The P&L is hypothetical, in the regulatory sense: positions are frozen and
    only the market moves. That also makes the computation cheap — the P&L of
    every historical move against the frozen book is computed once, and the
    rolling forecast is a quantile over a sliding window of that same vector.

    Args:
        portfolio: The book, held constant throughout.
        market: The market the book is valued in.
        history: Factor moves to walk through.
        confidence: Confidence level for the forecasts.
        window: How many days of history each forecast may use.

    Returns:
        The test results and a day-by-day frame of forecast, outcome and breach.

    Raises:
        ValueError: if the history is not longer than the window.
    """
    if len(history) <= window:
        raise ValueError(
            f"need more than {window} observations to backtest a {window}-day window, "
            f"got {len(history)}"
        )

    pnl = historical_pnl(portfolio, market, history)
    forecasts = np.array(
        [value_at_risk(pnl[day - window : day], confidence) for day in range(window, pnl.size)]
    )
    outcomes = pnl[window:]
    dates = history.dates[window:]

    result = backtest_series(outcomes, forecasts, confidence)
    frame = pl.DataFrame(
        {
            "value_date": list(dates),
            "pnl": outcomes,
            "var_forecast": forecasts,
            "breach": outcomes < -forecasts,
            "shortfall": np.where(outcomes < -forecasts, -(outcomes + forecasts), 0.0),
        }
    )
    return result, frame
