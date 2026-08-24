"""Monte Carlo pricing, executed in the compiled Rust core.

This module is a thin, typed front door to ``aegis._core``. The numerical work
happens in Rust; what lives here is the argument checking, the result type, and
the documentation of what the engine actually guarantees.

Two of those guarantees are worth repeating, because they are the reason the
engine is written this way rather than as fifteen lines of numpy:

* **The answer does not depend on the machine.** Work is split into fixed-size
  chunks seeded from the run seed and the chunk index, so one thread and ninety-
  six threads produce bit-identical prices. A risk number that changes when the
  job lands on a different box cannot be reconciled, and a P&L explain that
  cannot be reconciled cannot be signed off.
* **Every price comes with an error bar.** A Monte Carlo price without a
  standard error is an opinion. Under quasi-random sampling the error is
  estimated across independently shifted replicas rather than across points,
  because points inside a low-discrepancy sequence are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import t as student_t

from aegis import _core
from aegis.pricing.black_scholes import OptionRight

__all__ = ["McPrice", "Sampler", "monte_carlo_european", "monte_carlo_european_batch"]

FloatArray = npt.NDArray[np.float64]

#: Sampler names the compiled core accepts.
Sampler = str

_CONFIDENCE_95 = 1.959963984540054


@dataclass(frozen=True)
class McPrice:
    """A simulated price and its uncertainty.

    Attributes:
        price: The estimated present value.
        standard_error: Standard error of that estimate.
        paths: Payoff evaluations consumed, including antithetic partners.
        replicas: How many independent observations the standard error rests on.
            Under pseudo-random sampling that is the sample count. Under a
            randomised low-discrepancy sequence it is the number of independently
            shifted replicas, which is often only a handful — and a standard
            error estimated from eight numbers is itself a noisy quantity.
    """

    price: float
    standard_error: float
    paths: int
    replicas: int = 0

    @property
    def confidence_95(self) -> float:
        """Return the half-width of a two-sided 95% confidence interval.

        Built on a Student-t quantile with ``replicas - 1`` degrees of freedom.
        With eight replicas that is 2.36 rather than 1.96, a 20% wider interval;
        with four it is 3.18. Using the normal quantile on a handful of replicas
        reports an interval the estimate does not actually earn.
        """
        if self.replicas > 1:
            return float(student_t.ppf(0.975, self.replicas - 1)) * self.standard_error
        return _CONFIDENCE_95 * self.standard_error

    def agrees_with(self, reference: float, sigmas: float = 3.0, absolute: float = 0.0) -> bool:
        """Return whether a reference value sits inside the simulation's error bars.

        The comparison carries a relative floor as well as the error bars, and
        the reason is worth knowing. On a deeply in-the-money option the control
        variate is almost perfect — the payoff is very nearly the linear function
        being subtracted — so the sample variance collapses towards zero and the
        reported standard error with it. What does not collapse is the estimator's
        blindness to a tail it never sampled: the closed form still prices the
        one-in-a-hundred-million path that finishes out of the money, and a
        simulation of sixty thousand paths never sees one. The result is a
        discrepancy that is economically nil but arbitrarily many "standard
        errors" wide. A vanishing error bar means the *sampling* error is small,
        not that the answer is exact.

        The mirror-image case is a deeply out-of-the-money option, where the
        payoff is almost always zero and the price is carried by a handful of
        paths in ten thousand. Plain Monte Carlo is simply the wrong tool there —
        pricing a 1e-4 option to 1% needs importance sampling, not more paths —
        and the standard error says so honestly by being the same size as the
        price. What saves the situation is that nobody cares: an absolute error
        of a hundredth of a basis point of notional is not a risk number, it is
        rounding. Callers who know the notional can say so through ``absolute``.

        Args:
            reference: An independently computed value, typically a closed form.
            sigmas: How many standard errors count as agreement.
            absolute: An absolute difference to accept regardless of the error
                bars — the caller's statement of what is immaterial at this size.

        Returns:
            ``True`` when the reference lies within ``sigmas`` standard errors,
            within ``absolute``, or within a relative floor of one part in a
            hundred million.
        """
        floor = 1e-8 * max(abs(self.price), 1.0)
        # Scale the requested number of sigmas the same way the interval is
        # scaled, so a run resting on few replicas is judged on the wider bar it
        # actually earns.
        widening = self.confidence_95 / max(_CONFIDENCE_95 * self.standard_error, 1e-300)
        return abs(self.price - reference) <= max(
            sigmas * widening * self.standard_error, floor, absolute
        )

    def __str__(self) -> str:
        """Return the price with its 95% interval."""
        return f"{self.price:.6f} ± {self.confidence_95:.6f} ({self.paths:,} paths)"


def monte_carlo_european(
    forward: float,
    strike: float,
    vol: float,
    time: float,
    discount: float = 1.0,
    right: OptionRight = OptionRight.CALL,
    paths: int = 1 << 20,
    seed: int = 0,
    sampler: Sampler = "quasi",
    antithetic: bool | None = None,
    control_variate: bool | None = None,
) -> McPrice:
    """Price a European option by simulation.

    Args:
        forward: Forward price of the underlying to expiry.
        strike: Strike price.
        vol: Black implied volatility.
        time: Time to expiry in years.
        discount: Discount factor to the payment date.
        right: Call or put.
        paths: Requested path count; rounded up to a whole number of chunks.
        seed: Seed for the run. The same seed always gives the same answer.
        sampler: ``"quasi"`` for a randomised Sobol sequence, ``"pseudo"`` for
            xoshiro256++.
        antithetic: Override the sampler's recommended setting.
        control_variate: Override the sampler's recommended setting.

    Returns:
        The simulated price and its standard error.

    Raises:
        ValueError: if the option terms are not economically sensible.
    """
    _validate(forward, strike, vol, time, discount, paths)
    price, standard_error, consumed, replicas = _core.mc_european(
        forward,
        strike,
        vol,
        time,
        discount,
        right.value,
        paths,
        seed,
        sampler,
        antithetic,
        control_variate,
    )
    return McPrice(price, standard_error, consumed, replicas)


def monte_carlo_european_batch(
    forwards: FloatArray,
    strikes: FloatArray,
    vols: FloatArray,
    times: FloatArray,
    discounts: FloatArray,
    rights: list[str],
    paths: int = 1 << 18,
    seed: int = 0,
    sampler: Sampler = "quasi",
    antithetic: bool | None = None,
    control_variate: bool | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Price a book of European options, in parallel across the book.

    Revaluing a portfolio is embarrassingly parallel across positions, which is
    a better place to spend threads than inside a single option: there is no
    reduction at the end and no shared state at all.

    Args:
        forwards: Forward price per option.
        strikes: Strike per option.
        vols: Implied volatility per option.
        times: Time to expiry per option, in years.
        discounts: Discount factor per option.
        rights: ``"C"`` or ``"P"`` per option.
        paths: Paths per option.
        seed: Base seed; each option is offset from it.
        sampler: ``"quasi"`` or ``"pseudo"``.
        antithetic: Override the sampler's recommended setting.
        control_variate: Override the sampler's recommended setting.

    Returns:
        Prices and standard errors, aligned with the inputs.
    """
    return _core.mc_european_batch(
        np.ascontiguousarray(forwards, dtype=np.float64),
        np.ascontiguousarray(strikes, dtype=np.float64),
        np.ascontiguousarray(vols, dtype=np.float64),
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(discounts, dtype=np.float64),
        rights,
        paths,
        seed,
        sampler,
        antithetic,
        control_variate,
    )


def _validate(
    forward: float, strike: float, vol: float, time: float, discount: float, paths: int
) -> None:
    if forward <= 0 or strike <= 0:
        raise ValueError("forward and strike must be positive")
    if vol < 0:
        raise ValueError("volatility cannot be negative")
    if time < 0:
        raise ValueError("time to expiry cannot be negative")
    if not 0 < discount <= 1.0:
        raise ValueError("discount factor must lie in (0, 1]")
    if paths <= 0:
        raise ValueError("path count must be positive")
