"""Reference implementations, kept for benchmarking and cross-checking.

Neither of these is used in anger. They exist so the compiled core can be
measured against something honest — the same algorithm, same variance
reduction, same seed handling — rather than against a straw man. A speedup over
a deliberately bad baseline is not a speedup.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from aegis.pricing.black_scholes import OptionRight, black_delta

__all__ = ["numpy_european", "python_loop_european"]

FloatArray = npt.NDArray[np.float64]


def python_loop_european(
    forward: float,
    strike: float,
    vol: float,
    time: float,
    discount: float = 1.0,
    right: OptionRight = OptionRight.CALL,
    paths: int = 100_000,
    seed: int = 0,
) -> float:
    """Price a European option with a plain Python loop.

    The version somebody writes first, before discovering that a million paths
    takes a coffee break.

    Args:
        forward: Forward price.
        strike: Strike price.
        vol: Implied volatility.
        time: Time to expiry in years.
        discount: Discount factor.
        right: Call or put.
        paths: Number of paths.
        seed: Seed for the generator.

    Returns:
        The simulated price.
    """
    rng = np.random.default_rng(seed)
    drift = -0.5 * vol * vol * time
    diffusion = vol * math.sqrt(time)
    sign = right.sign
    beta = float(black_delta(forward, strike, vol, time, 1.0, right)[0])

    total = 0.0
    for _ in range(paths):
        z = rng.standard_normal()
        terminal = forward * math.exp(drift + diffusion * z)
        payoff = max(sign * (terminal - strike), 0.0)
        total += payoff - beta * (terminal - forward)
    return discount * total / paths


def numpy_european(
    forward: float,
    strike: float,
    vol: float,
    time: float,
    discount: float = 1.0,
    right: OptionRight = OptionRight.CALL,
    paths: int = 1 << 20,
    seed: int = 0,
) -> float:
    """Price a European option with vectorised numpy.

    The version somebody writes second. It is the fair baseline for the compiled
    core: same algorithm, same control variate, no Python-level loop.

    Args:
        forward: Forward price.
        strike: Strike price.
        vol: Implied volatility.
        time: Time to expiry in years.
        discount: Discount factor.
        right: Call or put.
        paths: Number of paths.
        seed: Seed for the generator.

    Returns:
        The simulated price.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(paths)
    terminal = forward * np.exp(-0.5 * vol * vol * time + vol * math.sqrt(time) * z)
    payoff = np.maximum(right.sign * (terminal - strike), 0.0)
    beta = float(black_delta(forward, strike, vol, time, 1.0, right)[0])
    return float(discount * np.mean(payoff - beta * (terminal - forward)))
