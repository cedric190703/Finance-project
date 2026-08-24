"""Black-Scholes-Merton in forward form, and the inversion back to implied vol.

Everything is written against the *forward*, not the spot. Working in forward
terms means the dividend yield, the borrow cost and the repo rate never appear
separately: whatever the market thinks of them is already in the forward, which
is exactly what put-call parity lets us read off the quotes. It also makes the
same code price an equity option, an FX option and a bond future option without
a single branch.

    call = DF · [F·N(d1) − K·N(d2)]
    put  = DF · [K·N(−d2) − F·N(−d1)]
    d1   = [ln(F/K) + σ²T/2] / (σ√T),   d2 = d1 − σ√T

The greeks below are *forward* greeks — sensitivities to the forward, the
volatility and time. Converting a forward delta into the spot delta a desk
hedges with is the pricing layer's job in phase 6, because that conversion is
where the dividend assumption finally has to be pinned down.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

__all__ = [
    "OptionRight",
    "black_delta",
    "black_gamma",
    "black_price",
    "black_theta",
    "black_vega",
    "implied_volatility",
]

FloatArray = npt.NDArray[np.float64]
Number = float | FloatArray

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
#: Below this the option is treated as expired or the vol as degenerate.
_TINY = 1e-12


class OptionRight(StrEnum):
    """Whether an option is a call or a put."""

    CALL = "C"
    PUT = "P"

    @property
    def sign(self) -> float:
        """Return +1 for a call and -1 for a put, the usual pricing shorthand."""
        return 1.0 if self is OptionRight.CALL else -1.0


def _d1_d2(
    forward: FloatArray, strike: FloatArray, vol: FloatArray, time: FloatArray
) -> tuple[FloatArray, FloatArray]:
    total_vol = vol * np.sqrt(time)
    safe = np.maximum(total_vol, _TINY)
    d1 = (np.log(np.maximum(forward, _TINY) / np.maximum(strike, _TINY)) + 0.5 * safe**2) / safe
    return d1, d1 - safe


def _normal_pdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x**2) / _SQRT_2PI


def _broadcast(*values: Number) -> tuple[FloatArray, ...]:
    arrays = [np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in values]
    return tuple(np.broadcast_arrays(*arrays))


def black_price(
    forward: Number,
    strike: Number,
    vol: Number,
    time: Number,
    discount: Number = 1.0,
    right: OptionRight = OptionRight.CALL,
) -> FloatArray:
    """Price a European option off the forward.

    Args:
        forward: Forward price of the underlying to the option's expiry.
        strike: Strike price.
        vol: Black implied volatility, annualised.
        time: Time to expiry in years.
        discount: Discount factor to the payment date.
        right: Call or put.

    Returns:
        The option premium. At zero time or zero volatility the intrinsic value
        is returned rather than a division by zero.
    """
    f, k, v, t, df = _broadcast(forward, strike, vol, time, discount)
    d1, d2 = _d1_d2(f, k, v, t)
    sign = right.sign

    value = sign * df * (f * ndtr(sign * d1) - k * ndtr(sign * d2))
    intrinsic = df * np.maximum(sign * (f - k), 0.0)
    degenerate = (t <= _TINY) | (v <= _TINY)
    return np.where(degenerate, intrinsic, value)


def black_delta(
    forward: Number,
    strike: Number,
    vol: Number,
    time: Number,
    discount: Number = 1.0,
    right: OptionRight = OptionRight.CALL,
) -> FloatArray:
    """Return the forward delta, the sensitivity to a move in the forward.

    Args:
        forward: Forward price.
        strike: Strike price.
        vol: Implied volatility.
        time: Time to expiry in years.
        discount: Discount factor to the payment date.
        right: Call or put.

    Returns:
        ``dPrice/dF``, bounded by ±1 times the discount factor.
    """
    f, k, v, t, df = _broadcast(forward, strike, vol, time, discount)
    d1, _ = _d1_d2(f, k, v, t)
    sign = right.sign
    live = df * sign * ndtr(sign * d1)
    expired = df * sign * (sign * (f - k) > 0.0)
    return np.where((t <= _TINY) | (v <= _TINY), expired, live)


def black_gamma(
    forward: Number,
    strike: Number,
    vol: Number,
    time: Number,
    discount: Number = 1.0,
) -> FloatArray:
    """Return the forward gamma; identical for calls and puts.

    Args:
        forward: Forward price.
        strike: Strike price.
        vol: Implied volatility.
        time: Time to expiry in years.
        discount: Discount factor to the payment date.

    Returns:
        ``d²Price/dF²``.
    """
    f, k, v, t, df = _broadcast(forward, strike, vol, time, discount)
    d1, _ = _d1_d2(f, k, v, t)
    total_vol = np.maximum(v * np.sqrt(t), _TINY)
    gamma = df * _normal_pdf(d1) / (np.maximum(f, _TINY) * total_vol)
    return np.where((t <= _TINY) | (v <= _TINY), 0.0, gamma)


def black_vega(
    forward: Number,
    strike: Number,
    vol: Number,
    time: Number,
    discount: Number = 1.0,
) -> FloatArray:
    """Return vega per unit of volatility (not per vol point).

    Args:
        forward: Forward price.
        strike: Strike price.
        vol: Implied volatility.
        time: Time to expiry in years.
        discount: Discount factor to the payment date.

    Returns:
        ``dPrice/dσ``. Divide by 100 for the "per vol point" number a desk quotes.
    """
    f, k, v, t, df = _broadcast(forward, strike, vol, time, discount)
    d1, _ = _d1_d2(f, k, v, t)
    vega = df * f * _normal_pdf(d1) * np.sqrt(t)
    return np.where((t <= _TINY) | (v <= _TINY), 0.0, vega)


def black_theta(
    forward: Number,
    strike: Number,
    vol: Number,
    time: Number,
    discount: Number = 1.0,
    right: OptionRight = OptionRight.CALL,
    rate: Number = 0.0,
) -> FloatArray:
    """Return theta per year, holding the forward fixed.

    Args:
        forward: Forward price.
        strike: Strike price.
        vol: Implied volatility.
        time: Time to expiry in years.
        discount: Discount factor to the payment date.
        right: Call or put.
        rate: Continuously compounded discount rate, needed because the
            discount factor itself rolls up as expiry approaches.

    Returns:
        ``dPrice/dt`` in premium per year; negative for a long option in the
        usual case.
    """
    f, k, v, t, df, r = _broadcast(forward, strike, vol, time, discount, rate)
    d1, d2 = _d1_d2(f, k, v, t)
    sign = right.sign
    decay = -df * f * _normal_pdf(d1) * v / (2.0 * np.sqrt(np.maximum(t, _TINY)))
    carry = r * df * sign * (f * ndtr(sign * d1) - k * ndtr(sign * d2))
    return np.where((t <= _TINY) | (v <= _TINY), 0.0, decay + carry)


def implied_volatility(
    price: Number,
    forward: Number,
    strike: Number,
    time: Number,
    discount: Number = 1.0,
    right: OptionRight = OptionRight.CALL,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> FloatArray:
    """Invert the Black formula for implied volatility.

    Newton's method converges in a handful of steps near the money but is
    fragile in the wings, where vega collapses and the price barely responds to
    volatility. So Newton is run first and any point that fails to converge —
    or steps outside the bracket — is handed to a bisection that cannot diverge.

    Three situations return NaN rather than a number:

    * the quote violates the no-arbitrage bounds — below intrinsic or above the
      forward — so no volatility could have produced it;
    * the iteration did not converge to the given price;
    * vega at the solution is negligible relative to the option's scale. Deep in
      the wings the premium is the same to double precision across a wide band
      of volatilities, so the price does not identify one. Returning the number
      the iteration happened to stop on would be inventing a data point, and a
      surface fitted to invented points is worse than a surface with holes.

    Args:
        price: Observed option premium.
        forward: Forward price.
        strike: Strike price.
        time: Time to expiry in years.
        discount: Discount factor to the payment date.
        right: Call or put.
        tolerance: Absolute price tolerance for convergence.
        max_iterations: Cap on Newton and bisection iterations.

    Returns:
        The implied volatility, or NaN where no volatility can reproduce the price.
    """
    p, f, k, t, df = _broadcast(price, forward, strike, time, discount)

    intrinsic = df * np.maximum(right.sign * (f - k), 0.0)
    upper_bound = df * (f if right is OptionRight.CALL else k)
    unpriceable = (p < intrinsic - tolerance) | (p > upper_bound + tolerance) | (t <= _TINY)

    vol = np.where(unpriceable, np.nan, 0.30)
    low = np.full_like(vol, 1e-8)
    high = np.full_like(vol, 5.0)

    for _ in range(max_iterations):
        active = ~unpriceable & (np.abs(black_price(f, k, vol, t, df, right) - p) > tolerance)
        if not np.any(active):
            break
        error = black_price(f, k, vol, t, df, right) - p
        vega = black_vega(f, k, vol, t, df)
        # Where vega is meaningful, step with Newton; elsewhere, bisect.
        newton_ok = active & (vega > 1e-8)
        step = np.where(newton_ok, error / np.maximum(vega, 1e-12), 0.0)
        candidate = vol - step

        low = np.where(active & (error < 0), np.maximum(low, vol), low)
        high = np.where(active & (error > 0), np.minimum(high, vol), high)
        outside = (candidate <= low) | (candidate >= high) | ~np.isfinite(candidate)
        candidate = np.where(outside, 0.5 * (low + high), candidate)
        vol = np.where(active, candidate, vol)

    residual = np.abs(black_price(f, k, vol, t, df, right) - p)
    scale = df * np.maximum(f, k)
    vega_at_solution = black_vega(f, k, vol, t, df)
    unidentified = vega_at_solution < 1e-8 * scale
    failed = residual > np.maximum(1e-6, 1e-8 * scale)
    return np.where(unpriceable | failed | unidentified, np.nan, vol)
