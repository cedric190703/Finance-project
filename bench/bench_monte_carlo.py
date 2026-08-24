"""Benchmark: the same Monte Carlo, four ways.

Run with `uv run python bench/bench_monte_carlo.py`. The numbers it prints are
the ones quoted in the README, so regenerate it rather than editing them by hand.

The comparison is deliberately fair. Every implementation prices the same
option, with the same control variate, to the same path count. The point is not
to show that Rust is faster than a bad Python loop — everyone knows that — but
to show what each step actually buys, and where the interesting win comes from.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass

from aegis.pricing import OptionRight, black_price, monte_carlo_european
from aegis.pricing.reference import numpy_european, python_loop_european

FORWARD, STRIKE, VOL, TIME, DISCOUNT = 100.0, 100.0, 0.25, 1.0, 0.96
RIGHT = OptionRight.CALL
PATHS = 1 << 20
LOOP_PATHS = 1 << 15  # the pure-Python loop is timed on fewer and scaled up


@dataclass(frozen=True)
class Measurement:
    """One timed implementation.

    Attributes:
        label: What was run.
        seconds: Wall time for the full path count.
        price: The price it produced.
        extrapolated: Whether the time was scaled up from a shorter run.
    """

    label: str
    seconds: float
    price: float
    extrapolated: bool = False


def _time(label: str, run: Callable[[], float], repeats: int = 3) -> Measurement:
    best, price = float("inf"), float("nan")
    for _ in range(repeats):
        started = time.perf_counter()
        price = run()
        best = min(best, time.perf_counter() - started)
    return Measurement(label, best, price)


def main() -> None:
    """Run the benchmark and print the comparison table."""
    exact = float(black_price(FORWARD, STRIKE, VOL, TIME, DISCOUNT, RIGHT)[0])

    loop = _time(
        "Python loop",
        lambda: python_loop_european(
            FORWARD, STRIKE, VOL, TIME, DISCOUNT, RIGHT, LOOP_PATHS, seed=1
        ),
        repeats=1,
    )
    loop = Measurement(
        loop.label, loop.seconds * (PATHS / LOOP_PATHS), loop.price, extrapolated=True
    )

    measurements = [
        loop,
        _time(
            "numpy vectorised",
            lambda: numpy_european(FORWARD, STRIKE, VOL, TIME, DISCOUNT, RIGHT, PATHS, seed=1),
        ),
        _time(
            "Rust, pseudo-random",
            lambda: (
                monte_carlo_european(
                    FORWARD, STRIKE, VOL, TIME, DISCOUNT, RIGHT, PATHS, 1, "pseudo"
                ).price
            ),
        ),
        _time(
            "Rust, quasi-random",
            lambda: (
                monte_carlo_european(
                    FORWARD, STRIKE, VOL, TIME, DISCOUNT, RIGHT, PATHS, 1, "quasi"
                ).price
            ),
        ),
    ]

    baseline = measurements[0].seconds
    print(
        f"{platform.processor() or platform.machine()} — {PATHS:,} paths, "
        f"at-the-money one-year call, control variate on"
    )
    print(f"exact price {exact:.6f}\n")
    header = f"{'implementation':<22}{'seconds':>10}{'speedup':>10}{'error':>12}"
    print(header)
    print("-" * len(header))
    for measurement in measurements:
        marker = "*" if measurement.extrapolated else " "
        print(
            f"{measurement.label:<22}"
            f"{measurement.seconds:>9.3f}{marker}"
            f"{baseline / measurement.seconds:>9.0f}x"
            f"{abs(measurement.price - exact):>12.2e}"
        )
    print("\n* extrapolated from a shorter run; the loop would take too long otherwise.")

    # Wall time at equal path count is the boring half of the comparison, because
    # the implementations are not equally accurate at equal path count. What a
    # desk actually cares about is how long it takes to reach a given accuracy,
    # and there the sampler matters more than the language.
    print("\nStandard error at equal path count:")
    errors = {}
    for sampler in ("pseudo", "quasi"):
        result = monte_carlo_european(
            FORWARD, STRIKE, VOL, TIME, DISCOUNT, RIGHT, PATHS, 1, sampler
        )
        errors[sampler] = result.standard_error
        print(f"  {sampler:<8} {result.standard_error:.3e}  ({result})")

    numpy_seconds = measurements[1].seconds
    quasi_seconds = measurements[3].seconds
    # Plain Monte Carlo converges at 1/sqrt(n), so matching an error k times
    # smaller costs k^2 times the paths.
    ratio = (errors["pseudo"] / errors["quasi"]) ** 2
    print(
        f"\nTo match the quasi-random error bar, a pseudo-random run needs {ratio:,.0f}x the paths."
    )
    print(
        f"Same accuracy, wall time: {quasi_seconds * 1e3:.1f} ms here against "
        f"roughly {numpy_seconds * ratio:,.0f} s for the vectorised numpy version "
        f"— a factor of {numpy_seconds * ratio / quasi_seconds:,.0f}."
    )


if __name__ == "__main__":
    main()
