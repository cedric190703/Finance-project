# Aegis — Multi-Asset Risk & P&L Engine

A "mini middle office": ingest real market data, build discount curves and
arbitrage-checked volatility surfaces, price a multi-asset book, compute risk
(Greeks, VaR/ES, stress), validate the VaR against the regulatory backtests, and
explain the daily P&L down to the unexplained residual.

Python for orchestration and modelling; **Rust for the compute kernels**, bound
through PyO3 and shipped as a wheel.

> Status: under construction — see [`PLAN.md`](PLAN.md) for the full design and
> the phase-by-phase build order.

## The compiled core

The Monte Carlo engine lives in Rust behind PyO3. Same option, same control
variate, same path count, on an M-series laptop:

| implementation      | seconds | speedup | error vs closed form |
| ------------------- | ------: | ------: | -------------------: |
| Python loop         |  0.226* |      1x |             7.13e-02 |
| numpy vectorised    |   0.010 |     24x |             2.24e-02 |
| Rust, pseudo-random |   0.004 |     53x |             2.76e-03 |
| Rust, quasi-random  |   0.002 |    127x |             9.08e-05 |

<sub>* extrapolated from a shorter run. 2<sup>20</sup> paths, at-the-money
one-year call. Regenerate with `uv run python bench/bench_monte_carlo.py`.</sub>

The wall-clock column is the least interesting one. The randomised Sobol
sequence reports a standard error of 7.9e-5 against pseudo-random's 7.1e-3 at
the same path count, and plain Monte Carlo converges at 1/√n — so matching that
error bar the other way costs **8,200x the paths**. Equal accuracy, equal
machine: 1.8 ms here, or about 79 seconds for the vectorised numpy version.

Three properties the engine is built around:

- **Bit-identical across machines.** Work is chunked by a fixed size seeded from
  the run seed and chunk index, not split across available cores, so one thread
  and ninety-six produce the same number. A risk figure that changes with the
  box it ran on cannot be reconciled.
- **Every price carries an error bar**, and the interval is built with a
  Student-t quantile when it rests on a handful of randomised replicas rather
  than pretending 1.96 was earned.
- **Variance reductions are measured, not stacked.** Antithetic sampling on top
  of a control variate makes pseudo-random results *worse*; under a
  low-discrepancy sequence it makes them better. Both facts are in the test
  suite so neither can silently reverse.

## Quick start

```bash
make install     # uv venv + deps + compiles the Rust core
make check       # lint, types, clippy, both test suites
uv run aegis version
```

## Layout

| Path | What lives there |
| --- | --- |
| `crates/aegis-core` | Pure-Rust numerical kernels (no Python dependency) |
| `crates/aegis-py` | PyO3 bindings → the `aegis._core` extension module |
| `python/aegis` | The engine: conventions, market data, curves, pricing, risk, P&L |
| `tests` | pytest + Hypothesis property tests |
| `bench` | Python-side benchmarks; Rust benchmarks live in `crates/*/benches` |
| `docs` | Architecture, methodology, and architecture decision records |

## Licence

MIT
