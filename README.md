# Aegis — Multi-Asset Risk & P&L Engine

A "mini middle office": ingest real market data, build discount curves and
arbitrage-checked volatility surfaces, price a multi-asset book, compute risk
(Greeks, VaR/ES, stress), validate the VaR against the regulatory backtests, and
explain the daily P&L down to the unexplained residual.

Python for orchestration and modelling; **Rust for the compute kernels**, bound
through PyO3 and shipped as a wheel.

> Status: under construction — see [`PLAN.md`](PLAN.md) for the full design and
> the phase-by-phase build order.

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
