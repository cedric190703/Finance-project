# Aegis — Multi-Asset Risk & P&L Engine

> A "mini middle office": ingest real market data, build curves and vol surfaces,
> price a multi-asset book, compute risk (Greeks, VaR/ES, stress), backtest the VaR
> against regulatory tests, and explain daily P&L down to the residual.
> Python for orchestration and modelling, Rust for the compute kernels.

Target signal: **quant developer / trading tech** + **data engineer in finance**.
So the depth goes into architecture, performance, correctness and data lineage —
with enough genuine financial content that a desk quant recognises it as real work.

---

## 1. What makes this stand out

Three things almost no portfolio project has:

1. **P&L explain** — decomposing one day's P&L into delta / gamma / vega / theta /
   carry / FX / unexplained. Every risk desk produces this daily; nobody builds it
   for a portfolio piece.
2. **VaR backtesting with regulatory tests** — Kupiec POF, Christoffersen
   independence, Basel traffic-light zones. Shows awareness that models get
   validated, not just written.
3. **Bitemporal, point-in-time market data** — every observation carries both a
   *value date* and a *knowledge date*, so a valuation can be reproduced exactly as
   it was seen on any past day. This is the single hardest thing to get right in
   financial data engineering, and the one that most impresses a data-platform team.

Plus the engineering: a Rust core behind PyO3, property-based tests, benchmark
regression gates in CI, and reproducible Docker builds.

---

## 2. Architecture

```
                 ┌──────────── providers (free, no API keys) ─────────────┐
                 │  Yahoo Finance   FRED (UST curve)   ECB SDMX (FX)      │
                 └───────────────────────┬───────────────────────────────┘
                                         │  raw immutable landing zone
                                         ▼  (parquet, one file per fetch)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  Ingestion layer  — retry/backoff, rate limiting, response caching,   │
   │  schema validation, replay-from-cache for offline CI                  │
   └───────────────────────┬──────────────────────────────────────────────┘
                           ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  DuckDB warehouse — bitemporal fact tables (value_date, knowledge_date)│
   │  prices · yields · fx · option_quotes · dividends · corporate_actions  │
   └───────────────────────┬──────────────────────────────────────────────┘
                           ▼
   ┌───────────────┐  ┌─────────────────┐  ┌────────────────────────────┐
   │ Curve builder │  │ Vol surface     │  │ Instrument / portfolio     │
   │ (bootstrap)   │  │ (SVI, no-arb)   │  │ model (pydantic)           │
   └───────┬───────┘  └────────┬────────┘  └─────────────┬──────────────┘
           └───────────────────┴─────────────────────────┘
                                         ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  Pricing engine   Python API  ──►  aegis._core  (Rust, PyO3)          │
   │  analytic (BS, Black-76, bond PV) · Monte Carlo (antithetic + CV)     │
   │  · full-revaluation risk kernels · parallel via rayon                 │
   └───────────────────────┬──────────────────────────────────────────────┘
                           ▼
   ┌──────────────┬──────────────┬──────────────┬────────────────────────┐
   │ Greeks       │ VaR / ES     │ Stress       │ P&L explain            │
   │ analytic +   │ hist / param │ 2008, COVID, │ delta·gamma·vega·theta │
   │ bump-reval   │ / Monte Carlo│ rates shock  │ ·carry·fx·unexplained  │
   └──────────────┴──────┬───────┴──────────────┴────────────────────────┘
                         ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  FastAPI service (async, OpenAPI, Prometheus metrics)                 │
   │  Streamlit dashboard · CLI (`aegis run eod --date …`)                 │
   └──────────────────────────────────────────────────────────────────────┘
```

### Repo layout

```
Finance-project/
├── Cargo.toml                    # Rust workspace
├── crates/
│   ├── aegis-core/               # pure Rust: kernels, no Python dependency
│   │   ├── src/{mc.rs,bs.rs,paths.rs,rng.rs,risk.rs,lib.rs}
│   │   └── benches/mc.rs         # criterion benchmarks
│   └── aegis-py/                 # PyO3 bindings → module `aegis._core`
├── pyproject.toml                # maturin backend, uv-managed
├── python/aegis/
│   ├── conventions/              # day-count, business-day calendars, roll rules
│   ├── marketdata/               # providers, cache, bitemporal store, loaders
│   ├── curves/                   # bootstrap, interpolation, discount factors
│   ├── vol/                      # SVI calibration, arbitrage checks, interp
│   ├── instruments/              # equity, fx, bond, european/american option
│   ├── pricing/                  # engine dispatch, analytic, MC (→ Rust)
│   ├── risk/                     # greeks, var, es, stress, backtest
│   ├── pnl/                      # attribution / explain
│   ├── portfolio/                # book, positions, valuation run
│   ├── pipeline/                 # task DAG, run ledger, idempotent EOD job
│   ├── api/                      # FastAPI routers + schemas
│   ├── dashboard/                # Streamlit app
│   └── cli.py                    # typer CLI
├── tests/                        # pytest + hypothesis + golden files
├── bench/                        # pytest-benchmark, perf regression gate
├── docs/                         # architecture.md, methodology.md, adr/
├── docker/                       # Dockerfile (multi-stage), compose.yml
└── .github/workflows/ci.yml
```

---

## 3. Data sources (free, no keys — verified reachable)

| Source | What | Notes |
|---|---|---|
| Yahoo Finance chart API | equity/ETF OHLCV, dividends, splits | needs a browser User-Agent; 429s without it |
| Yahoo options API | option chains → implied vols for the surface | sparse but real |
| FRED `fredgraph.csv` | US Treasury constant-maturity yields DGS1MO…DGS30 | CSV, no key |
| ECB SDMX | EUR FX reference rates, EUR yield curve | CSV, no key |
| CoinGecko | optional crypto leg for a 24/7 asset | no key |

Every response is written verbatim to `data/raw/<source>/<date>/<hash>.json|csv`
before parsing. Tests and CI replay from that cache, so the suite is deterministic
and runs offline — real data, reproducible pipeline.

---

## 4. Financial content, concretely

- **Conventions**: ACT/360, ACT/365F, 30/360, ACT/ACT ISDA; TARGET and NYSE
  calendars; modified-following roll.
- **Curve**: bootstrap zero rates from the Treasury par curve, monotone-convex or
  log-linear discount-factor interpolation, forward-rate extraction.
- **Vol surface**: SVI slice calibration per expiry, Gatheral butterfly and
  calendar-spread arbitrage checks, interpolation in total variance.
- **Pricing**: Black-Scholes with dividends, Black-76, bond dirty/clean price with
  accrued, Monte Carlo GBM with antithetic variates + control variate; MC is
  asserted against the closed form inside the test suite.
- **Greeks**: analytic where available, bump-and-revalue everywhere, reconciled to
  tolerance in tests; DV01, duration, convexity for bonds.
- **VaR / ES**: historical (with EWMA-weighted variant), parametric (variance-
  covariance, Cornish-Fisher option), Monte Carlo full revaluation; 1-day and
  10-day, 97.5% ES alongside 99% VaR.
- **Stress**: replay 2008-09, Mar-2020, a +200bp parallel rates shock, a vol
  spike — as scenario definitions in YAML, applied to the live book.
- **Backtesting**: Kupiec unconditional coverage, Christoffersen independence and
  conditional coverage, Basel green/amber/red traffic light.
- **P&L explain**: Taylor decomposition on the risk factors, with the residual
  reported explicitly — a good explain has a small unexplained bucket, and the
  dashboard shows it as a waterfall chart.

---

## 5. Engineering content, concretely

- **Rust core** (`aegis-core`): Monte Carlo path generation and payoff evaluation,
  full-revaluation VaR loop, rayon parallelism, xoshiro256++ RNG with Sobol
  quasi-random option, `#[no_std]`-friendly numerics. Criterion benchmarks in-crate.
- **Bindings** (`aegis-py`): PyO3, zero-copy numpy in/out via `numpy` crate,
  GIL released around the compute, typed `.pyi` stubs.
- **Performance story**: naive Python loop → vectorised numpy → Rust single-thread
  → Rust + rayon, benchmarked and charted in the README. Expect ~2-3 orders of
  magnitude end to end; the numbers get measured, not claimed.
- **Data engineering**: bitemporal store, idempotent tasks keyed on
  `(task, value_date, params_hash)`, a run ledger table, restart-from-failure,
  and a lineage view (which raw file fed which valuation).
- **Testing**: pytest + Hypothesis property tests (put-call parity, price monotone
  in vol, discount factors ≤ 1 and decreasing, VaR monotone in confidence),
  golden-file tests for the P&L report, `cargo test` + `cargo clippy -D warnings`.
- **Quality gates in CI**: ruff, mypy --strict, pytest with coverage floor,
  clippy, `cargo test`, maturin wheel build on Linux+macOS, and a benchmark
  regression check that fails the build on >15% slowdown.
- **Ops**: multi-stage Dockerfile (Rust builder → slim runtime), compose file for
  API + dashboard, structured JSON logging, Prometheus `/metrics`, health probes.
- **Docs**: `architecture.md` with the diagram above, `methodology.md` with the
  actual formulas and their assumptions, and ADRs recording why DuckDB, why SVI,
  why bitemporal.

---

## 6. Phases

Each phase ends in something demoable and committed separately, so the git history
itself reads like a professional project.

| # | Phase | Deliverable |
|---|---|---|
| 0 | Scaffolding | uv + maturin mixed project, Rust workspace, ruff/mypy/pytest/clippy, CI green, Makefile |
| 1 | Conventions | day-count, calendars, roll rules + property tests |
| 2 | Market data | providers, raw cache, bitemporal DuckDB store, `aegis fetch` CLI |
| 3 | Curves | bootstrap from FRED par yields, interpolation, discount factors |
| 4 | Vol surface | option-chain parsing, SVI calibration, arbitrage checks, plots |
| 5 | Rust kernels | MC engine, RNG, rayon, criterion benches, PyO3 bindings, wheel in CI |
| 6 | Pricing | instruments, analytic pricers, engine dispatch, MC-vs-analytic tests |
| 7 | Portfolio & risk | book model, Greeks, VaR/ES, stress scenarios |
| 8 | Validation | VaR backtesting (Kupiec / Christoffersen / Basel) |
| 9 | P&L explain | attribution engine, waterfall, residual analysis |
| 10 | Pipeline | task DAG, run ledger, idempotent EOD job, lineage view |
| 11 | Service | FastAPI + OpenAPI + metrics, Streamlit dashboard |
| 12 | Polish | Docker, benchmark chart, README, methodology, ADRs, demo GIF |

---

## 7. Decisions taken (say if you disagree)

- **Rust over C++** for the hot path — maturin/PyO3 gives clean wheels and CI, and
  it is the stronger signal for modern trading tech. Requires installing `rustup`
  (one command, I will ask before running it). C++20 + nanobind is the fallback if
  you would rather not add a toolchain.
- **DuckDB over Postgres** — embedded, no service to run, excellent parquet and
  analytical performance. A compose Postgres profile can be added later.
- **Streamlit over React** — you chose the Rust hot path over a React frontend, so
  the UI stays functional rather than bespoke.
- **Data cached to disk** — free APIs only for live ingestion, but every response is
  archived so CI and tests never depend on the network. Real data, deterministic runs.
- **Equity / FX / rates / vanilla options** in scope. No exotics, no XVA, no
  American MC — the depth goes into the pipeline and the risk layer instead.
