# Architecture

Aegis is a reproducible, multi-asset middle-office engine. Its boundary is
deliberate: providers are allowed to write immutable raw payloads; all pricing
and risk calculations consume a point-in-time market snapshot rather than a
network connection or mutable global state.

```text
providers → raw archive → bitemporal DuckDB → market snapshot → pricing/risk
                                                 │              │
                                                 │              ├─ VaR validation
                                                 │              ├─ P&L explain
                                                 │              └─ EOD run ledger
                                                 ▼
                                          FastAPI /metrics / OpenAPI
                                                 │
                                            Streamlit dashboard
```

## Data and reproducibility

Each warehouse observation carries `value_date` (the date it describes) and
`knowledge_date` (when it was learned). `MarketStore.as_of` selects the most
recent observation known at a requested point in knowledge time. This prevents
a historical calculation from silently using later data revisions.

The raw archive stores the unparsed provider response before it reaches the
warehouse. Tests replay those captured responses, so provider formatting changes
and rate limits do not make the suite non-deterministic.

## Compute boundary

The Python package owns the business model, bitemporal reads, report assembly,
and API contracts. `aegis-core` owns the numerical Monte Carlo kernels. PyO3
binds the Rust core as `aegis._core`; the Python-facing engine never relies on a
network call while pricing.

## Operational model

`EodPipeline` records each task in the warehouse with a value date and a
canonical parameter hash. Successful tasks with the same key are skipped on a
retry; failed tasks are retried while successful dependencies remain skipped.
Every task records named inputs and outputs in the lineage table.

The API has no authentication because the supplied compose configuration is a
local/single-trusted-network deployment. Put it behind the organisation's
identity-aware proxy before exposing it beyond that boundary.
