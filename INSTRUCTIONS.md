# Aegis contributor guide

Use Python 3.11+ with `uv`, and Rust 1.82+ for the PyO3 extension. The normal
local gates are `make check` and `uv run pytest -m "not network"`.

Keep valuation inputs explicit through `MarketSnapshot`; do not fetch data from
pricing or risk code. Warehouse facts are append-only and must retain both
value and knowledge dates. Do not add real credentials or raw production data
to the repository.
