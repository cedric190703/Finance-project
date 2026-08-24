# Repository conventions

- Keep an independently reviewable feature on one branch and test it before merge.
- Use `ruff`, strict `mypy`, `pytest`, `cargo fmt --check`, `cargo clippy`, and
  `cargo test` as appropriate to a change.
- Preserve immutable `MarketSnapshot`, portfolio, curve, and surface objects;
  return a replacement instead of mutating them.
- Validate external data at its provider/store boundary and fail explicitly for
  missing market data. Never substitute plausible default prices or secrets.
- Commit messages follow `Phase N: concise description` for roadmap slices.
