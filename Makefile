.PHONY: help install build test lint fmt typecheck rust-test rust-lint bench check clean

PYTHON := uv run

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install everything, including the Rust core
	uv sync --all-extras

build:  ## Rebuild the Rust extension module in place
	uv run maturin develop --release --uv

test:  ## Run the Python test suite
	$(PYTHON) pytest

lint:  ## Lint Python
	$(PYTHON) ruff check python tests bench
	$(PYTHON) ruff format --check python tests bench

fmt:  ## Format Python and Rust
	$(PYTHON) ruff format python tests bench
	$(PYTHON) ruff check --fix python tests bench
	cargo fmt

typecheck:  ## Type-check Python
	$(PYTHON) mypy

rust-test:  ## Run the Rust unit tests
	cargo test --workspace

rust-lint:  ## Clippy with warnings as errors
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --check

bench:  ## Run the Rust criterion benchmarks
	cargo bench --workspace

check: lint typecheck rust-lint test rust-test  ## Everything CI runs

clean:
	cargo clean
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
