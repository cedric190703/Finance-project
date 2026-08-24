# Build the PyO3 extension with the pinned Rust toolchain, then copy only the
# application virtual environment into the small runtime image.
FROM rust:1.82-slim-bookworm AS builder

WORKDIR /app
RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3 python3-venv curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --silent --show-error https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

COPY pyproject.toml uv.lock Cargo.toml Cargo.lock rust-toolchain.toml README.md ./
COPY crates ./crates
COPY python ./python
RUN uv sync --locked --all-extras --no-dev

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AEGIS_DB=/app/data/warehouse/market.duckdb

COPY --from=builder /app/.venv /app/.venv
COPY python ./python
COPY config ./config
COPY README.md ./
RUN useradd --create-home --uid 10001 aegis \
    && mkdir -p /app/data/warehouse /app/data/raw \
    && chown -R aegis:aegis /app
USER aegis

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "aegis.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
