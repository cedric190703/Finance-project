"""HTTP service: explicit inputs, documented output and Prometheus metrics."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from aegis import __version__
from aegis.marketdata import MarketStore
from aegis.pipeline import EodPipeline
from aegis.pnl import explain_pnl
from aegis.portfolio import Portfolio
from aegis.risk import build_market, run_report

__all__ = ["create_app"]

_REQUESTS = Counter("aegis_http_requests_total", "HTTP requests", ["method", "path", "status"])
_LATENCY = Histogram(
    "aegis_http_request_duration_seconds", "HTTP request duration", ["method", "path"]
)


class HealthResponse(BaseModel):
    """Service liveness and warehouse reachability."""

    status: str
    version: str
    warehouse: str


class RiskResponse(BaseModel):
    """The API representation of a daily risk report."""

    value_date: date
    portfolio_value: float
    greeks: dict[str, float]
    var: list[dict[str, object]]
    stress: dict[str, float]


class PnlResponse(BaseModel):
    """The API representation of a daily P&L waterfall."""

    opening_value: float
    closing_value: float
    total_pnl: float
    unexplained_pnl: float
    components: list[dict[str, object]]


def create_app(db_path: Path | str | None = None) -> FastAPI:
    """Build the API application.

    Args:
        db_path: DuckDB warehouse location. Defaults to ``AEGIS_DB`` or the
            local project warehouse; this is intentionally configuration, not a
            hard-coded production endpoint.
    """
    warehouse = str(db_path or os.environ.get("AEGIS_DB", "data/warehouse/market.duckdb"))
    app = FastAPI(
        title="Aegis Risk API",
        version=__version__,
        description="Reproducible multi-asset risk, validation and P&L reporting.",
    )

    @app.middleware("http")
    async def instrument_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Record method, route and duration without ever logging request bodies."""
        started = perf_counter()
        response = await call_next(request)
        path = request.url.path
        _REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        _LATENCY.labels(request.method, path).observe(perf_counter() - started)
        return response

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        """Return service and warehouse liveness."""
        try:
            with MarketStore(warehouse) as store:
                store.coverage()
        except Exception as error:  # pragma: no cover - host filesystem failure
            raise HTTPException(status_code=503, detail="warehouse unavailable") from error
        return HealthResponse(status="ok", version=__version__, warehouse="reachable")

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Expose Prometheus metrics."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/store/coverage", tags=["market-data"])
    def store_coverage() -> list[dict[str, object]]:
        """Return bitemporal warehouse coverage by logical table."""
        with MarketStore(warehouse) as store:
            return store.coverage().to_dicts()

    @app.get("/pipeline/lineage", tags=["operations"])
    def pipeline_lineage(value_date: date | None = None) -> list[dict[str, object]]:
        """Show raw-input to report-output lineage recorded by the EOD runner."""
        with MarketStore(warehouse) as store:
            return EodPipeline(store, ()).lineage(value_date).to_dicts()

    @app.get("/risk/report", response_model=RiskResponse, tags=["risk"])
    def risk_report(
        value_date: date,
        portfolio_path: Path = Path("config/portfolio.yaml"),
        lookback_days: int = 730,
        confidence: float = 0.99,
    ) -> RiskResponse:
        """Build a full daily risk report from the point-in-time warehouse."""
        try:
            book = Portfolio.from_yaml(portfolio_path)
            with MarketStore(warehouse) as store:
                report = run_report(store, book, value_date, lookback_days, confidence)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return RiskResponse(
            value_date=report.value_date,
            portfolio_value=report.portfolio_value,
            greeks=report.greeks,
            var=[
                {
                    "method": str(result.method),
                    "var": result.var,
                    "expected_shortfall": result.expected_shortfall,
                    "confidence": result.confidence,
                }
                for result in report.var_results
            ],
            stress=report.stress,
        )

    @app.get("/pnl/explain", response_model=PnlResponse, tags=["pnl"])
    def pnl_explain(
        start_date: date,
        end_date: date,
        portfolio_path: Path = Path("config/portfolio.yaml"),
    ) -> PnlResponse:
        """Return a daily P&L waterfall, including the reconciliation residual."""
        try:
            book = Portfolio.from_yaml(portfolio_path)
            with MarketStore(warehouse) as store:
                opening = build_market(store, start_date, knowledge_date=start_date)
                closing = build_market(store, end_date, knowledge_date=end_date)
            explain = explain_pnl(book, opening, closing)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return PnlResponse(
            opening_value=explain.opening_value,
            closing_value=explain.closing_value,
            total_pnl=explain.total_pnl,
            unexplained_pnl=explain.unexplained_pnl,
            components=explain.components.to_dicts(),
        )

    return app
