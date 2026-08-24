"""API contracts: health, observability and empty warehouse coverage."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from fastapi import FastAPI
from fastapi.responses import Response

from aegis.api import create_app
from aegis.api.app import HealthResponse


def _endpoint(app: FastAPI, path: str) -> Callable[..., object]:
    """Return a route's endpoint without depending on a test HTTP client."""
    for route in app.routes:
        if getattr(route, "path", None) == path:
            endpoint = getattr(route, "endpoint", None)
            assert callable(endpoint)
            return cast(Callable[..., object], endpoint)
    raise AssertionError(f"route {path} was not registered")


def test_health_and_openapi_are_available(tmp_path: object) -> None:
    path = tmp_path / "warehouse.duckdb"  # type: ignore[operator]
    app = create_app(path)
    health = _endpoint(app, "/health")
    response = health()
    assert isinstance(response, HealthResponse)
    assert response.status == "ok"
    assert "/health" in app.openapi()["paths"]


def test_coverage_and_metrics_are_exposed(tmp_path: object) -> None:
    path = tmp_path / "warehouse.duckdb"  # type: ignore[operator]
    app = create_app(path)
    coverage = _endpoint(app, "/store/coverage")()
    metrics = _endpoint(app, "/metrics")()
    assert isinstance(coverage, list)
    assert isinstance(metrics, Response)
    assert {row["table_name"] for row in coverage} >= {"price_eod", "curve_point"}
    assert metrics.status_code == 200
    assert b"aegis_http_requests_total" in metrics.body
