"""The end-of-day DAG is restartable, idempotent and lineage-aware."""

from __future__ import annotations

from datetime import date

import pytest

from aegis.marketdata import MarketStore
from aegis.pipeline import EodPipeline, Task, TaskOutput


def test_pipeline_runs_dependencies_once_and_records_lineage() -> None:
    calls: list[str] = []

    def ingest(_: object) -> TaskOutput:
        calls.append("ingest")
        return TaskOutput(inputs=("raw/fred.csv",), outputs=("curve_point",))

    def report(_: object) -> TaskOutput:
        calls.append("report")
        return TaskOutput(inputs=("curve_point",), outputs=("risk_report",))

    with MarketStore() as store:
        pipeline = EodPipeline(store, (Task("ingest", ingest), Task("report", report, ("ingest",))))
        first = pipeline.run(date(2026, 8, 24), {"book": "demo"})
        second = pipeline.run(date(2026, 8, 24), {"book": "demo"})
        lineage = pipeline.lineage(date(2026, 8, 24))

    assert [outcome.status for outcome in first] == ["succeeded", "succeeded"]
    assert [outcome.status for outcome in second] == ["skipped", "skipped"]
    assert calls == ["ingest", "report"]
    assert lineage.to_dicts() == [
        {
            "value_date": date(2026, 8, 24),
            "task_name": "ingest",
            "input_name": "raw/fred.csv",
            "output_name": "curve_point",
        },
        {
            "value_date": date(2026, 8, 24),
            "task_name": "report",
            "input_name": "curve_point",
            "output_name": "risk_report",
        },
    ]


def test_pipeline_retries_failed_tasks_without_repeating_successes() -> None:
    calls: list[str] = []
    fail = True

    def source(_: object) -> TaskOutput:
        calls.append("source")
        return TaskOutput(outputs=("source",))

    def flaky(_: object) -> TaskOutput:
        nonlocal fail
        calls.append("flaky")
        if fail:
            fail = False
            raise RuntimeError("temporary provider outage")
        return TaskOutput(inputs=("source",), outputs=("report",))

    with MarketStore() as store:
        pipeline = EodPipeline(store, (Task("source", source), Task("flaky", flaky, ("source",))))
        with pytest.raises(RuntimeError, match="temporary provider outage"):
            pipeline.run(date(2026, 8, 24))
        retry = pipeline.run(date(2026, 8, 24))

    assert calls == ["source", "flaky", "flaky"]
    assert [outcome.status for outcome in retry] == ["skipped", "succeeded"]


def test_pipeline_rejects_unknown_dependencies_and_cycles() -> None:
    with MarketStore() as store:
        with pytest.raises(ValueError, match="not defined"):
            EodPipeline(store, (Task("report", lambda _: TaskOutput(), ("missing",)),))

        pipeline = EodPipeline(
            store,
            (
                Task("one", lambda _: TaskOutput(), ("two",)),
                Task("two", lambda _: TaskOutput(), ("one",)),
            ),
        )
        with pytest.raises(ValueError, match="cycle"):
            pipeline.run(date(2026, 8, 24))
