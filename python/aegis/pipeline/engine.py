"""A small, durable DAG runner for reproducible end-of-day work.

Tasks are keyed by their name, valuation date and a canonical hash of their
parameters.  A successful task with that key is never run twice; a failed task
can be retried without rerunning its successful prerequisites.  The ledger and
lineage records live alongside the bitemporal data in DuckDB.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from hashlib import sha256
from uuid import uuid4

import polars as pl

from aegis.marketdata import MarketStore

__all__ = ["EodPipeline", "PipelineContext", "Task", "TaskOutput", "TaskRun"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR PRIMARY KEY,
    value_date DATE NOT NULL,
    parameters_hash VARCHAR NOT NULL,
    parameters_json VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pipeline_task_runs (
    task_run_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    task_name VARCHAR NOT NULL,
    value_date DATE NOT NULL,
    parameters_hash VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    error_message VARCHAR
);
CREATE TABLE IF NOT EXISTS pipeline_lineage (
    task_run_id VARCHAR NOT NULL,
    input_name VARCHAR NOT NULL,
    output_name VARCHAR NOT NULL,
    recorded_at TIMESTAMP NOT NULL
);
"""


@dataclass(frozen=True)
class TaskOutput:
    """The named inputs and outputs a task used, for data-lineage reporting."""

    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineContext:
    """Inputs available to every task in one end-of-day run."""

    store: MarketStore
    value_date: date
    parameters: Mapping[str, object]
    run_id: str


TaskFunction = Callable[[PipelineContext], TaskOutput]


@dataclass(frozen=True)
class Task:
    """One idempotent node in an end-of-day DAG."""

    name: str
    run: TaskFunction
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskRun:
    """The outcome of one task requested by the pipeline."""

    task_name: str
    status: str
    task_run_id: str | None = None


@dataclass
class EodPipeline:
    """Execute a validated DAG and persist an idempotency ledger."""

    store: MarketStore
    tasks: tuple[Task, ...]
    _by_name: dict[str, Task] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate task names and dependencies, then create ledger tables."""
        self._by_name = {task.name: task for task in self.tasks}
        if len(self._by_name) != len(self.tasks):
            raise ValueError("pipeline task names must be unique")
        unknown = {
            dependency
            for task in self.tasks
            for dependency in task.depends_on
            if dependency not in self._by_name
        }
        if unknown:
            raise ValueError(f"task dependencies are not defined: {sorted(unknown)}")
        self.store._con.execute(_SCHEMA)

    def run(
        self, value_date: date, parameters: Mapping[str, object] | None = None
    ) -> tuple[TaskRun, ...]:
        """Run all tasks in dependency order, skipping already-successful work."""
        supplied = parameters or {}
        encoded = json.dumps(supplied, sort_keys=True, separators=(",", ":"), default=str)
        parameters_hash = sha256(encoded.encode()).hexdigest()
        run_id = str(uuid4())
        now = _now()
        self.store._con.execute(
            "INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?, NULL)",
            [run_id, value_date, parameters_hash, encoded, "running", now],
        )
        context = PipelineContext(self.store, value_date, supplied, run_id)
        outcomes: dict[str, TaskRun] = {}
        try:
            for task in self.tasks:
                self._execute(task, context, parameters_hash, outcomes, active=set())
        except Exception:
            self.store._con.execute(
                "UPDATE pipeline_runs SET status = ?, finished_at = ? WHERE run_id = ?",
                ["failed", _now(), run_id],
            )
            raise
        self.store._con.execute(
            "UPDATE pipeline_runs SET status = ?, finished_at = ? WHERE run_id = ?",
            ["succeeded", _now(), run_id],
        )
        return tuple(outcomes[task.name] for task in self.tasks)

    def lineage(self, value_date: date | None = None) -> pl.DataFrame:
        """Return recorded input-to-output links, optionally for one date."""
        query = """
            SELECT task.value_date, task.task_name, lineage.input_name, lineage.output_name
            FROM pipeline_lineage AS lineage
            JOIN pipeline_task_runs AS task USING (task_run_id)
            ORDER BY task.value_date, task.task_name, lineage.input_name, lineage.output_name
        """
        if value_date is not None:
            query = query.replace("ORDER BY", "WHERE task.value_date = ? ORDER BY")
            return self.store._con.execute(query, [value_date]).pl()
        return self.store._con.execute(query).pl()

    def _execute(
        self,
        task: Task,
        context: PipelineContext,
        parameters_hash: str,
        outcomes: dict[str, TaskRun],
        active: set[str],
    ) -> TaskRun:
        if task.name in outcomes:
            return outcomes[task.name]
        if task.name in active:
            raise ValueError(f"pipeline contains a cycle at task {task.name!r}")
        active.add(task.name)
        for dependency in task.depends_on:
            self._execute(self._by_name[dependency], context, parameters_hash, outcomes, active)
        active.remove(task.name)

        previous = self.store._con.execute(
            """
            SELECT task_run_id FROM pipeline_task_runs
            WHERE task_name = ? AND value_date = ? AND parameters_hash = ? AND status = 'succeeded'
            ORDER BY finished_at DESC LIMIT 1
            """,
            [task.name, context.value_date, parameters_hash],
        ).fetchone()
        if previous:
            outcome = TaskRun(task.name, "skipped", str(previous[0]))
            outcomes[task.name] = outcome
            return outcome

        task_run_id = str(uuid4())
        self.store._con.execute(
            "INSERT INTO pipeline_task_runs VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
            [
                task_run_id,
                context.run_id,
                task.name,
                context.value_date,
                parameters_hash,
                "running",
                _now(),
            ],
        )
        try:
            output = task.run(context)
        except Exception as error:
            self.store._con.execute(
                """
                UPDATE pipeline_task_runs
                SET status = ?, finished_at = ?, error_message = ? WHERE task_run_id = ?
                """,
                ["failed", _now(), str(error), task_run_id],
            )
            raise
        self.store._con.execute(
            "UPDATE pipeline_task_runs SET status = ?, finished_at = ? WHERE task_run_id = ?",
            ["succeeded", _now(), task_run_id],
        )
        for input_name in output.inputs:
            for output_name in output.outputs:
                self.store._con.execute(
                    "INSERT INTO pipeline_lineage VALUES (?, ?, ?, ?)",
                    [task_run_id, input_name, output_name, _now()],
                )
        outcome = TaskRun(task.name, "succeeded", task_run_id)
        outcomes[task.name] = outcome
        return outcome


def _now() -> datetime:
    """Return a naive UTC time for DuckDB timestamp fields."""
    return datetime.now(UTC).replace(tzinfo=None)
