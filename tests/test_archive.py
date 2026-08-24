"""The raw payload archive."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aegis.marketdata import RawArchive, request_key


def test_request_key_ignores_parameter_order() -> None:
    a = request_key("fred", "treasury", {"id": "DGS10", "cosd": "2024-01-01"})
    b = request_key("fred", "treasury", {"cosd": "2024-01-01", "id": "DGS10"})
    assert a == b


def test_request_key_separates_different_requests() -> None:
    assert request_key("fred", "treasury", {"id": "DGS10"}) != request_key(
        "fred", "treasury", {"id": "DGS30"}
    )
    assert request_key("fred", "treasury", {"id": "DGS10"}) != request_key(
        "ecb", "treasury", {"id": "DGS10"}
    )


def test_store_then_load_round_trips(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path)
    stored = archive.store("fred", "treasury", {"id": "DGS10"}, b"date,value\n", suffix="csv")
    loaded = archive.load("fred", "treasury", {"id": "DGS10"})

    assert loaded is not None
    assert loaded.body == b"date,value\n"
    assert loaded.content_sha256 == stored.content_sha256
    assert loaded.text() == "date,value\n"


def test_loading_an_unknown_request_returns_none(tmp_path: Path) -> None:
    assert RawArchive(tmp_path).load("fred", "treasury", {"id": "NOPE"}) is None


def test_a_manifest_is_written_alongside_every_payload(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path)
    archive.store("cboe", "chain", {"symbol": "KO"}, b'{"data": {}}')
    entries = archive.entries("cboe")

    assert len(entries) == 1
    assert entries[0]["source"] == "cboe"
    assert entries[0]["bytes"] == len(b'{"data": {}}')


def test_load_returns_the_newest_payload(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path)
    params = {"symbol": "KO"}
    archive.store("cboe", "chain", params, b"old", fetched_at=datetime(2024, 1, 1, tzinfo=UTC))
    archive.store("cboe", "chain", params, b"new", fetched_at=datetime(2024, 6, 1, tzinfo=UTC))

    latest = archive.load("cboe", "chain", params)
    assert latest is not None
    assert latest.body == b"new"


def test_point_in_time_load_ignores_payloads_fetched_later(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path)
    params = {"symbol": "KO"}
    archive.store("cboe", "chain", params, b"old", fetched_at=datetime(2024, 1, 1, tzinfo=UTC))
    archive.store("cboe", "chain", params, b"new", fetched_at=datetime(2024, 6, 1, tzinfo=UTC))

    replayed = archive.load(
        "cboe", "chain", params, knowledge_date=datetime(2024, 3, 1, tzinfo=UTC).date()
    )
    assert replayed is not None
    assert replayed.body == b"old"
