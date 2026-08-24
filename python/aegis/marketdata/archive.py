"""The raw response archive.

Every byte a provider returns is written to disk before anything parses it. Two
reasons, both of which come up in a real data platform:

* **Reproducibility.** A valuation that disagrees with yesterday's can be traced
  back to the exact payload it was built from, not to a re-fetch that may since
  have been revised.
* **Offline determinism.** Tests and CI replay the archive instead of hitting the
  network, so the suite gives the same answer on a plane as it does in a runner.

Files are content-addressed by a hash of the request, so re-fetching the same
request on the same day is idempotent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

__all__ = ["ArchivedResponse", "RawArchive", "request_key"]


def request_key(source: str, endpoint: str, params: dict[str, str]) -> str:
    """Return the content-addressed key identifying a request.

    Args:
        source: Provider name, e.g. ``"yahoo"``.
        endpoint: Logical endpoint, e.g. ``"chart"``.
        params: Request parameters; order does not matter.

    Returns:
        A 16-character hex digest, short enough to read in a filename.
    """
    payload = json.dumps(
        {"source": source, "endpoint": endpoint, "params": dict(sorted(params.items()))},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ArchivedResponse:
    """A payload as it came off the wire, together with its provenance.

    Attributes:
        key: The request key this payload answers.
        source: Provider name.
        endpoint: Logical endpoint.
        params: The request parameters.
        body: The raw response bytes.
        fetched_at: When the payload was retrieved.
        content_sha256: Digest of ``body``, so tampering is detectable.
    """

    key: str
    source: str
    endpoint: str
    params: dict[str, str]
    body: bytes
    fetched_at: datetime
    content_sha256: str

    @property
    def knowledge_date(self) -> date:
        """Return the date on which this payload became known to us."""
        return self.fetched_at.date()

    def text(self) -> str:
        """Return the payload decoded as UTF-8."""
        return self.body.decode("utf-8")

    def json(self) -> object:
        """Return the payload parsed as JSON."""
        return json.loads(self.body)


class RawArchive:
    """An append-only store of raw provider payloads on the local filesystem.

    Attributes:
        root: Directory the archive is rooted at.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, source: str, key: str, fetched_on: date, suffix: str) -> Path:
        """Return the file path a payload is stored at.

        Args:
            source: Provider name.
            key: Request key.
            fetched_on: The date the payload was retrieved.
            suffix: File extension without the dot, e.g. ``"json"``.

        Returns:
            The absolute path of the payload file.
        """
        return self.root / source / fetched_on.isoformat() / f"{key}.{suffix}"

    def store(
        self,
        source: str,
        endpoint: str,
        params: dict[str, str],
        body: bytes,
        suffix: str = "json",
        fetched_at: datetime | None = None,
    ) -> ArchivedResponse:
        """Write a payload and its manifest to the archive.

        Args:
            source: Provider name.
            endpoint: Logical endpoint.
            params: Request parameters.
            body: Raw response bytes.
            suffix: File extension without the dot.
            fetched_at: Retrieval timestamp; defaults to now in UTC.

        Returns:
            The archived response, including its digest.
        """
        stamp = fetched_at or datetime.now(UTC)
        key = request_key(source, endpoint, params)
        target = self.path_for(source, key, stamp.date(), suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)

        digest = hashlib.sha256(body).hexdigest()
        manifest = {
            "key": key,
            "source": source,
            "endpoint": endpoint,
            "params": params,
            "fetched_at": stamp.isoformat(),
            "content_sha256": digest,
            "bytes": len(body),
            "file": target.name,
        }
        target.with_suffix(f".{suffix}.meta.json").write_text(json.dumps(manifest, indent=2))

        return ArchivedResponse(key, source, endpoint, params, body, stamp, digest)

    def load(
        self,
        source: str,
        endpoint: str,
        params: dict[str, str],
        knowledge_date: date | None = None,
    ) -> ArchivedResponse | None:
        """Replay a previously archived payload, if one exists.

        Args:
            source: Provider name.
            endpoint: Logical endpoint.
            params: Request parameters.
            knowledge_date: Return the newest payload known on or before this
                date. ``None`` returns the newest payload of all — which is what
                a normal fetch wants, and what a point-in-time rebuild must not use.

        Returns:
            The archived response, or ``None`` when the request was never made.
        """
        key = request_key(source, endpoint, params)
        candidates = sorted(self.root.glob(f"{source}/*/{key}.*.meta.json"))
        best: ArchivedResponse | None = None
        for manifest_path in candidates:
            manifest = json.loads(manifest_path.read_text())
            fetched_at = datetime.fromisoformat(manifest["fetched_at"])
            if knowledge_date is not None and fetched_at.date() > knowledge_date:
                continue
            body_path = manifest_path.parent / str(manifest["file"])
            if not body_path.exists():  # pragma: no cover - manifest without payload
                continue
            candidate = ArchivedResponse(
                key=key,
                source=source,
                endpoint=endpoint,
                params=params,
                body=body_path.read_bytes(),
                fetched_at=fetched_at,
                content_sha256=str(manifest["content_sha256"]),
            )
            if best is None or candidate.fetched_at > best.fetched_at:
                best = candidate
        return best

    def entries(self, source: str | None = None) -> list[dict[str, object]]:
        """List archive manifests, newest first.

        Args:
            source: Restrict to one provider, or ``None`` for all of them.

        Returns:
            The manifests as dictionaries.
        """
        pattern = f"{source or '*'}/*/*.meta.json"
        manifests = [json.loads(p.read_text()) for p in self.root.glob(pattern)]
        return sorted(manifests, key=lambda m: str(m["fetched_at"]), reverse=True)
