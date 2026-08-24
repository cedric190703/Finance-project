"""Shared plumbing for provider adapters."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import ClassVar

from aegis.marketdata.archive import ArchivedResponse, RawArchive
from aegis.marketdata.http import FetchError, HttpFetcher

__all__ = ["Provider"]


@dataclass
class Provider(ABC):
    """Base adapter: fetch through the archive, or replay from it.

    A provider never touches the network in ``offline`` mode. That is how the
    test suite and CI run: the payloads were captured once, committed as
    fixtures, and are replayed byte for byte thereafter.

    Attributes:
        archive: Where payloads are written to and replayed from.
        fetcher: The HTTP client used when a payload is not already archived.
        offline: When true, a missing payload is an error rather than a fetch.
    """

    name: ClassVar[str] = "provider"

    #: Overridden by adapters whose endpoint refuses the default identity.
    user_agent: ClassVar[str | None] = None

    archive: RawArchive
    fetcher: HttpFetcher = field(default_factory=HttpFetcher)
    offline: bool = False

    @classmethod
    def replaying(cls, fixtures: Path | str, **kwargs: object) -> Provider:
        """Build a provider that only ever replays an archive.

        Args:
            fixtures: Root of the archive to replay.
            **kwargs: Extra keyword arguments for the concrete provider.

        Returns:
            An offline provider instance.
        """
        return cls(archive=RawArchive(fixtures), offline=True, **kwargs)  # type: ignore[arg-type]

    def _payload(
        self,
        endpoint: str,
        url: str,
        params: dict[str, str],
        suffix: str = "json",
        knowledge_date: date | None = None,
    ) -> ArchivedResponse:
        """Return an archived payload, fetching it first if that is allowed.

        Args:
            endpoint: Logical endpoint name, used in the archive key.
            url: Absolute URL to fetch.
            params: Query parameters; also part of the archive key.
            suffix: File extension for the archived body.
            knowledge_date: Replay the newest payload known on or before this
                date. Setting it forces replay: a point-in-time rebuild must not
                be allowed to silently fetch today's revised data.

        Returns:
            The archived response.

        Raises:
            FetchError: in offline or point-in-time mode when nothing is archived.
        """
        cached = self.archive.load(self.name, endpoint, params, knowledge_date=knowledge_date)
        if cached is not None:
            return cached
        if self.offline or knowledge_date is not None:
            raise FetchError(
                f"no archived {self.name}/{endpoint} payload for {params}"
                f"{f' as known on {knowledge_date}' if knowledge_date else ''}"
            )
        body = self.fetcher.get(url, params, user_agent=self.user_agent)
        return self.archive.store(self.name, endpoint, params, body, suffix=suffix)
