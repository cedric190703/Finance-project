"""A small HTTP client with the manners free market-data endpoints expect.

Requests are spaced by a minimum interval and back off exponentially on
throttling or a server error.

The user agent is deliberately per-provider, because the endpoints disagree
about what they want to see. FRED serves an honestly identified client happily
but tarpits anything claiming to be Chrome — the TLS fingerprint gives the lie
away and its bot mitigation simply stops responding. Yahoo does the opposite and
returns ``429`` to anything that looks like a script. Hard-coding one answer
breaks half the sources, so each adapter declares its own.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import httpx

__all__ = ["BROWSER_USER_AGENT", "DEFAULT_USER_AGENT", "FetchError", "HttpFetcher", "RetryPolicy"]

#: Identifies the engine honestly. The default, and what most sources prefer.
DEFAULT_USER_AGENT = "aegis-risk-engine/0.1 (+https://github.com/cbrzyski/aegis)"

#: Only for endpoints that refuse non-browser clients outright.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """Raised when a provider request fails after exhausting its retries."""


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter.

    Attributes:
        attempts: Total number of tries, including the first.
        backoff: Base delay in seconds; attempt *n* waits ``backoff * 2**n``.
        max_backoff: Cap on any single wait.
        min_interval: Minimum spacing between two requests to the same host.
    """

    attempts: int = 4
    backoff: float = 0.5
    max_backoff: float = 8.0
    min_interval: float = 0.25

    def delay(self, attempt: int) -> float:
        """Return the wait before a given retry, with jitter applied.

        Args:
            attempt: Zero-based index of the retry about to be made.

        Returns:
            The delay in seconds.
        """
        raw = min(self.backoff * (2**attempt), self.max_backoff)
        jitter = 0.5 + random.random() / 2  # noqa: S311 - jitter, not cryptography
        return float(raw * jitter)


@dataclass
class HttpFetcher:
    """Fetches URLs with retries, throttling and a browser user agent.

    Attributes:
        policy: The retry and throttling policy.
        timeout: Per-request timeout in seconds.
        user_agent: Default identity sent unless a caller overrides it.
    """

    policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = 30.0
    user_agent: str = DEFAULT_USER_AGENT
    _last_request_at: float = field(default=0.0, init=False, repr=False)

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        user_agent: str | None = None,
    ) -> bytes:
        """Fetch a URL and return its body.

        Args:
            url: Absolute URL.
            params: Query-string parameters.
            user_agent: Identity to send, overriding the fetcher's default.

        Returns:
            The raw response body.

        Raises:
            FetchError: if every attempt fails, or the server returns a status
                that retrying will not fix.
        """
        headers = {"User-Agent": user_agent or self.user_agent, "Accept": "*/*"}
        last_error: str = "no attempt was made"

        for attempt in range(self.policy.attempts):
            self._throttle()
            try:
                response = httpx.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    follow_redirects=True,
                )
            except httpx.HTTPError as exc:  # transport failure: worth retrying
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == httpx.codes.OK:
                    return response.content
                last_error = f"HTTP {response.status_code}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise FetchError(f"{url}: {last_error}")

            if attempt < self.policy.attempts - 1:
                time.sleep(self.policy.delay(attempt))

        raise FetchError(f"{url}: giving up after {self.policy.attempts} attempts ({last_error})")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.policy.min_interval:
            time.sleep(self.policy.min_interval - elapsed)
        self._last_request_at = time.monotonic()
