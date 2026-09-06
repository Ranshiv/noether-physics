"""The single choke point for outbound HTTP.

Every connector goes through here, which is what makes three properties true
project-wide rather than per-connector: rate limits are actually honoured, all
traffic is recordable, and no request escapes without a User-Agent that lets an
archive operator work out who we are.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self
from urllib.parse import urlparse

import httpx

from . import __version__
from .cassettes import CassetteStore, Interaction, identity
from .config import DEFAULT_MIN_INTERVAL_S, USER_AGENT, Config


class RateLimiter:
    """Enforce a minimum interval between requests, per host.

    Deliberately a blocking sleep rather than a token bucket: we are not trying
    to burst. arXiv's three seconds is a floor we respect, not a budget we spend.
    """

    def __init__(self, default_interval_s: float = DEFAULT_MIN_INTERVAL_S) -> None:
        self._default = default_interval_s
        self._intervals: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def configure(self, host: str, interval_s: float) -> None:
        with self._lock:
            self._intervals[host] = interval_s

    def interval_for(self, host: str) -> float:
        return self._intervals.get(host, self._default)

    def wait(self, host: str) -> float:
        """Block until this host may be called again. Returns seconds slept."""
        with self._lock:
            interval = self._intervals.get(host, self._default)
            now = time.monotonic()
            last = self._last.get(host)
            delay = 0.0 if last is None else max(0.0, interval - (now - last))
            # Reserve the slot before releasing the lock, so concurrent callers
            # queue behind us instead of all reading the same stale timestamp.
            self._last[host] = now + delay
        if delay > 0:
            time.sleep(delay)
        return delay


@dataclass(frozen=True)
class Response:
    """What a connector sees. Deliberately smaller than httpx.Response."""

    status: int
    content: bytes
    headers: dict[str, str]
    url: str
    #: True when served from a cassette rather than the network.
    replayed: bool = False

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def raise_for_status(self) -> Response:
        if self.status >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status} from {self.url}",
                request=None,  # type: ignore[arg-type]
                response=None,  # type: ignore[arg-type]
            )
        return self


class NetClient:
    """Rate-limited, recordable HTTP client."""

    #: Statuses worth retrying: transient server faults and explicit throttling.
    RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self.limiter = RateLimiter()
        self.store = CassetteStore(self.config.cassettes())
        self._client: httpx.Client | None = None

    # ---- lifecycle -----------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.config.timeout_s,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT.format(version=__version__)},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- requests ------------------------------------------------------

    def get(
        self,
        provider: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return self.request("GET", provider, url, params=params, headers=headers)

    def request(
        self,
        method: str,
        provider: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Issue one request, subject to the configured cassette mode."""
        params = dict(params or {})
        key = identity(provider, method, url, params)
        mode = self.config.cassette_mode

        if mode == "replay":
            # A miss raises rather than falling through to the network: a test
            # suite that quietly starts hitting arXiv is worse than one that fails.
            recorded = self.store.load(key)
            return Response(
                status=recorded.status,
                content=recorded.content,
                headers=recorded.headers,
                url=recorded.url,
                replayed=True,
            )

        response = self._live(method, url, params, headers)

        if mode == "record":
            self.store.save(
                Interaction(
                    key=key,
                    provider=provider,
                    method=method.upper(),
                    url=url,
                    params=params,
                    status=response.status,
                    content=response.content,
                    headers=response.headers,
                )
            )
        return response

    def _live(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str] | None,
    ) -> Response:
        host = urlparse(url).netloc
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            self.limiter.wait(host)
            try:
                raw = self._http().request(
                    method.upper(), url, params=dict(params), headers=dict(headers or {})
                )
            except httpx.HTTPError as exc:
                last_error = exc
                self._backoff(attempt)
                continue

            if raw.status_code in self.RETRY_STATUSES and attempt < self.config.max_retries - 1:
                # Honour Retry-After when the server bothers to send one.
                self._backoff(attempt, retry_after=raw.headers.get("Retry-After"))
                continue

            return Response(
                status=raw.status_code,
                content=raw.content,
                headers=dict(raw.headers),
                url=str(raw.url),
            )

        raise RuntimeError(f"{method} {url} failed after {self.config.max_retries} attempts") from last_error

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 60.0))
                return
            except ValueError:
                pass  # Retry-After can be an HTTP date; fall back to exponential.
        time.sleep(min(2.0**attempt, 30.0))
