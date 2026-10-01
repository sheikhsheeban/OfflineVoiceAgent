"""The only code in Kin that opens a network connection.

Every request goes through `Egress.get`, which enforces (sheet 4):
  - the host must be on the allowlist (three hosts), HTTPS only, no redirects;
  - the request line is at most 2 KB and the response is read to at most 16 KB;
  - a 3 s deadline;
  - a 15 minute cache, so repeated questions do not repeat calls;
  - a LedgerEntry for every call, holding the exact request that was sent.
"""

from __future__ import annotations

import time
from urllib.parse import urlsplit

import httpx

from kin.contracts import LedgerEntry

MAX_REQUEST_BYTES = 2048
MAX_RESPONSE_BYTES = 16 * 1024
TIMEOUT_S = 3.0
CACHE_TTL_S = 15 * 60


class Rejected(Exception):
    """The request was refused before anything was sent."""


class Offline(Exception):
    """No route, DNS failure or connection refused."""

    def __init__(self, ledger: LedgerEntry):
        self.ledger = ledger


class Timeout(Exception):
    def __init__(self, ledger: LedgerEntry):
        self.ledger = ledger


class Egress:
    def __init__(self, allowed_hosts: set[str], transport: httpx.BaseTransport | None = None, clock=time.time):
        self.allowed = {h.lower() for h in allowed_hosts}
        self._transport = transport
        self._clock = clock
        self._cache: dict[str, tuple[float, str]] = {}
        self.ledger: list[LedgerEntry] = []  # this process's record; kin-core stores the durable copy (CL3)

    def _record(self, host: str, payload: str, out: int, inn: int, status: str) -> LedgerEntry:
        e = LedgerEntry(ts=self._clock(), host=host, payload=payload, bytes_out=out, bytes_in=inn, status=status)
        self.ledger.append(e)
        return e

    def get(self, url: str, headers: dict | None = None) -> tuple[str, LedgerEntry]:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https" or host not in self.allowed:
            raise Rejected(f"host {host or url[:40]!r} is not on the allowlist")
        payload = f"GET {parts.path}{'?' + parts.query if parts.query else ''} HTTP/1.1\nHost: {host}"
        if len(payload.encode()) > MAX_REQUEST_BYTES:
            raise Rejected("request is larger than 2 KB")

        cached = self._cache.get(url)
        if cached and self._clock() - cached[0] < CACHE_TTL_S:
            return cached[1], self._record(host, payload, 0, 0, "cached")

        out = len(payload.encode())
        t0 = time.monotonic()
        body = b""
        try:
            with httpx.Client(transport=self._transport, timeout=TIMEOUT_S, follow_redirects=False) as client:
                with client.stream("GET", url, headers=headers or {}) as resp:
                    for chunk in resp.iter_bytes():
                        body += chunk
                        if len(body) >= MAX_RESPONSE_BYTES or time.monotonic() - t0 > TIMEOUT_S:
                            break  # the rest is never read
                    body = body[:MAX_RESPONSE_BYTES]
                    status = resp.status_code
        except httpx.TimeoutException:
            raise Timeout(self._record(host, payload, out, 0, "timeout")) from None
        except httpx.HTTPError:
            raise Offline(self._record(host, payload, out, 0, "failed")) from None
        entry = self._record(host, payload, out, len(body), str(status))
        if status != 200:
            raise Rejected(f"{host} answered {status}") from None
        text = body.decode("utf-8", errors="replace")
        self._cache[url] = (self._clock(), text)
        return text, entry
