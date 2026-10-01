"""kin-gateway (G1): the only process allowed online.

Receives a typed FactRequest from kin-core over local IPC, re-validates it,
scrubs it, calls one of three allowlisted hosts and returns a FactResponse that
always carries the LedgerEntry of the call (including failures), which kin-core
stores in `egress_log` and the UI shows (spec CL3).
"""

from __future__ import annotations

import logging
import threading
import time

from pydantic import ValidationError

from kin import ipc
from kin.contracts import FactRequest, FactResponse, LedgerEntry
from kin.gateway import scrub
from kin.gateway.clients import news, search, weather
from kin.gateway.transport import Egress, Offline, Rejected, Timeout
from kin.sync import wait_for

log = logging.getLogger("kin.gateway")

CLIENTS = {"weather": weather, "news": news, "search": search}


def allowed_hosts() -> set[str]:
    """Three hosts: the forecast, the news feeds and the configured search API."""
    return {weather.HOST, news.HOST, search.host()}


class Gateway:
    def __init__(self, egress: Egress | None = None):
        self.egress = egress or Egress(allowed_hosts())

    def _fail(self, kind: str, error: str, ledger: LedgerEntry | None = None, reason: str = "") -> FactResponse:
        return FactResponse(ok=False, kind=kind, data={"reason": reason} if reason else {}, error=error, ledger=ledger)

    def handle(self, message: dict) -> dict:
        kind = str(message.get("request", {}).get("kind", "?"))
        try:
            req = FactRequest.model_validate(message.get("request", {}))  # extra fields are refused
        except ValidationError as e:
            log.warning("rejected request: %s", e.errors()[0]["msg"])
            return self._fail(kind, "rejected", reason="invalid request").model_dump()
        if req.kind == "search":
            if search.backend() is None:
                return self._fail(kind, "rejected", reason="search is not configured").model_dump()
            q = scrub.scrub_query(req.q or "", message.get("names"))
            if not q:
                return self._fail(kind, "rejected", reason="nothing left to search for").model_dump()
            req = req.model_copy(update={"q": q})
        elif req.kind == "weather" and (req.lat is None or req.lon is None):
            return self._fail(kind, "rejected", reason="weather needs a location").model_dump()
        client = CLIENTS[req.kind]
        headers = search.headers() if req.kind == "search" else None
        try:
            text, entry = self.egress.get(client.url(req), headers)
            data = client.parse(text, req)
            return FactResponse(ok=True, kind=req.kind, data=data, ledger=entry).model_dump()
        except Rejected as e:
            return self._fail(req.kind, "rejected", reason=str(e)).model_dump()
        except Timeout as e:
            return self._fail(req.kind, "timeout", e.ledger).model_dump()
        except Offline as e:
            return self._fail(req.kind, "offline", e.ledger).model_dump()
        except (ValueError, KeyError):  # the host answered with something unexpected
            return self._fail(req.kind, "rejected", self.egress.ledger[-1], "unexpected response").model_dump()


def run(session: str, authkey: bytes, stop_evt) -> None:
    from kin.logs import setup

    setup("kin-gateway")
    gw = Gateway()
    listener = ipc.listen("gateway", session, authkey)
    log.info("ready (hosts: %s)", ", ".join(sorted(gw.egress.allowed)))

    def serve() -> None:
        while True:
            try:
                conn = listener.accept()
            except Exception:  # noqa: BLE001 - failed handshake or closed listener
                if stop_evt.is_set():
                    return
                time.sleep(0.05)
                continue
            try:
                msg = conn.recv()
                conn.send(gw.handle(msg))
            except Exception:  # noqa: BLE001 - one bad client must not stop the gateway
                log.exception("request failed")
            finally:
                conn.close()

    threading.Thread(target=serve, name="serve", daemon=True).start()
    wait_for(stop_evt)
    log.info("stopped")
