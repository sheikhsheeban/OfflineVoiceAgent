"""kin-ui: local-only web UI (127.0.0.1 + per-session token).

Shows the live transcript, listening state and per-process CPU/RAM, and the
memory timeline with search and delete. The UI never opens the memory store
itself: every memory request is forwarded to kin-core, which holds the key.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import queue
import secrets
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse

log = logging.getLogger("kin.ui")
STATIC = Path(__file__).parent / "static"


class Hub:
    """Fans messages out to every connected browser and keeps a short backlog."""

    def __init__(self, backlog: int = 200):
        self.transcripts: collections.deque = collections.deque(maxlen=backlog)
        self.latest: dict[str, dict] = {}  # last status / stats / audio_ready / error
        self._clients: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, msg: dict) -> None:
        """Thread-safe entry point."""
        if self._loop is None:
            self._record(msg)
            return
        self._loop.call_soon_threadsafe(self._fanout, msg)

    def _record(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind in ("transcript", "wake", "dropped", "answer", "reminder"):
            self.transcripts.append(msg)
        elif kind not in ("memory_changed",):  # a signal to refresh, not state
            self.latest[msg.get("type", "?")] = msg

    def _fanout(self, msg: dict) -> None:
        self._record(msg)
        for q in list(self._clients):
            if q.qsize() < 500:  # a stalled tab must not grow memory without bound
                q.put_nowait(msg)

    def snapshot(self) -> dict:
        return {"transcripts": list(self.transcripts), "latest": self.latest}

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)


class CoreClient:
    """Request/response with kin-core over the supervisor's queues."""

    def __init__(self, cmd_q):
        self._cmd_q = cmd_q
        self._pending: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._lock = threading.Lock()

    async def call(self, method: str, params: dict | None = None, timeout: float = 15.0):
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        req_id = secrets.token_hex(8)
        with self._lock:
            self._pending[req_id] = (loop, fut)
        self._cmd_q.put({"id": req_id, "method": method, "params": params or {}})
        try:
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "kin-core did not answer") from None
        finally:
            with self._lock:
                self._pending.pop(req_id, None)
        if "error" in msg:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, msg["error"])
        return msg["result"]

    def resolve(self, msg: dict) -> None:
        """Thread-safe: called by the queue pump."""
        with self._lock:
            entry = self._pending.get(msg.get("id"))
        if entry:
            loop, fut = entry
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(msg))


def create_app(token: str, hub: Hub | None = None, core: CoreClient | None = None) -> FastAPI:
    hub = hub or Hub()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.bind(asyncio.get_running_loop())
        yield

    app = FastAPI(title="Kin", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.hub = hub
    # Loopback names only: blocks DNS-rebinding pages from reaching the API.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    expected = token

    def check(t: str | None) -> None:
        if not t or not secrets.compare_digest(t, expected):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "bad or missing token")

    @app.get("/", response_class=HTMLResponse)
    async def index(token: str | None = Query(None)) -> str:
        check(token)
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    async def state(token: str | None = Query(None)) -> dict:
        check(token)
        return hub.snapshot()

    async def memory(method: str, params: dict | None = None):
        if core is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "memory is not connected")
        return await core.call(method, params)

    @app.get("/api/memory/timeline")
    async def timeline(token: str | None = Query(None), limit: int = 50, before: float | None = None):
        check(token)
        return await memory("timeline", {"limit": min(max(limit, 1), 200), "before": before})

    @app.get("/api/memory/search")
    async def search(q: str, token: str | None = Query(None), k: int = 5):
        check(token)
        return await memory("search", {"q": q[:200], "k": min(max(k, 1), 20)})

    @app.get("/api/memory/stats")
    async def memory_stats(token: str | None = Query(None)):
        check(token)
        return await memory("stats")

    @app.get("/api/memory/episodes/{ep_id}")
    async def episode(ep_id: int, token: str | None = Query(None)):
        check(token)
        ep = await memory("episode", {"id": ep_id})
        if ep is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such episode")
        return ep

    @app.delete("/api/memory/episodes/{ep_id}")
    async def delete_episode(ep_id: int, token: str | None = Query(None)):
        check(token)
        return await memory("delete_episode", {"id": ep_id})

    @app.post("/api/memory/forget")
    async def forget(token: str | None = Query(None), minutes: float = 10):
        check(token)
        if not 0 < minutes <= 24 * 60:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "minutes must be between 0 and 1440")
        return await memory("forget", {"minutes": minutes})

    @app.post("/api/memory/wipe")
    async def wipe(token: str | None = Query(None), confirm: str = ""):
        check(token)
        if confirm != "wipe":
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "add confirm=wipe")
        return await memory("wipe", {"confirm": "wipe"})

    @app.post("/api/ask")
    async def ask(token: str | None = Query(None), q: str = ""):
        check(token)
        if not q.strip():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty question")
        if core is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "kin-core is not connected")
        return await core.call("ask", {"q": q[:300]}, timeout=60)

    @app.get("/api/tasks")
    async def tasks(token: str | None = Query(None)):
        check(token)
        return await memory("tasks")

    @app.post("/api/tasks/{task_id}/{action}")
    async def task_action(task_id: int, action: str, token: str | None = Query(None)):
        check(token)
        statuses = {"confirm": "open", "done": "done", "dismiss": "dismissed", "reopen": "open"}
        if action in statuses:
            return await memory("task_set", {"id": task_id, "status": statuses[action]})
        if action == "delete":
            return await memory("task_delete", {"id": task_id})
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown action {action!r}")

    @app.post("/api/reminders/{reminder_id}/cancel")
    async def reminder_cancel(reminder_id: int, token: str | None = Query(None)):
        check(token)
        return await memory("reminder_cancel", {"id": reminder_id})

    @app.post("/api/notes")
    async def note_add(token: str | None = Query(None), text: str = ""):
        check(token)
        if not text.strip():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty note")
        return await memory("note_add", {"text": text[:300]})

    @app.websocket("/ws")
    async def ws(sock: WebSocket, token: str | None = Query(None)) -> None:
        if not token or not secrets.compare_digest(token, expected):
            await sock.close(code=4403)
            return
        await sock.accept()
        await sock.send_text(json.dumps({"type": "snapshot", **hub.snapshot()}))
        q = hub.subscribe()
        try:
            while True:
                await sock.send_text(json.dumps(await q.get()))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(q)

    return app


def _pump(ui_q, hub: Hub, core: CoreClient, stop_evt) -> None:
    while not stop_evt.is_set():
        try:
            msg = ui_q.get(timeout=0.2)
            if msg.get("type") == "rpc_result":
                core.resolve(msg)
            else:
                hub.publish(msg)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            return


def run(ui_q, cmd_q, stop_evt, host: str, port: int, token: str) -> None:
    import uvicorn

    from kin.logs import setup

    setup("kin-ui")
    cmd_q.cancel_join_thread()
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("kin-ui must bind to loopback only")
    hub = Hub()
    core = CoreClient(cmd_q)
    app = create_app(token, hub, core)
    threading.Thread(target=_pump, args=(ui_q, hub, core, stop_evt), daemon=True).start()
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False))

    def _watch_stop() -> None:
        stop_evt.wait()
        server.should_exit = True

    threading.Thread(target=_watch_stop, daemon=True).start()
    log.info("UI at http://%s:%d/?token=%s", host, port, token)
    server.run()
