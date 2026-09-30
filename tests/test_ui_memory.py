"""kin-ui memory endpoints, with a fake kin-core answering on the other end of the queues."""

import queue
import threading
import time

import pytest
from fastapi.testclient import TestClient

from kin.ui.app import CoreClient, Hub, _pump, create_app

TOKEN = "t0ken"


class FakeCore:
    def __init__(self, cmd_q, ui_q):
        self.calls = []
        self._cmd_q, self._ui_q = cmd_q, ui_q
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                req = self._cmd_q.get(timeout=0.05)
            except queue.Empty:
                continue
            self.calls.append((req["method"], req["params"]))
            if req["method"] == "episode" and req["params"]["id"] == 404:
                self._ui_q.put({"type": "rpc_result", "id": req["id"], "result": None})
            elif req["method"] == "boom":
                self._ui_q.put({"type": "rpc_result", "id": req["id"], "error": "kaboom"})
            else:
                self._ui_q.put({"type": "rpc_result", "id": req["id"], "result": {"ok": req["method"]}})


@pytest.fixture
def client():
    cmd_q, ui_q, stop = queue.Queue(), queue.Queue(), threading.Event()
    hub, core_client = Hub(), CoreClient(cmd_q)
    fake = FakeCore(cmd_q, ui_q)
    threading.Thread(target=_pump, args=(ui_q, hub, core_client, stop), daemon=True).start()
    with TestClient(create_app(TOKEN, hub, core_client)) as c:
        yield c, fake
    stop.set()
    fake._stop.set()


def test_memory_endpoints_need_the_token(client):
    c, fake = client
    for method, path in [
        ("get", "/api/memory/timeline"),
        ("get", "/api/memory/search?q=x"),
        ("get", "/api/memory/stats"),
        ("delete", "/api/memory/episodes/1"),
        ("post", "/api/memory/forget"),
        ("post", "/api/memory/wipe?confirm=wipe"),
    ]:
        assert getattr(c, method)(path).status_code == 403, path
    assert fake.calls == []


def test_requests_are_forwarded_to_core(client):
    c, fake = client
    t = f"token={TOKEN}"
    assert c.get(f"/api/memory/timeline?{t}&limit=999").json() == {"ok": "timeline"}
    assert c.get(f"/api/memory/search?{t}&q=budget&k=3").json() == {"ok": "search"}
    assert c.delete(f"/api/memory/episodes/7?{t}").json() == {"ok": "delete_episode"}
    assert c.post(f"/api/memory/forget?{t}&minutes=10").json() == {"ok": "forget"}
    assert c.post(f"/api/memory/wipe?{t}&confirm=wipe").json() == {"ok": "wipe"}
    assert fake.calls == [
        ("timeline", {"limit": 200, "before": None}),  # limit is capped
        ("search", {"q": "budget", "k": 3}),
        ("delete_episode", {"id": 7}),
        ("forget", {"minutes": 10.0}),
        ("wipe", {"confirm": "wipe"}),
    ]


def test_guard_rails(client):
    c, fake = client
    t = f"token={TOKEN}"
    assert c.post(f"/api/memory/wipe?{t}").status_code == 400  # needs confirm=wipe
    assert c.post(f"/api/memory/forget?{t}&minutes=0").status_code == 400
    assert c.get(f"/api/memory/episodes/404?{t}").status_code == 404
    assert c.get(f"/?{t}", headers={"host": "evil.example"}).status_code == 400  # DNS rebinding
    assert all(m != "wipe" for m, _ in fake.calls)


def test_core_errors_and_timeouts():
    import asyncio

    from fastapi import HTTPException

    cmd_q, ui_q, stop = queue.Queue(), queue.Queue(), threading.Event()
    core_client = CoreClient(cmd_q)
    fake = FakeCore(cmd_q, ui_q)
    threading.Thread(target=_pump, args=(ui_q, Hub(), core_client, stop), daemon=True).start()
    try:
        with pytest.raises(HTTPException) as err:
            asyncio.run(core_client.call("boom"))
        assert err.value.status_code == 400 and "kaboom" in err.value.detail
        fake._stop.set()  # core stops answering
        time.sleep(0.2)  # let its loop notice before the next request
        with pytest.raises(HTTPException) as err:
            asyncio.run(core_client.call("stats", timeout=0.3))
        assert err.value.status_code == 504
    finally:
        stop.set()


def test_memory_changed_is_streamed_but_not_kept(client):
    c, _ = client
    hub = c.app.state.hub
    with c.websocket_connect(f"/ws?token={TOKEN}") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        hub.publish({"type": "memory_changed", "reason": "utterance"})
        assert ws.receive_json()["type"] == "memory_changed"
    assert "memory_changed" not in hub.snapshot()["latest"]


def test_week3_endpoints_are_forwarded(client):
    c, fake = client
    t = f"token={TOKEN}"
    assert c.post(f"/api/ask?{t}&q=When is the site visit?").json() == {"ok": "ask"}
    assert c.get(f"/api/tasks?{t}").json() == {"ok": "tasks"}
    assert c.post(f"/api/tasks/4/confirm?{t}").json() == {"ok": "task_set"}
    assert c.post(f"/api/tasks/4/done?{t}").json() == {"ok": "task_set"}
    assert c.post(f"/api/tasks/4/dismiss?{t}").json() == {"ok": "task_set"}
    assert c.post(f"/api/tasks/4/delete?{t}").json() == {"ok": "task_delete"}
    assert c.post(f"/api/reminders/2/cancel?{t}").json() == {"ok": "reminder_cancel"}
    assert c.post(f"/api/notes?{t}&text=Buy milk").json() == {"ok": "note_add"}
    assert fake.calls == [
        ("ask", {"q": "When is the site visit?"}),
        ("tasks", {}),
        ("task_set", {"id": 4, "status": "open"}),
        ("task_set", {"id": 4, "status": "done"}),
        ("task_set", {"id": 4, "status": "dismissed"}),
        ("task_delete", {"id": 4}),
        ("reminder_cancel", {"id": 2}),
        ("note_add", {"text": "Buy milk"}),
    ]


def test_week3_endpoints_need_the_token_and_valid_input(client):
    c, fake = client
    for method, path in [("post", "/api/ask?q=x"), ("get", "/api/tasks"), ("post", "/api/tasks/1/done"),
                         ("post", "/api/reminders/1/cancel"), ("post", "/api/notes?text=x")]:  # fmt: skip
        assert getattr(c, method)(path).status_code == 403, path
    t = f"token={TOKEN}"
    assert c.post(f"/api/ask?{t}&q=%20").status_code == 400
    assert c.post(f"/api/tasks/1/explode?{t}").status_code == 404
    assert fake.calls == []


def test_answers_and_reminders_are_kept_for_late_browsers(client):
    c, hub = client[0], client[0].app.state.hub
    hub.publish({"type": "answer", "text": "Thursday at 4 pm."})
    hub.publish({"type": "reminder", "text": "leave"})
    time.sleep(0.05)
    with c.websocket_connect(f"/ws?token={TOKEN}") as ws:
        snap = ws.receive_json()
    assert [m["type"] for m in snap["transcripts"]] == ["answer", "reminder"]
