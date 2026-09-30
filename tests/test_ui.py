import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from kin.ui.app import Hub, create_app

TOKEN = "t0ken"


@pytest.fixture
def client():
    hub = Hub()
    with TestClient(create_app(TOKEN, hub)) as c:
        yield c, hub


def test_token_required(client):
    c, _ = client
    assert c.get("/").status_code == 403
    assert c.get("/?token=wrong").status_code == 403
    assert c.get("/api/state").status_code == 403
    r = c.get(f"/?token={TOKEN}")
    assert r.status_code == 200 and "Live transcript" in r.text


def test_websocket_rejects_bad_token(client):
    c, _ = client
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws?token=nope") as ws:
            ws.receive_text()


def test_live_messages_reach_browser(client):
    c, hub = client
    with c.websocket_connect(f"/ws?token={TOKEN}") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        hub.publish({"type": "transcript", "id": 1, "text": "hello kin"})
        assert ws.receive_json()["text"] == "hello kin"
        hub.publish({"type": "status", "gate": True})
        assert ws.receive_json()["type"] == "status"
    time.sleep(0.05)
    state = c.get(f"/api/state?token={TOKEN}").json()
    assert [t["text"] for t in state["transcripts"]] == ["hello kin"]
    assert state["latest"]["status"]["gate"] is True


def test_late_browser_gets_backlog(client):
    c, hub = client
    hub.publish({"type": "transcript", "id": 1, "text": "said earlier"})
    time.sleep(0.05)
    with c.websocket_connect(f"/ws?token={TOKEN}") as ws:
        snap = ws.receive_json()
        assert [t["text"] for t in snap["transcripts"]] == ["said earlier"]
