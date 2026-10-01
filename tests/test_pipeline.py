"""Full multi-process runs through the supervisor: the real week-1 system."""

import time

import psutil
import pytest

from kin.config import Settings, get_profile
from kin.supervisor import Supervisor
from tests.conftest import DATA, wer

pytestmark = pytest.mark.models


def replay(wav, speed=2.0):
    sup = Supervisor(Settings(profile=get_profile("phone")), source=str(wav), speed=speed, ui=False)
    try:
        sup.start()
        sup.run(until_eos=True, duration=60)
    finally:
        sup.stop()
    return sup


def test_replay_command_end_to_end():
    sup = replay(DATA / "command.wav")
    types = [m["type"] for m in sup.messages]
    assert "audio_ready" in types and "eos" in types
    wakes = [m for m in sup.messages if m["type"] == "wake"]
    trs = [m for m in sup.messages if m["type"] == "transcript" and m["text"]]
    assert len(wakes) == 1
    assert trs and all(t["wake"] for t in trs)
    commands = " ".join(t["command"] for t in trs if t["command"])
    assert wer("remind me to call rohan about the site visit on thursday", commands) < 0.2, commands
    assert max(t["latency_ms"] for t in trs) < 1500  # end of speech -> text


def test_replay_ambient_has_no_command():
    sup = replay(DATA / "ambient.wav", speed=3.0)
    trs = [m for m in sup.messages if m["type"] == "transcript" and m["text"]]
    assert trs and not any(t["wake"] for t in trs)
    assert not any(m["type"] == "wake" for m in sup.messages)
    text = " ".join(t["text"] for t in trs)
    ref = "the budget meeting moved to friday afternoon priya will send the slides tomorrow morning"
    assert wer(ref, text) < 0.25, text


def test_resources_are_measured_and_within_budget():
    sup = replay(DATA / "command.wav", speed=1.0)
    assert len(sup.stats_history) >= 3
    assert set(sup.stats_history[-1]["procs"]) == {"kin-audio", "kin-core", "kin-gateway"}
    assert max(s["procs"]["kin-gateway"]["rss_mb"] for s in sup.stats_history) < 100  # sheet 3: about 45 MB
    # Sheet 3 budgets, with headroom for Windows working-set accounting.
    assert max(s["procs"]["kin-audio"]["rss_mb"] for s in sup.stats_history) < 120
    assert max(s["procs"]["kin-core"]["rss_mb"] for s in sup.stats_history) < 1100


def test_crashed_process_is_restarted():
    sup = Supervisor(Settings(profile=get_profile("phone")), source=str(DATA / "ambient.wav"), ui=False)
    try:
        sup.start()
        audio = next(c for c in sup.children if c.name == "kin-audio")
        old_pid = audio.proc.pid
        psutil.Process(old_pid).kill()
        t0 = time.monotonic()
        while audio.proc.pid == old_pid and time.monotonic() - t0 < 5:
            sup.run(duration=0.2)
        assert audio.proc.pid != old_pid and audio.proc.is_alive()
        assert time.monotonic() - t0 < 2.5  # sheet 8: restart within 2 s
        assert len(audio.restarts) == 1
    finally:
        sup.stop()


def test_ui_streams_live_transcript_to_browser(tmp_path):
    """Real uvicorn + real WebSocket: what the browser page sees."""
    import json

    import httpx
    from websockets.sync.client import connect

    s = Settings(profile=get_profile("phone"), ui_port=8799, memory_db=str(tmp_path / "kin.db"))
    sup = Supervisor(s, source=str(DATA / "command.wav"), speed=2.0, ui=True)
    try:
        sup.start()
        deadline = time.monotonic() + 15
        while True:
            try:
                r = httpx.get(sup.ui_url, timeout=1)
                break
            except httpx.TransportError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        assert r.status_code == 200
        assert httpx.get("http://127.0.0.1:8799/", timeout=2).status_code == 403

        seen: list[dict] = []
        with connect(f"ws://127.0.0.1:8799/ws?token={sup.token}", open_timeout=5) as ws:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                sup.run(duration=0.05)  # keep supervising (stats, restarts) while we listen
                try:
                    msg = json.loads(ws.recv(timeout=0.2))
                except TimeoutError:
                    continue
                items = msg["transcripts"] + list(msg["latest"].values()) if msg["type"] == "snapshot" else [msg]
                seen += items
                if any(m["type"] == "eos" for m in items):
                    break
        types = {m["type"] for m in seen}
        assert {"wake", "transcript", "status", "stats"} <= types, types
        text = " ".join(m["text"] for m in seen if m["type"] == "transcript")
        assert "Rohan" in text
        # The memory tab's API, answered by kin-core through kin-ui.
        eps = httpx.get(f"http://127.0.0.1:8799/api/memory/timeline?token={sup.token}", timeout=10).json()
        assert any("Rohan" in e["text"] and e["utterances"][0]["command"] for e in eps)
        hits = httpx.get(f"http://127.0.0.1:8799/api/memory/search?token={sup.token}&q=site visit", timeout=10).json()
        assert hits and "Rohan" in hits[0]["text"]
    finally:
        sup.stop()


def test_replay_builds_searchable_memory_that_survives_restart(tmp_path):
    db = str(tmp_path / "kin.db")
    settings = Settings(profile=get_profile("phone"), memory_db=db)
    sup = Supervisor(settings, source=str(DATA / "ambient.wav"), speed=3.0, ui=False)
    try:
        sup.start()
        sup.run(until_eos=True, duration=60)
        eps = sup.rpc("timeline")
        assert len(eps) == 1 and "budget" in eps[0]["text"].lower()
        assert eps[0]["closed"] and eps[0]["embedded"]  # indexed before end-of-replay was signalled
        hits = sup.rpc("search", {"q": "What did Priya say about the budget?"})
        assert hits[0]["id"] == eps[0]["id"]
        assert hits[0]["keyword_match"] and hits[0]["vector_similarity"] > 0.4
        st = sup.rpc("stats")
        assert st["vector_backend"] == "sqlite-vec" and st["cipher"].startswith("SQLCipher")
    finally:
        sup.stop()

    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
    assert b"budget" not in raw.lower() and b"priya" not in raw.lower()  # encrypted at rest

    sup2 = Supervisor(settings, source=str(DATA / "hey_jarvis.wav"), speed=3.0, ui=False)
    try:
        sup2.start()
        assert [e["id"] for e in sup2.rpc("timeline")] == [eps[0]["id"]]
        q = "when is the meeting about money"  # meaning, not keywords
        assert sup2.rpc("search", {"q": q})[0]["id"] == eps[0]["id"]
    finally:
        sup2.stop()


def test_spoken_command_is_answered_and_acted_on(tmp_path):
    settings = Settings(profile=get_profile("phone"), memory_db=str(tmp_path / "kin.db"))
    sup = Supervisor(settings, source=str(DATA / "command.wav"), speed=2.0, ui=False)
    try:
        sup.start()
        sup.run(until_eos=True, duration=60)
        answers = [m for m in sup.messages if m["type"] == "answer"]
        assert len(answers) == 1, answers
        a = answers[0]
        assert a["source"] == "voice" and a["intent"] == "task.add"
        assert a["text"].startswith("Added: call Rohan about the site visit")
        tasks = sup.rpc("tasks")["tasks"]
        assert [t["title"] for t in tasks] == ["Call Rohan about the site visit"]
        assert tasks[0]["due_at"] is not None  # "on Thursday"
        # Asked by typing, through the same heavy lane:
        reply = sup.rpc("ask", {"q": "What's on my list?"}, timeout=60)
        assert "call Rohan about the site visit" in reply["text"]
    finally:
        sup.stop()
