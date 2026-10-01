"""Week 4: the gateway, the ledger, voice, the clip link, mute, network isolation.

Test names carry the spec ID they cover (`test_<ID>_<what>`); tests/spec_report.py prints the table.
The gateway is exercised with httpx.MockTransport so no test needs the internet; two `online`
tests hit the real services and skip themselves when there is no connection.
"""

import datetime as dt
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from kin import ipc
from kin.audio.clip_serial import ClipLink, MuteLatch, led_command, parse_line
from kin.contracts import FactRequest, FactResponse, LedgerEntry, MuteState, StatusEvent
from kin.gateway.server import Gateway, allowed_hosts
from kin.gateway.transport import Egress, Rejected
from kin.output.status import PRIORITY, StatusController
from kin.output.voice import Voice, spoken_form

ROOT = Path(__file__).resolve().parent.parent
SOON = dt.datetime.now().replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=3)


def weather_json(rain_at=SOON, pct=70):
    times = [(rain_at - dt.timedelta(hours=3) + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(9)]
    return json.dumps({"hourly": {"time": times,
                                  "precipitation_probability": [pct if t == rain_at.strftime("%Y-%m-%dT%H:00") else 5 for t in times],
                                  "temperature_2m": [27.0] * len(times)}})  # fmt: skip


RSS = ("<rss><channel><title>BBC</title>"
       + "".join(f"<item><title><![CDATA[Headline number {i} &amp; more]]></title></item>" for i in range(8))
       + "</channel></rss>")  # fmt: skip


class Net:
    """A fake internet: records every request and answers by host."""

    def __init__(self, fail=None):
        self.requests: list[httpx.Request] = []
        self.fail = fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            raise self.fail
        host = request.url.host
        if host == "api.open-meteo.com":
            return httpx.Response(200, text=weather_json())
        if host == "feeds.bbci.co.uk":
            return httpx.Response(200, text=RSS)
        if host == "search.example.org":
            return httpx.Response(200, json={"results": [{"title": "Wardha road", "content": "Open again <b>today</b>"}]})
        return httpx.Response(404)


def gateway(net=None, clock=time.time):
    net = net or Net()
    return Gateway(Egress(allowed_hosts(), transport=httpx.MockTransport(net), clock=clock)), net


# ---- G1 / X1 / X2 / X3 / E27 ------------------------------------------------------------


def test_G1_rejects_hosts_off_the_allowlist_and_plain_http():
    net = Net()
    eg = Egress(allowed_hosts(), transport=httpx.MockTransport(net))
    with pytest.raises(Rejected):
        eg.get("https://evil.example.com/steal")
    with pytest.raises(Rejected):
        eg.get("http://api.open-meteo.com/v1/forecast")  # not HTTPS
    assert net.requests == []  # nothing was sent


def test_G1_does_not_follow_redirects():
    eg = Egress(allowed_hosts(), transport=httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": "https://evil.example.com"})))
    with pytest.raises(Rejected, match="302"):
        eg.get("https://api.open-meteo.com/v1/forecast?x=1")


def test_G1_caps_request_at_2kb_and_response_at_16kb():
    eg = Egress(allowed_hosts(), transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 100_000)))
    with pytest.raises(Rejected, match="2 KB"):
        eg.get("https://api.open-meteo.com/v1/forecast?" + "a=b&" * 600)
    text, entry = eg.get("https://api.open-meteo.com/v1/forecast?ok=1")
    assert len(text) == 16 * 1024 and entry.bytes_in == 16 * 1024


def test_G1_times_out_with_a_ledger_entry():
    gw, _ = gateway(Net(fail=httpx.ReadTimeout("slow")))
    r = FactResponse.model_validate(gw.handle({"request": {"kind": "weather", "lat": 21.1, "lon": 79.1, "hours": 6}}))
    assert not r.ok and r.error == "timeout" and r.ledger.status == "timeout"


def test_G1_no_route_means_offline_with_a_failed_ledger_entry():
    gw, _ = gateway(Net(fail=httpx.ConnectError("no route")))
    r = FactResponse.model_validate(gw.handle({"request": {"kind": "news", "topic": "tech"}}))
    assert not r.ok and r.error == "offline" and r.ledger.status == "failed" and r.ledger.bytes_in == 0


def test_G1_caches_for_15_minutes():
    now = [1000.0]
    gw, net = gateway(clock=lambda: now[0])
    req = {"request": {"kind": "weather", "lat": 21.1, "lon": 79.1, "hours": 12}}
    a, b = gw.handle(req), gw.handle(req)
    assert a["ledger"]["status"] == "200" and b["ledger"]["status"] == "cached" and len(net.requests) == 1
    assert b["ledger"]["bytes_out"] == 0 and a["data"] == b["data"]
    now[0] += 16 * 60
    gw.handle(req)
    assert len(net.requests) == 2


def test_G1_scrubs_names_numbers_and_emails_from_search(monkeypatch):
    monkeypatch.setenv("KIN_SEARCH_URL", "https://search.example.org")
    gw, net = gateway()
    r = gw.handle({"request": {"kind": "search", "q": "Rohan 9876543210 rohan@x.com Wardha road closed 482913"},
                   "names": ["Rohan"]})  # fmt: skip
    assert r["ok"]
    sent = str(net.requests[0].url)
    for leaked in ("Rohan", "rohan", "9876543210", "482913", "@"):
        assert leaked not in sent
    assert "Wardha" in sent


def test_X1_open_meteo_returns_a_fixed_struct_and_only_lat_lon_hours_leave():
    gw, net = gateway()
    r = FactResponse.model_validate(gw.handle({"request": {"kind": "weather", "lat": 21.1234, "lon": 79.0876, "hours": 24}}))
    assert r.ok and set(r.data) == {"lat", "lon", "hours", "hourly"} and len(r.data["hourly"]) == 9
    assert net.requests[0].url.host == "api.open-meteo.com"
    assert set(net.requests[0].url.params) == {"latitude", "longitude", "hourly", "forecast_hours", "timezone"}
    assert net.requests[0].url.params["latitude"] == "21.1"  # rounded to 0.1 degree before it left


def test_X2_search_needs_configuration_and_returns_five_snippets_at_most(monkeypatch):
    gw, _ = gateway()
    off = gw.handle({"request": {"kind": "search", "q": "wardha road"}})
    assert not off["ok"] and off["error"] == "rejected" and "not configured" in off["data"]["reason"]
    monkeypatch.setenv("KIN_SEARCH_URL", "https://search.example.org")
    gw, net = gateway()
    on = gw.handle({"request": {"kind": "search", "q": "wardha road"}})
    assert on["ok"] and on["data"]["results"] == [{"title": "Wardha road", "snippet": "Open again today"}]
    assert net.requests[0].url.host == "search.example.org"


def test_X3_news_returns_five_headlines_from_an_allowlisted_feed():
    gw, net = gateway()
    r = gw.handle({"request": {"kind": "news", "topic": "sport"}})
    assert r["ok"] and len(r["data"]["headlines"]) == 5
    assert r["data"]["headlines"][0]["title"] == "Headline number 0 & more"
    assert net.requests[0].url.host == "feeds.bbci.co.uk" and net.requests[0].url.path.startswith("/sport")


def test_E27_gateway_may_reach_exactly_three_hosts():
    assert allowed_hosts() == {"api.open-meteo.com", "feeds.bbci.co.uk", "api.search.brave.com"}


# ---- INV7: only a valid FactRequest leaves ---------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "email", "q": "hi"},
        {"kind": "search", "q": "x" * 81},
        {"kind": "weather", "lat": 1, "lon": 2, "hours": 72},
        {"kind": "search", "q": "hi", "transcript": "everything I said today"},
        {"kind": "news", "topic": "gossip"},
        {"kind": "weather", "lat": 1, "lon": 2, "prompt": "leak"},
    ],
)
def test_INV7_invalid_requests_never_leave(bad):
    gw, net = gateway()
    r = gw.handle({"request": bad})
    assert not r["ok"] and r["error"] == "rejected" and net.requests == []


def test_INV7_a_valid_request_to_the_allowlist_is_small():
    gw, net = gateway()
    r = gw.handle({"request": {"kind": "weather", "lat": 21.1, "lon": 79.1, "hours": 48}})
    assert r["ok"] and r["ledger"]["bytes_out"] <= 2048


# ---- E25 / E26 / INV8 / O2 ---------------------------------------------------------------------


@pytest.fixture
def served_gateway():
    """A real local-IPC endpoint (named pipe on Windows, Unix socket elsewhere) in front of a mocked internet."""
    gw, net = gateway()
    session, key = uuid.uuid4().hex[:8], b"k" * 16
    listener = ipc.listen("gateway", session, key)
    done = threading.Event()

    def serve():
        while not done.is_set():
            try:
                conn = listener.accept()
            except Exception:  # noqa: BLE001
                return
            try:
                conn.send(gw.handle(conn.recv()))
            finally:
                conn.close()

    threading.Thread(target=serve, daemon=True).start()
    yield session, key, gw, net
    done.set()
    listener.close()


def test_E25_fact_request_travels_over_local_ipc(served_gateway):
    from kin.reasoning.facts import FactService

    session, key, gw, net = served_gateway
    resp = FactService(session, key).fetch(FactRequest(kind="weather", lat=21.1, lon=79.1, hours=12))
    assert resp.ok and resp.ledger.host == "api.open-meteo.com" and len(net.requests) == 1


def test_E25_wrong_key_is_refused(served_gateway):
    from kin.reasoning.facts import FactService

    session, _, _, net = served_gateway
    resp = FactService(session, b"wrong-key-wrong-k").fetch(FactRequest(kind="news", topic="tech"))
    assert not resp.ok and resp.error == "offline" and net.requests == []


def test_E26_INV8_every_online_call_is_in_the_ledger(served_gateway, tmp_path):
    from kin.memory.store import Store
    from kin.reasoning.facts import FactService

    session, key, gw, net = served_gateway
    store = Store(tmp_path / "k.db", "ab" * 32)
    shown = []

    def record(resp):  # what kin-core does with every response (CL3)
        if resp.ledger is not None:
            store.add_egress(resp.ledger.model_dump())
            shown.append(resp.ledger)

    svc = FactService(session, key, record=record)
    for topic in ("tech", "sport", "tech"):  # the third is served from cache but still recorded
        svc.fetch(FactRequest(kind="news", topic=topic))
    svc.fetch(FactRequest(kind="weather", lat=21.1, lon=79.1, hours=6))
    rows = store.egress()
    assert len(rows) == 4 == len(shown) == len(gw.egress.ledger)  # gateway calls = ledger rows = rows shown
    assert {r["status"] for r in rows} == {"200", "cached"}
    assert "GET /v1/forecast?latitude=21.1&longitude=79.1" in [r for r in rows if "forecast" in r["payload"]][0]["payload"]
    store.close()


def test_O2_ledger_is_served_to_the_ui_with_exact_payloads():
    from fastapi.testclient import TestClient

    from kin.ui.app import CoreClient, create_app

    class FakeCmdQ:
        def put(self, req):
            self.req = req
            client.resolve({"id": req["id"], "result": {"count": 1, "entries": [
                {"id": 1, "ts": 1.0, "host": "api.open-meteo.com", "payload": "GET /v1/forecast?latitude=21.1", "bytes_out": 90,
                 "bytes_in": 600, "status": "200"}]}})  # fmt: skip

    client = CoreClient(FakeCmdQ())
    c = TestClient(create_app("tok", core=client))
    body = c.get("/api/ledger?token=tok").json()
    assert body["entries"][0]["payload"].startswith("GET /v1/forecast") and body["count"] == 1
    assert "Egress ledger" in c.get("/?token=tok").text and "ledgerRows" in c.get("/?token=tok").text


# ---- O3 voice / E29 -----------------------------------------------------------------------------


class FakeEngine:
    def __init__(self):
        self.said = []

    def installed(self):
        return True

    def synth(self, text):
        import numpy as np

        self.said.append(text)
        return np.ones(1600, dtype=np.int16), 16000


def make_voice(**kw):
    played = []

    def player(audio, rate, stop):
        played.append((len(audio), rate))
        time.sleep(0.05)

    return Voice(FakeEngine(), player, **kw), played


def test_O3_speaks_the_answer_text_without_ids_and_cuts_long_replies():
    assert spoken_form("On Wednesday, I heard: it is fine #e-12 [e-3] cal-1") == "On Wednesday, I heard: it is fine"
    long = "This is a sentence about something. " * 30
    assert len(spoken_form(long)) <= 420 and spoken_form(long).endswith(".")


def test_O3_voice_plays_once_and_can_be_turned_off():
    v, played = make_voice()
    assert v.speak("Hello there")
    deadline = time.time() + 3
    while not played and time.time() < deadline:
        time.sleep(0.02)
    assert played == [(1600, 16000)]
    v.enabled = False
    assert not v.speak("Silent now")
    v.close()


def test_O3_real_piper_speaks_a_sentence():
    from kin.output.voice import PiperEngine

    eng = PiperEngine()
    if not eng.installed():
        pytest.skip("voice model not fetched")
    audio, rate = eng.synth("Yes. Rohan said Thursday at 4, and there is a 70 percent chance of rain then.")
    assert rate == 16000 and len(audio) / rate > 2.0 and int(abs(audio).max()) > 1000


def test_E29_answers_to_voice_commands_are_spoken_and_typed_ones_are_not(tmp_path):
    import queue

    from kin.audio.ring import Ring
    from kin.config import get_profile
    from kin.core.process import Core
    from kin.memory.service import Memory
    from kin.memory.store import Store
    from tests.test_core_memory import HashEmbedder, ScriptedASR
    from tests.test_core_reasoning import FakeReasoner

    class Reasoner(FakeReasoner):
        def handle(self, q):
            a = super().handle(q)
            a.voice_text = a.text
            return a

    voice, played = make_voice()
    ring = Ring.create(16000 * 5)
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    core = Core(ring, queue.Queue(), queue.Queue(), get_profile("phone"), transcriber=ScriptedASR(), memory=mem,
                reasoner=Reasoner(), voice=voice)
    try:
        core.answer("typed question", "typed")
        time.sleep(0.2)
        assert played == []
        core.answer("spoken question", "voice")
        deadline = time.time() + 3
        while not played and time.time() < deadline:
            time.sleep(0.02)
        assert len(played) == 1
    finally:
        voice.close()
        mem.store.close()
        ring.close()


# ---- O1 status / E37 ------------------------------------------------------------------------------


def test_O1_state_follows_priority_and_publishes_status_events():
    import queue

    ui, led = queue.Queue(), queue.Queue()
    s = StatusController(ui, led)
    assert PRIORITY == ["muted", "error", "fetching", "thinking", "capturing", "listening"]
    s.set("capturing")
    s.set("thinking")
    s.set("fetching")
    s.set("muted")
    assert s.state == "muted"
    s.set("muted", False)
    assert s.state == "fetching"
    s.set("fetching", False)
    s.set("thinking", False)
    s.set("capturing", False)
    assert s.state == "listening" and StatusEvent(state=s.state)
    shown = [ui.get_nowait()["state"] for _ in range(ui.qsize())]
    assert shown == ["capturing", "thinking", "fetching", "muted", "fetching", "thinking", "capturing", "listening"]


def test_E37_led_patterns_go_to_kin_audio_on_each_change():
    import queue

    ui, led = queue.Queue(), queue.Queue()
    s = StatusController(ui, led)
    for reason in ("capturing", "thinking", "fetching"):
        s.set(reason)
    s.set("muted")
    patterns = []
    while not led.empty():
        patterns.append(led.get_nowait()["pattern"])
    assert patterns == ["green_pulse", "blue_pulse", "amber_blink", "off"]
    assert led_command("amber_blink") == b"Lamber_blink\n"


def test_O1_capturing_times_out_when_no_command_follows():
    import queue

    t = [0.0]
    s = StatusController(queue.Queue(), None, clock=lambda: t[0])
    s.capture_started()
    t[0] = 10.0
    s.tick()
    assert s.state == "listening"


# ---- A6 clip serial / E05 / F4 mute -------------------------------------------------------------


class FakeSerial:
    def __init__(self):
        self.to_host = bytearray()
        self.written = bytearray()
        self.closed = False

    def read(self, n):
        if self.to_host:
            chunk, self.to_host = bytes(self.to_host[:n]), self.to_host[n:]
            return chunk
        time.sleep(0.01)
        return b""

    def write(self, data):
        self.written += data

    def close(self):
        self.closed = True


def test_A6_parses_clip_lines_and_writes_led_commands():
    assert parse_line("M1\r\n") == {"muted": True} and parse_line("M0") == {"muted": False}
    assert parse_line("HELLO kin-clip 1") == {"hello": "1"} and parse_line("garbage") is None
    ser = FakeSerial()
    got = []
    link = ClipLink(got.append, port="COM99", opener=lambda port: ser)
    link.start()
    try:
        link.set_led("green_solid")
        ser.to_host += b"HELLO kin-clip 1\nM1\nM0\nM1\n"
        deadline = time.time() + 3
        while len(got) < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert got == [True, False, True] and link.muted is True
        assert b"Lgreen_solid\n" in bytes(ser.written)
    finally:
        link.stop()


def test_A6_reconnects_and_resends_the_led_pattern_after_unplugging():
    sers = [FakeSerial(), FakeSerial()]
    opened = []

    def opener(port):
        s = sers[len(opened)]
        opened.append(s)
        return s

    link = ClipLink(lambda m: None, port="COM99", opener=opener)
    link.set_led("blue_pulse")
    link.start()
    try:
        time.sleep(0.3)
        sers[0].read = lambda n: (_ for _ in ()).throw(OSError("unplugged"))  # the cable is pulled
        deadline = time.time() + 5
        while len(opened) < 2 and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)
        assert len(opened) == 2 and b"Lblue_pulse\n" in bytes(sers[1].written)
    finally:
        link.stop()


def test_E05_mute_state_contract_and_latch():
    assert MuteState(muted=True, source="clip").muted
    with pytest.raises(ValidationError):
        MuteState(muted=True, source="bluetooth")


def test_F4_only_the_clips_switch_can_unmute_the_clip():
    latch = MuteLatch()
    assert latch.request(True, "clip") and latch.muted
    assert latch.request(False, "ui") is False and latch.muted  # the UI cannot re-enable a hardware mute
    assert latch.request(False, "os") is False and latch.muted
    assert latch.request(False, "clip") and not latch.muted
    assert latch.request(True, "ui") and latch.request(False, "ui")  # a UI mute is the UI's to lift


def test_F4_core_shows_muted_stops_speaking_and_status_goes_muted(tmp_path):
    import queue

    from kin.audio.ring import Ring
    from kin.config import get_profile
    from kin.core.process import Core
    from tests.test_core_memory import ScriptedASR

    ui = queue.Queue()
    voice, _ = make_voice()
    status = StatusController(ui)
    ring = Ring.create(16000 * 2)
    core = Core(ring, ui, queue.Queue(), get_profile("phone"), transcriber=ScriptedASR(), voice=voice, status=status)
    try:
        core.handle({"type": "mute", "muted": True, "source": "clip", "t": time.time()})
        assert core.muted and status.state == "muted"
        types = [ui.get_nowait()["type"] for _ in range(ui.qsize())]
        assert "mute" in types and "kin_state" in types
        core.handle({"type": "mute", "muted": False, "source": "clip", "t": time.time()})
        assert status.state == "listening"
    finally:
        voice.close()
        ring.close()


def test_degrade_clip_unplugged_falls_back_to_the_laptop_mic(monkeypatch):
    from kin.audio import capture

    class FakeMic:
        def __init__(self, ring, device=None, prefer_clip=True):
            self.device_name = "Laptop Microphone" if not prefer_clip else "Kin Clip"
            self.stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(capture, "MicSource", FakeMic)
    clip = FakeMic(None, None, True)
    new, ev = capture.fall_back_to_laptop_mic(clip, None)
    assert clip.stopped and new.device_name == "Laptop Microphone"
    assert ev["type"] == "audio_fallback" and ev["lost"] == "Kin Clip" and ev["device"] == "Laptop Microphone"


# ---- INV3 / INV4 / INV5 / INV10 --------------------------------------------------------------------


def run_py(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-800:]
    return out.stdout.strip().splitlines()[-1]


def test_INV3_guard_blocks_connections_and_name_lookups_but_not_local_sockets():
    res = json.loads(run_py(
        "import json,socket;from kin import guard;guard.install();p=guard.probe();"
        "ok=True\n"
        "try:\n a,b=socket.socketpair();a.close();b.close()\n"
        "except OSError:\n ok=False\n"
        "p['socketpair']=ok;print(json.dumps(p))"))  # fmt: skip
    assert res["blocked"] and res["dns_blocked"] and res["method"] in ("guard", "netns+guard")
    if sys.platform != "win32":  # Windows implements socketpair over loopback TCP, which the guard rightly refuses
        assert res["socketpair"]


def test_INV3_running_processes_report_that_the_network_is_blocked():
    from tests.test_pipeline import DATA, replay

    sup = replay(DATA / "command.wav", speed=4.0)
    ready = [m for m in sup.messages if m.get("type") == "audio_ready"]
    info = [m for m in sup.messages if m.get("type") == "core_info"]
    assert ready and ready[0]["net"]["blocked"] and ready[0]["net"]["dns_blocked"]
    assert info and info[0]["net"]["blocked"] and info[0]["net"]["dns_blocked"]


def test_INV4_only_kin_gateway_imports_http_libraries():
    import shutil

    exe = shutil.which("lint-imports", path=str(Path(sys.executable).parent))
    assert exe, "import-linter is not installed (dev dependency)"
    out = subprocess.run([exe], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-500:]
    assert "Contracts: 2 kept, 0 broken" in out.stdout


def test_INV5_no_cloud_llm_sdk_is_installed():
    import tomllib

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    names = {p["name"].lower() for p in lock["package"]}
    banned = {"openai", "anthropic", "google-generativeai", "google-genai", "cohere", "mistralai", "litellm", "langchain",
              "transformers", "torch", "vertexai", "groq"}
    assert not (names & banned), names & banned


def test_INV10_processes_run_offline_and_without_telemetry():
    from tests.test_pipeline import DATA, replay

    sup = replay(DATA / "command.wav", speed=4.0)
    ready = [m for m in sup.messages if m.get("type") == "audio_ready"][0]
    info = [m for m in sup.messages if m.get("type") == "core_info"][0]
    for env in (ready["env"], info["env"]):
        assert env == {"HF_HUB_OFFLINE": "1", "DO_NOT_TRACK": "1"}


# ---- F3 / F5 --------------------------------------------------------------------------------------


@pytest.mark.models
def test_F3_umbrella_question_uses_memory_and_a_typed_forecast_request(served_gateway, tmp_path):
    """Memory says Thursday 4 pm; the only thing sent out is {kind, lat, lon, hours}; the answer cites both."""
    from kin.memory.embed import Embedder
    from kin.memory.service import Memory
    from kin.memory.store import Store
    from kin.reasoning.calendar import Calendar
    from kin.reasoning.engine import Reasoner
    from kin.reasoning.facts import FactService
    from tests.engine_fixtures import ScriptedLLM, ts

    session, key, gw, net = served_gateway
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), Embedder())
    t = ts(29, 18, 12)
    mem.add(t, t + 4, "Rohan said the site visit is Thursday at 4 pm at Butibori.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    recorded = []
    facts = FactService(session, key, record=recorded.append)
    now = dt.datetime(2026, 9, 30, 10, 40)
    thursday = dt.datetime(2026, 10, 1, 16, 0)  # the mocked forecast puts 70% rain at Thursday 4 pm

    def handler(request):
        net.requests.append(request)
        return httpx.Response(200, text=weather_json(thursday))

    gw.egress._transport = httpx.MockTransport(handler)
    r = Reasoner(mem, ScriptedLLM("Rohan said Thursday at 4 pm, and there is a 70% chance of rain then."),
                 Calendar(None), facts=facts, clock=lambda: now)  # fmt: skip
    a = r.handle("Should I take an umbrella to the site visit Rohan mentioned?")
    assert a.intent == "recall.query+fact.weather" and a.gate == "grounded"
    assert "70%" in a.text and {c["id"] for c in a.citations} >= {"forecast"}
    assert a.fact["host"] == "api.open-meteo.com" and a.fact_request == {"kind": "weather", "lat": 21.1, "lon": 79.1, "hours": 48}
    sent = str(net.requests[0].url)
    assert "Rohan" not in sent and "umbrella" not in sent and "site" not in sent  # no question text, no names
    assert len(recorded) == 1 and recorded[0].ledger.host == "api.open-meteo.com"
    mem.store.close()


@pytest.mark.models
def test_F5_offline_answers_the_local_part_and_says_plainly_it_cannot_check(tmp_path):
    from kin.memory.embed import Embedder
    from kin.memory.service import Memory
    from kin.memory.store import Store
    from kin.reasoning.calendar import Calendar
    from kin.reasoning.engine import Reasoner
    from kin.reasoning.facts import FactService
    from tests.engine_fixtures import ScriptedLLM, ts

    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), Embedder())
    t = ts(29, 18, 12)
    mem.add(t, t + 4, "Rohan said the site visit is Thursday at 4 pm at Butibori.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    r = Reasoner(mem, ScriptedLLM("unused"), Calendar(None), facts=FactService(), clock=lambda: dt.datetime(2026, 9, 30, 10, 40))
    a = r.handle("Should I take an umbrella to the site visit Rohan mentioned?")
    assert a.gate == "local_part" and "Thursday at 4 pm at Butibori" in a.text
    assert a.text.endswith("I can't check the weather right now.")
    assert a.fact["status"] == "not sent" and a.abstained is False
    mem.store.close()


def test_degrade_no_network_answers_with_a_plain_statement_for_pure_lookups(tmp_path):
    from kin.reasoning.facts import FactService, unavailable_text

    resp = FactService().fetch(FactRequest(kind="news", topic="tech"))
    assert unavailable_text(resp) == "I can't check the news right now."


# ---- process supervision (CL8, sheet 8) --------------------------------------------------------------


def test_CL8_a_crashed_gateway_is_restarted_within_two_seconds():
    from kin.config import Settings
    from kin.supervisor import Supervisor
    from tests.test_pipeline import DATA

    sup = Supervisor(Settings(), source=str(DATA / "command.wav"), speed=0.2, ui=False)
    sup.start()
    try:
        child = next(c for c in sup.children if c.name == "kin-gateway")
        old = child.proc.pid
        child.proc.kill()
        t0 = time.time()
        while time.time() - t0 < 5:
            sup._check_children()
            if child.proc.pid != old and child.proc.is_alive():
                break
            time.sleep(0.05)
        assert child.proc.pid != old and child.proc.is_alive()
        assert time.time() - t0 < 2.0
    finally:
        sup.stop()


# ---- live services (skipped without a connection) -------------------------------------------------------


def _online() -> bool:
    try:
        httpx.get("https://api.open-meteo.com", timeout=3)
        return True
    except httpx.HTTPError:
        return False


@pytest.mark.online
def test_X1_X3_live_services_answer_through_the_real_gateway():
    if not _online():
        pytest.skip("no internet")
    gw = Gateway()
    w = FactResponse.model_validate(gw.handle({"request": {"kind": "weather", "lat": 21.1, "lon": 79.1, "hours": 12}}))
    n = FactResponse.model_validate(gw.handle({"request": {"kind": "news", "topic": "tech"}}))
    assert w.ok and len(w.data["hourly"]) >= 6 and n.ok and len(n.data["headlines"]) >= 3
    assert isinstance(w.ledger, LedgerEntry) and w.ledger.host == "api.open-meteo.com"
