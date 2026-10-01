"""Spec ID coverage for weeks 1 to 3 (`test_<ID>_<what>`): every component and edge of those weeks that
the older, topic-named tests do not name. Each one exercises the real component."""

import datetime as dt
import glob
import queue
import time

import numpy as np
import pytest

from kin.audio.capture import MicSource, WavSource, load_wav
from kin.audio.gate import EnergyGate
from kin.audio.ring import Ring
from kin.config import FRAME, SAMPLE_RATE, model_path
from kin.contracts import Intent, SpeechEvent
from kin.memory.service import Memory
from kin.memory.store import Store
from tests.conftest import DATA
from tests.test_core_memory import HashEmbedder
from tests.test_detector import pad, run_detector

# ---- Week 1 -------------------------------------------------------------------------------------------------


def test_HW4_laptop_mic_is_the_default_input_when_no_clip_is_named():
    import sounddevice as sd

    if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
        pytest.skip("no input device")
    ring = Ring.create(SAMPLE_RATE)
    try:
        mic = MicSource(ring, None, prefer_clip=False)
        assert mic.device is None and not mic.is_clip  # the OS default microphone
    finally:
        ring.close()


def test_E04_fallback_mic_feeds_the_same_capture_callback():
    ring = Ring.create(SAMPLE_RATE)
    try:
        mic = MicSource.__new__(MicSource)  # no device needed: E04 is "PCM from the laptop mic into A1"
        mic.ring, mic.overflows = ring, 0
        status = type("S", (), {"input_overflow": False})()
        mic._callback(np.full((FRAME, 1), 7, dtype=np.int16), FRAME, None, status)
        assert ring.write_pos == FRAME and int(ring.read(0, FRAME)[0]) == 7
    finally:
        ring.close()


def test_A1_capture_writes_16khz_mono_frames_into_the_ring():
    ring = Ring.create(10 * SAMPLE_RATE)
    src = WavSource(ring, DATA / "command.wav", speed=0, lead_s=0, tail_s=0)
    src.start()
    assert src.finished.wait(10)
    assert ring.write_pos >= 5 * SAMPLE_RATE and FRAME == 512  # 32 ms frames (the Silero window)
    ring.close()


def test_E06_frames_are_copied_into_the_ring_in_order():
    ring = Ring.create(SAMPLE_RATE)
    for i in range(5):
        ring.write(np.full(FRAME, i, dtype=np.int16))
    assert [int(ring.read(i * FRAME, i * FRAME + 1)[0]) for i in range(5)] == [0, 1, 2, 3, 4]
    ring.close()


def test_A2_ring_is_shared_memory_of_30_seconds_and_a_second_process_sees_the_same_samples():
    import multiprocessing as mp

    from kin.config import RING_SECONDS

    assert RING_SECONDS == 30
    ring = Ring.create(2000)
    ring.write(np.arange(100, dtype=np.int16))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child_reads_ring, args=(ring.name, q))
    p.start()
    assert q.get(timeout=20) == list(range(10))
    p.join(5)
    ring.close()


def _child_reads_ring(name, q):
    from kin.audio.ring import Ring as R

    r = R.attach(name)
    q.put([int(x) for x in r.read(0, 10)])
    r.close()


def test_E07_every_frame_reaches_the_energy_gate():
    g = EnergyGate()
    loud = np.full(FRAME, 3277, dtype=np.int16)  # 3277 / 32768 = -20 dBFS
    g.update(loud)
    assert abs(g.level - -20.0) < 0.5  # the gate measured exactly this frame
    g.update(np.zeros(FRAME, np.int16))
    assert g.level < -90  # and the next one


def test_A3_energy_gate_opens_on_loud_and_stays_shut_on_quiet():
    rng = np.random.default_rng(1)
    quiet = lambda: (rng.standard_normal(FRAME) * 15).astype(np.int16)  # noqa: E731
    loud = lambda: (rng.standard_normal(FRAME) * 4000).astype(np.int16)  # noqa: E731
    g = EnergyGate(calibration_frames=0)
    assert not any(g.update(quiet()) for _ in range(100))
    assert g.update(loud())


@pytest.mark.models
def test_E08_only_loud_frames_reach_the_vad():
    rng = np.random.default_rng(2)
    events, det = run_detector((rng.standard_normal(SAMPLE_RATE * 6) * 15).astype(np.int16))
    assert det.frames_vad / det.frames_total < 0.05 and events == []  # quiet frames stop at the gate


@pytest.mark.models
def test_A4_vad_marks_speech_and_ignores_loud_noise():
    from kin.audio.vad import SileroVAD

    vad = SileroVAD(model_path("vad", "silero_vad.onnx"))
    speech = load_wav(DATA / "command.wav")
    probs = [vad(speech[i : i + FRAME]) for i in range(0, len(speech) - FRAME, FRAME)]
    assert max(probs) > 0.9
    vad.reset()
    noise = (np.random.default_rng(3).standard_normal(SAMPLE_RATE * 2) * 3000).astype(np.int16)
    assert max(vad(noise[i : i + FRAME]) for i in range(0, len(noise) - FRAME, FRAME)) < 0.5


@pytest.mark.models
def test_E09_vad_speech_end_no_longer_gates_the_wake_word():
    """E09 (A4 -> A5) now carries only speech start/end, to close the command window; the wake word
    is fed by E41 whatever the gate and VAD say (see tests/test_A5_wake.py)."""
    events, det = run_detector(pad(load_wav(DATA / "hey_jarvis.wav"), lead=3.0, tail=3.0))
    assert any(e["type"] == "wake" for e in events)
    assert det.frames_vad < det.frames_total * 0.8  # VAD still gated, the wake word not


@pytest.mark.models
def test_A5_wake_word_marks_a_command_and_ambient_speech_has_no_mode_flag():
    cmd, _ = run_detector(pad(load_wav(DATA / "hey_jarvis.wav")))
    assert any(e["type"] == "wake" and e["score"] >= 0.5 for e in cmd)
    amb, _ = run_detector(pad(load_wav(DATA / "ambient.wav")))
    assert not any(e["type"] == "wake" for e in amb)
    assert all(e["wake"] is False for e in amb if e["type"] == "segment")


@pytest.mark.models
def test_E10_speech_event_refers_to_ring_offsets_and_carries_the_mode():
    events, _ = run_detector(pad(load_wav(DATA / "command.wav")))
    seg = next(e for e in events if e["type"] == "segment")
    ev = SpeechEvent(seq=1, start_offset=seg["start"], end_offset=seg["end"],
                     mode="command" if seg["wake"] else "ambient", t_start=seg["t"] - 1, t_end=seg["t"])  # fmt: skip
    assert ev.end_offset > ev.start_offset and ev.mode in ("command", "ambient")


@pytest.mark.models
def test_P1_speech_to_text_transcribes_a_recording_and_reports_timing():
    from kin.perception.asr import Transcriber

    asr = Transcriber(threads=2)
    asr.warm_up()
    res = asr.transcribe(load_wav(DATA / "command.wav"))
    assert "remind" in res.text.lower() and 0 < res.rtf < 1.0 and res.audio_s > 5


def test_E36_live_transcript_lines_reach_the_ui_queue(tmp_path):
    from kin.config import get_profile
    from kin.core.process import Core
    from tests.test_core_memory import ScriptedASR

    ring, ui_q = Ring.create(SAMPLE_RATE * 5), queue.Queue()
    asr = ScriptedASR()
    asr.texts[1] = "Hello Kin."
    core = Core(ring, ui_q, queue.Queue(), get_profile("phone"), transcriber=asr)
    ring.write(np.full(8000, 1, dtype=np.int16))
    core.handle({"type": "segment", "start": 0, "end": 8000, "wake": False, "t": time.time()})
    core._run_job(core.work.get()[2])
    msg = ui_q.get(timeout=5)
    assert msg["type"] == "transcript" and msg["text"] == "Hello Kin."
    ring.close()


def test_O4_ui_serves_transcript_and_answers_with_a_token_only():
    from fastapi.testclient import TestClient

    from kin.ui.app import Hub, create_app

    hub = Hub()
    c = TestClient(create_app("tok", hub))
    assert c.get("/").status_code == 403
    hub.publish({"type": "transcript", "text": "hello", "id": 1})
    hub.publish({"type": "answer", "question": "q", "text": "a", "id": 2})
    snap = c.get("/api/state?token=tok").json()
    assert [m["type"] for m in snap["transcripts"]] == ["transcript", "answer"]
    assert "Live transcript" in c.get("/?token=tok").text


def test_INV1_raw_audio_is_never_written_to_disk_by_a_session(tmp_path):
    import os

    from tests.test_pipeline import replay

    before = set(glob.glob(str(DATA.parent.parent / "**" / "*.wav"), recursive=True)) | set(
        glob.glob(os.environ["KIN_DATA_DIR"] + "/**/*", recursive=True))
    replay(DATA / "command.wav", speed=4.0)
    after = set(glob.glob(str(DATA.parent.parent / "**" / "*.wav"), recursive=True)) | set(
        glob.glob(os.environ["KIN_DATA_DIR"] + "/**/*", recursive=True))
    new = [f for f in after - before if f.lower().endswith((".wav", ".pcm", ".raw", ".flac", ".mp3"))]
    assert new == []


def test_INV2_the_ring_is_zeroed_after_a_segment_is_decoded(tmp_path):
    from kin.config import get_profile
    from kin.core.process import Core
    from tests.test_core_memory import ScriptedASR

    ring = Ring.create(SAMPLE_RATE * 5)
    asr = ScriptedASR()
    asr.texts[9] = "Some private words."
    core = Core(ring, queue.Queue(), queue.Queue(), get_profile("phone"), transcriber=asr)
    ring.write(np.full(4000, 9, dtype=np.int16))
    core._transcribe({"start": 0, "end": 4000, "wake": False, "t": time.time()})
    assert ring.read(0, 4000).shape == (4000,) and not ring.read(0, 4000).any()  # every sample is now zero
    ring.close()


def test_INV9_ui_needs_the_token_and_binds_loopback_only():
    from fastapi.testclient import TestClient

    from kin.ui.app import create_app, run

    c = TestClient(create_app("secret"))
    assert c.get("/api/state").status_code == 403 and c.get("/api/state?token=nope").status_code == 403
    assert c.get("/api/state?token=secret").status_code == 200
    with pytest.raises(SystemExit, match="loopback"):
        run(None, type("Q", (), {"cancel_join_thread": lambda self: None})(), None, "0.0.0.0", 1, "t")  # noqa: S104


# ---- Week 2 ---------------------------------------------------------------------------------------------------


def test_HW5_laptop_signals_come_from_psutil():
    from kin.context.fusion import ContextFusion
    from kin.reasoning.calendar import Calendar

    ev = ContextFusion(Calendar(None)).sample()
    assert isinstance(ev.on_ac, bool) and ev.idle_s >= 0 and ev.in_meeting is False


def test_P3_sensitive_numbers_are_removed_before_anything_is_stored():
    from kin.perception.filters import redact

    r = redact("My OTP is 482913 and my card is 4111 1111 1111 1111, call me at four.")
    assert "482913" not in r.text and "4111" not in r.text and r.redactions >= 2


def test_P4_segmenter_closes_an_episode_after_a_gap_and_opens_a_new_one(tmp_path):
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    a = mem.add(1000.0, 1004.0, "first thing", is_command=False, redactions=0)
    b = mem.add(1010.0, 1014.0, "second thing right after", is_command=False, redactions=0)
    c = mem.add(1000.0 + 900, 1004.0 + 900, "much later", is_command=False, redactions=0)
    assert a["episode_id"] == b["episode_id"] != c["episode_id"] and a["episode_id"] in c["closed"]
    mem.store.close()


def test_E13_redacted_text_is_what_the_segmenter_receives(tmp_path):
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    from kin.perception.filters import redact

    f = redact("The code is 123456.")
    mem.add(1.0, 2.0, f.text, is_command=False, redactions=f.redactions)
    assert "123456" not in mem.store.timeline()[0]["text"]
    mem.store.close()


def test_M1_embedder_gives_unit_384d_vectors_and_similar_sentences_score_higher(embedder):
    a = embedder.embed_one("The site visit is on Thursday afternoon")
    b = embedder.embed_one("When is the site visit?")
    c = embedder.embed_one("I like pasta with cheese")
    assert a.shape == (384,) and abs(float(np.linalg.norm(a)) - 1) < 1e-3 and float(a @ b) > float(a @ c)


def test_E14_E15_E16_a_closed_episode_is_embedded_and_its_vector_stored(tmp_path):
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    mem.add(1000.0, 1004.0, "Rohan said the site visit is Thursday.", is_command=False, redactions=0)
    (ep,) = mem.store.close_all()  # E14: the episode closes
    assert mem.store.pending_embeddings() == [ep]  # E15: its text is queued for the embedder
    assert mem.embed_episode(ep)  # E16: the vector is written to M2
    assert mem.store.pending_embeddings() == [] and mem.store.stats()["embedded"] == 1
    mem.store.close()


def test_E17_task_candidates_and_people_mentions_are_written_to_memory(core_rig_w1):
    core, ring, ui_q, asr, mem = core_rig_w1
    from tests.test_w3_gaps import say

    say(core, ring, asr, "I will send the drawings by Friday. Rohan said the hall is booked.")
    core._run_job(core.work.get()[2])
    assert [t["title"] for t in mem.store.tasks(("candidate",))] == ["Send the drawings"]
    mem.store.close_all()
    mem.embed_episode(mem.store.timeline()[0]["id"])
    assert [p["name"] for p in mem.store.people()] == ["Rohan"]


@pytest.fixture
def core_rig_w1(tmp_path):
    from kin.config import get_profile
    from kin.core.process import Core
    from tests.test_core_memory import ScriptedASR

    ring = Ring.create(SAMPLE_RATE * 10)
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    core = Core(ring, queue.Queue(), queue.Queue(), get_profile("phone"), transcriber=ScriptedASR(), memory=mem)
    yield core, ring, core.ui_q, core.asr, mem
    mem.store.close()
    ring.close()


def test_E18_the_consolidator_reads_and_writes_memory(tmp_path):
    from kin.memory.consolidate import Consolidator

    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    old = time.time() - 9 * 86400
    mem.add(old, old + 3, "An old conversation.", is_command=False, redactions=0)
    mem.add(time.time() - 60, time.time() - 57, "A new conversation.", is_command=False, redactions=0)
    mem.store.close_all()
    res = Consolidator(mem).run()
    assert res["expired_episodes"] == 1 and len(mem.store.timeline()) == 1 and len(res["digests"]) == 1
    mem.store.close()


def test_E38_user_actions_from_the_ui_change_memory_through_kin_core(tmp_path):
    from kin.config import get_profile
    from kin.core.process import Core

    ring = Ring.create(SAMPLE_RATE)
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    mem.add(1000.0, 1003.0, "delete me please", is_command=False, redactions=0)
    core = Core(ring, queue.Queue(), queue.Queue(), get_profile("phone"), transcriber=None, memory=mem)
    (ep,) = [e["id"] for e in core.rpc("timeline", {})]
    task = core.rpc("note_add", {"text": "buy milk"})
    assert core.rpc("task_set", {"id": task["id"], "status": "done"})["ok"]
    assert core.rpc("delete_episode", {"id": ep}) == {"deleted": True} and core.rpc("timeline", {}) == []
    assert core.rpc("wipe", {"confirm": "wipe"}) == {"wiped": True}
    mem.store.close()
    ring.close()


def test_INV6_the_memory_file_is_not_readable_without_the_key(tmp_path):
    import sqlite3

    store = Store(tmp_path / "k.db", "ab" * 32)
    store.close()
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(tmp_path / "k.db").execute("SELECT * FROM sqlite_master").fetchall()


def test_F1_ambient_memory_flow_stores_episodes_without_a_wake_word_and_sends_nothing_online():
    from tests.test_pipeline import replay

    sup = replay(DATA / "ambient.wav", speed=4.0)
    tr = [m for m in sup.messages if m.get("type") == "transcript"]
    assert tr and not any(m["wake"] for m in tr) and any(m["episode_id"] for m in tr)
    assert not any(m.get("type") in ("ledger", "answer") for m in sup.messages)  # nothing answered, nothing online


def test_F8_deleting_memory_removes_text_and_vectors(tmp_path):
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    mem.add(1000.0, 1003.0, "My secret plan is the moon.", is_command=False, redactions=0)
    mem.store.close_all()
    (ep,) = mem.store.pending_embeddings()
    mem.embed_episode(ep)
    mem.store.delete_episode(ep)
    st = mem.store
    assert st.timeline() == [] and st.keyword_search("secret") == []
    assert st._q("SELECT count(*) FROM episode_vec")[0][0] == 0 if st.vec_backend == "sqlite-vec" else True
    st.close()


# ---- Week 3 -----------------------------------------------------------------------------------------------------


def test_HW6_local_calendar_is_read_only_and_local():
    from kin.reasoning.calendar import Calendar

    evs = Calendar(DATA / "demo_calendar.ics").events()
    assert [e.title for e in evs][:1] == ["Site visit"] and evs[0].location == "Butibori"


def test_E34_calendar_entries_reach_the_context_builder(embedder, tmp_path):
    from kin.reasoning import context as ctx
    from kin.reasoning.calendar import Calendar

    cal = Calendar(DATA / "demo_calendar.ics")
    items = ctx.build([], cal.relevant("when is the dentist", dt.datetime(2026, 9, 30, 10, 0)), "dentist")
    assert items and items[0].kind == "calendar" and "Sitabuldi" in items[0].text


@pytest.mark.models
def test_R1_router_picks_the_intent_from_centroids_and_rules(embedder):
    from kin.reasoning.router import Router

    r = Router(embedder, None)
    assert r.route("What did Priya say about the budget?").intent == "recall.query"
    assert r.route("In 20 minutes, tell me to leave").intent == "reminder.set"
    assert r.route("Draft my tax return").intent == "unsupported"
    top = r.route("What's the weather like on Thursday?")
    assert top.intent == "fact.weather" and Intent(label=top.intent, score=0.5)


@pytest.mark.models
def test_E19_only_command_utterances_reach_the_router(tmp_path):
    from kin.config import get_profile
    from kin.core.process import Core
    from tests.test_core_memory import ScriptedASR
    from tests.test_core_reasoning import FakeReasoner

    ring, ui_q = Ring.create(SAMPLE_RATE * 5), queue.Queue()
    asr, reasoner = ScriptedASR(), FakeReasoner()
    asr.texts[1], asr.texts[2] = "Hey Jarvis, what time is it in Nagpur?", "Just chatting about lunch."
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    core = Core(ring, ui_q, queue.Queue(), get_profile("phone"), transcriber=asr, memory=mem, reasoner=reasoner)
    for i, wake in ((1, True), (2, False)):
        ring.write(np.full(8000, i, dtype=np.int16))
        core._transcribe({"start": (i - 1) * 8000, "end": i * 8000, "wake": wake, "t": time.time()})
    while not core.work.empty():
        core._run_job(core.work.get()[2])
    assert reasoner.questions == ["What time is it in Nagpur?"]  # the ambient line never reached R1
    mem.store.close()
    ring.close()


@pytest.mark.models
def test_E20_R2_E21_intent_retrieval_and_context(make):
    r, mem = make(None)
    q = "When is the site visit Rohan mentioned?"
    hits = mem.search(q, k=4, qvec=mem.embedder.embed_one(q))  # E21: top-k from M2
    assert hits and "Butibori" in hits[0]["text"]
    from kin.reasoning import context as ctx

    items = ctx.build(hits, [], q)  # R2: labelled, citable context
    assert items[0].id.startswith("e-") and ctx.render(items).startswith("[e-")


@pytest.mark.models
def test_E22_R3_the_prompt_is_short_and_the_local_model_answers(tmp_path):
    from kin.reasoning.llm import LocalLLM, load_prompt

    prompt = (load_prompt("answer").replace("{{EXTRA}}", "").replace("{{NOW}}", "")
              .replace("{{CONTEXT}}", "[e-1] Heard Tue 29 Sep 18:12: Rohan said the site visit is Thursday at 4 pm.")
              .replace("{{QUESTION}}", "When is the site visit?"))  # fmt: skip
    llm = LocalLLM(threads=4)
    if not llm.installed():
        pytest.skip("language model not fetched")
    assert len(prompt.split()) < 200  # about 1k tokens is the ceiling
    out = llm.complete(prompt, stop=["\n\n"])
    assert "Thursday" in out.text and out.ms < 15000
    llm.unload()


@pytest.mark.models
def test_E23_R4_grammar_constrained_tool_calls_become_valid_requests():
    from kin.contracts import FactRequest
    from kin.reasoning.facts import build_request
    from kin.reasoning.llm import LocalLLM

    llm = LocalLLM(threads=4)
    if not llm.installed():
        pytest.skip("language model not fetched")
    req = build_request("fact.weather", "Will it rain?", dt.datetime(2026, 9, 30, 10, 0), llm=llm)
    assert isinstance(req, FactRequest) and req.kind == "weather" and req.lat == 21.1
    llm.unload()


def test_E24_tool_results_are_written_to_memory(make):
    from tests.engine_fixtures import ScriptedLLM

    r, mem = make(ScriptedLLM())
    r.handle("Remind me to call the landlord after work")
    r.handle("In 20 minutes, tell me to leave")
    assert [t["title"] for t in mem.store.tasks(("open",))] == ["Call the landlord"]
    assert [x["text"] for x in mem.store.reminders()] == ["leave"]


@pytest.mark.models
def test_R5_E28_E30_the_gate_checks_sources_before_the_answer_reaches_the_ui(make):
    from tests.engine_fixtures import ScriptedLLM

    r, _ = make(ScriptedLLM("The site visit is on Wednesday at 4 pm."))  # invented day
    a = r.handle("When is the site visit Rohan mentioned?")
    assert a.gate == "ungrounded" and "Wednesday" not in a.text and a.citations  # E28: draft + sources checked
    msg = a.to_dict()
    assert msg["type"] == "answer" and msg["citations"][0]["id"].startswith("e-")  # E30: answer + citations to the UI


def test_G1_stub_without_a_gateway_always_answers_offline():
    from kin.contracts import FactRequest
    from kin.reasoning.facts import FactService

    for kind, extra in (("weather", {"lat": 21.1, "lon": 79.1}), ("news", {}), ("search", {"q": "x"})):
        r = FactService().fetch(FactRequest(kind=kind, **extra))
        assert not r.ok and r.error == "offline"  # CL5: the stub has no network code at all


def test_F2_command_answered_fully_offline_with_the_network_off(tmp_path, monkeypatch):
    import socket

    def refuse(*a, **k):
        raise OSError("network is off")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    from kin.memory.embed import Embedder
    from kin.reasoning.calendar import Calendar
    from kin.reasoning.engine import Reasoner
    from tests.engine_fixtures import ScriptedLLM, ts

    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), Embedder())
    mem.add(ts(29, 18, 12), ts(29, 18, 16), "Rohan said the site visit is Thursday at 4 pm at Butibori.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    r = Reasoner(mem, ScriptedLLM("Thursday at 4 pm at Butibori."), Calendar(None), clock=lambda: dt.datetime(2026, 9, 30, 10, 0))
    a = r.handle("When is the site visit Rohan mentioned?")
    assert a.gate == "grounded" and "Butibori" in a.text and a.fact_request is None
    mem.store.close()
