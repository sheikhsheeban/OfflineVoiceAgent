"""kin-core wiring for week 3: commands answered, commitments heard, reminders fired, task requests."""

import queue
import time

import numpy as np
import pytest

from kin.audio.ring import Ring
from kin.config import get_profile
from kin.core.process import Core
from kin.memory.service import Memory
from kin.memory.store import Store
from kin.reasoning.engine import Answer
from tests.test_core_memory import HashEmbedder, ScriptedASR


class FakeLLM:
    def __init__(self):
        self.loaded = False
        self.release = False

    def maybe_unload(self, now=None):
        if self.release and self.loaded:
            self.loaded = False
            return True
        return False


class FakeReasoner:
    def __init__(self):
        self.llm = FakeLLM()
        self.questions = []

    def handle(self, q):
        self.questions.append(q)
        self.llm.loaded = True
        return Answer(q, f"answer to: {q}", "recall.query", "centroid", gate="grounded")

    def warm(self):
        self.llm.loaded = True


@pytest.fixture
def rig(tmp_path):
    ring = Ring.create(16000 * 10)
    ui_q, ctl_q = queue.Queue(), queue.Queue()
    asr, reasoner = ScriptedASR(), FakeReasoner()
    mem = Memory(Store(tmp_path / "kin.db", "ab" * 32), HashEmbedder())
    core = Core(ring, ui_q, ctl_q, get_profile("phone"), transcriber=asr, memory=mem, reasoner=reasoner)
    core.start()
    yield ring, ui_q, asr, mem, core, reasoner
    core.stop()
    mem.store.close()
    ring.close()


def speak(rig, text, wake=False):
    ring, _, asr, _, core, _ = rig
    value = len(asr.texts) + 1
    asr.texts[value] = text
    start = ring.write_pos
    ring.write(np.full(1600, value, dtype=np.int16))
    core.handle({"type": "segment", "start": start, "end": start + 1600, "wake": wake, "t": time.time()})


def drain(rig, until, timeout=10):
    ui_q, core = rig[1], rig[4]
    core.handle({"type": "eos", "t": time.time()})
    out, deadline = [], time.time() + timeout
    while time.time() < deadline:
        try:
            m = ui_q.get(timeout=0.1)
        except queue.Empty:
            continue
        out.append(m)
        if m["type"] == until:
            return out
    raise AssertionError(out)


def test_voice_command_is_answered_after_its_transcript(rig):
    speak(rig, "Hey Jarvis, when is the site visit?", wake=True)
    msgs = drain(rig, "eos")
    kinds = [m["type"] for m in msgs if m["type"] in ("transcript", "answer")]
    assert kinds == ["transcript", "answer"]
    ans = next(m for m in msgs if m["type"] == "answer")
    assert ans["question"] == "When is the site visit?" and ans["source"] == "voice"
    assert rig[5].questions == ["When is the site visit?"]
    assert next(m for m in msgs if m["type"] == "llm_state")["loaded"] is True


def test_ambient_speech_is_not_answered(rig):
    speak(rig, "When is the site visit?")
    drain(rig, "eos")
    assert rig[5].questions == []


def test_heard_commitment_becomes_a_candidate(rig):
    speak(rig, "Sure, I will send the drawings to Rohan by Friday.")
    drain(rig, "eos")
    (t,) = rig[3].store.tasks(("candidate",))
    assert t["title"] == "Send the drawings to Rohan" and t["source"] == "heard" and t["due_at"]
    assert t["quote"] == "Sure, I will send the drawings to Rohan by Friday." and t["episode_id"]


def test_forgetting_speech_removes_what_was_derived_from_it(rig):
    *_, mem, core, _ = rig
    speak(rig, "I will call the plumber tomorrow.")
    speak(rig, "I will book the hall on Monday.")
    drain(rig, "eos")
    kept, dropped = sorted(mem.store.tasks(("candidate",)), key=lambda t: t["id"])
    core.rpc("task_set", {"id": kept["id"], "status": "open"})  # the user confirmed one
    core.rpc("forget", {"minutes": 10})
    assert mem.store.get_task(dropped["id"]) is None  # unconfirmed: gone with the speech
    k = mem.store.get_task(kept["id"])
    assert k["status"] == "open" and k["quote"] is None and k["episode_id"] is None  # kept, but not the quote


def test_reminders_fire_once_and_flag_late_ones(rig):
    _, ui_q, _, mem, core, _ = rig
    mem.store.add_reminder("leave for the site visit", time.time() - 1)
    mem.store.add_reminder("take medicine", time.time() - 3600)  # due while Kin was off
    mem.store.add_reminder("later", time.time() + 3600)
    core.tick()
    core.tick()
    fired = []
    while not ui_q.empty():
        m = ui_q.get()
        if m["type"] == "reminder":
            fired.append((m["text"], m["late"]))
    assert sorted(fired) == [("leave for the site visit", False), ("take medicine", True)]
    assert [r["text"] for r in mem.store.reminders()] == ["later"]


def test_task_requests(rig):
    *_, mem, core, reasoner = rig
    ans = core.rpc("ask", {"q": "  what's on my list  "})
    assert ans["type"] == "answer" and ans["source"] == "typed" and reasoner.questions == ["what's on my list"]
    note = core.rpc("note_add", {"text": "Buy a birthday gift"})
    assert note["source"] == "note" and note["status"] == "open"
    assert core.rpc("task_set", {"id": note["id"], "status": "done"}) == {"ok": True}
    assert [t["title"] for t in core.rpc("tasks", {})["done"]] == ["Buy a birthday gift"]
    rid = mem.store.add_reminder("x", time.time() + 99)
    assert core.rpc("reminder_cancel", {"id": rid}) == {"ok": True}
    assert core.rpc("tasks", {})["reminders"] == []
    with pytest.raises(ValueError):
        core.rpc("task_set", {"id": note["id"], "status": "deleted-by-magic"})
    with pytest.raises(ValueError):
        core.rpc("ask", {"q": "   "})


def test_idle_model_release_is_reported(rig):
    _, ui_q, _, _, core, reasoner = rig
    core.rpc("ask", {"q": "hello"})
    reasoner.llm.release = True
    core.tick()
    states = [m["loaded"] for m in list(ui_q.queue) if m["type"] == "llm_state"]
    assert states[-2:] == [True, False]
