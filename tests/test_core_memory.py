"""kin-core's memory path with a fake ASR and embedder: what gets stored, when, and the UI requests."""

import hashlib
import queue
import time

import numpy as np
import pytest

from kin.audio.ring import Ring
from kin.config import get_profile
from kin.core.process import Core
from kin.memory.service import Memory
from kin.memory.store import Store
from kin.perception.asr import AsrResult
from kin.perception.segmenter import EpisodePolicy


class ScriptedASR:
    """Returns the text registered for the first sample value of a segment."""

    def __init__(self):
        self.texts: dict[int, str] = {}

    def warm_up(self):
        pass

    def transcribe(self, audio):
        return AsrResult(self.texts[int(audio[0])], len(audio) / 16000, 1.0)


class HashEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_one(self, text):
        self.calls += 1
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture
def rig(tmp_path):
    ring = Ring.create(16000 * 10)
    ui_q, ctl_q = queue.Queue(), queue.Queue()
    asr, emb = ScriptedASR(), HashEmbedder()
    mem = Memory(Store(tmp_path / "kin.db", "ab" * 32), emb, EpisodePolicy())
    core = Core(ring, ui_q, ctl_q, get_profile("phone"), transcriber=asr, memory=mem)
    core.start()
    yield ring, ui_q, ctl_q, asr, mem, core
    core.stop()
    mem.store.close()
    ring.close()


def speak(rig, text, wake=False):
    ring, _, _, asr, _, core = rig
    value = len(asr.texts) + 1
    asr.texts[value] = text
    start = ring.write_pos
    ring.write(np.full(1600, value, dtype=np.int16))
    core.handle({"type": "segment", "start": start, "end": start + 1600, "wake": wake, "t": time.time()})


def finish(rig):
    """Send end-of-stream and collect UI messages until it arrives."""
    _, ui_q, _, _, _, core = rig
    core.handle({"type": "eos", "t": time.time()})
    out, deadline = [], time.time() + 10
    while time.time() < deadline:
        try:
            m = ui_q.get(timeout=0.1)
        except queue.Empty:
            continue
        out.append(m)
        if m["type"] == "eos":
            return out
    raise AssertionError(out)


def test_ambient_speech_is_stored_as_an_episode(rig):
    speak(rig, "Priya will send the budget slides tomorrow")
    speak(rig, "The review is on Friday")
    msgs = finish(rig)
    trs = [m for m in msgs if m["type"] == "transcript"]
    assert trs[0]["episode_id"] == trs[1]["episode_id"] is not None
    assert any(m["type"] == "memory_changed" for m in msgs)
    (ep,) = rig[4].store.timeline()
    assert ep["text"] == "Priya will send the budget slides tomorrow The review is on Friday"
    assert ep["closed"] and ep["embedded"], "eos arrives only after the episode is indexed"


def test_commands_are_stored_without_the_wake_phrase(rig):
    speak(rig, "Hey, Jarvis.", wake=True)  # wake word alone: nothing to remember
    speak(rig, "Hey Jarvis, remind me to call Rohan", wake=True)
    msgs = finish(rig)
    trs = [m for m in msgs if m["type"] == "transcript"]
    assert trs[0]["episode_id"] is None and trs[0]["command"] == ""
    (ep,) = rig[4].store.timeline()
    (u,) = ep["utterances"]
    assert u["text"] == "Remind me to call Rohan" and u["command"]


def test_codes_are_filtered_before_ui_and_store(rig):
    speak(rig, "Your OTP is 482913 do not share it")
    msgs = finish(rig)
    (tr,) = [m for m in msgs if m["type"] == "transcript"]
    assert "482913" not in tr["text"] and tr["redactions"] == 1
    (ep,) = rig[4].store.timeline()
    assert "482913" not in ep["text"] and ep["utterances"][0]["redactions"] == 1
    assert rig[4].store.keyword_search("482913") == []


def test_ui_requests(rig):
    *_, mem, core = rig
    speak(rig, "Rohan said the site visit is at Butibori")
    finish(rig)
    (ep,) = core.rpc("timeline", {})
    hits = core.rpc("search", {"q": "Butibori"})
    assert hits[0]["id"] == ep["id"] and hits[0]["keyword_match"]
    assert hits[0]["vector_similarity"] is not None  # the query was embedded on the heavy lane
    assert core.rpc("stats", {})["episodes"] == 1
    assert core.rpc("search", {"q": "   "}) == []
    assert core.rpc("delete_episode", {"id": ep["id"]}) == {"deleted": True}
    assert core.rpc("timeline", {}) == []
    with pytest.raises(ValueError):
        core.rpc("wipe", {})
    with pytest.raises(ValueError):
        core.rpc("drop_tables", {})


def test_forget_last_minutes(rig):
    *_, mem, core = rig
    old = mem.add(time.time() - 3600, time.time() - 3590, "an hour ago", is_command=False, redactions=0)
    speak(rig, "just now")
    finish(rig)
    res = core.rpc("forget", {"minutes": 10})
    assert res["utterances"] == 1
    assert [e["id"] for e in core.rpc("timeline", {})] == [old["episode_id"]]


def test_wipe(rig):
    *_, core = rig
    speak(rig, "something private")
    finish(rig)
    assert core.rpc("wipe", {"confirm": "wipe"}) == {"wiped": True}
    assert core.rpc("stats", {})["episodes"] == 0


def test_open_episodes_from_a_previous_run_are_closed_and_indexed(tmp_path):
    store = Store(tmp_path / "kin.db", "ab" * 32)
    Memory(store, None).add(time.time() - 5, time.time(), "left open by a crash", is_command=False, redactions=0)
    ring = Ring.create(16000)
    emb = HashEmbedder()
    core = Core(ring, queue.Queue(), queue.Queue(), get_profile("phone"), transcriber=ScriptedASR(),
                memory=Memory(store, emb))  # fmt: skip
    try:
        core.start()
        deadline = time.time() + 5
        while not store.timeline()[0]["embedded"] and time.time() < deadline:
            time.sleep(0.05)
        assert store.timeline()[0]["closed"] and store.timeline()[0]["embedded"]
    finally:
        core.stop()
        store.close()
        ring.close()


def test_old_stored_numbers_are_scrubbed_at_start(tmp_path):
    store = Store(tmp_path / "kin.db", "ab" * 32)
    Memory(store, None).add(time.time() - 60, time.time() - 55, "My credit card number is 1234567",
                            is_command=False, redactions=0)  # fmt: skip
    store.set_meta("filter_version", "1")  # stored by an older Kin
    ring = Ring.create(16000)
    core = Core(ring, queue.Queue(), queue.Queue(), get_profile("phone"), transcriber=ScriptedASR(),
                memory=Memory(store, HashEmbedder()))  # fmt: skip
    try:
        core.start()
        (ep,) = store.timeline()
        assert "1234567" not in ep["text"] and "[card details removed]" in ep["text"]
        from kin.perception.filters import FILTER_VERSION

        assert store.get_meta("filter_version") == str(FILTER_VERSION)
    finally:
        core.stop()
        store.close()
        ring.close()
