"""Two live-testing fixes: the recogniser as a second chance for the wake word, and shedding memory
instead of dying when kin-core reaches the phone-profile ceiling."""

import queue
import threading
import time

import numpy as np

from kin.audio.ring import Ring
from kin.config import get_profile
from kin.core.process import Core
from kin.memory.service import Memory
from kin.memory.store import Store
from kin.reasoning.llm import LocalLLM
from tests.test_core_memory import HashEmbedder, ScriptedASR
from tests.test_core_reasoning import FakeReasoner
from tests.test_w4_ids import make_voice


def build(tmp_path, **kw):
    ring, ui_q = Ring.create(16000 * 10), queue.Queue()
    asr = ScriptedASR()
    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), HashEmbedder())
    core = Core(ring, ui_q, queue.Queue(), get_profile("phone"), transcriber=asr, memory=mem, **kw)
    return core, ring, ui_q, asr, mem


def next_transcript(ui_q):
    while True:
        m = ui_q.get(timeout=5)
        if m["type"] == "transcript":
            return m


def hear(core, ring, asr, text, wake=False):
    value = len(asr.texts) + 1
    asr.texts[value] = text
    start = ring.write_pos
    ring.write(np.full(16000, value, dtype=np.int16))
    core.handle({"type": "segment", "start": start, "end": start + 16000, "wake": wake, "t": time.time()})
    core._run_job(core.work.get()[2])


def test_the_recogniser_hearing_hey_jarvis_counts_when_the_acoustic_wake_word_missed(tmp_path):
    reasoner = FakeReasoner()
    core, ring, ui_q, asr, mem = build(tmp_path, reasoner=reasoner)
    hear(core, ring, asr, "Hey, Jarvis, when I was born,", wake=False)  # the detector missed it
    msg = next_transcript(ui_q)
    assert msg["wake"] and msg["wake_miss"] and msg["command"] == "When I was born,"
    core._run_job(core.work.get()[2])  # the answer job
    assert reasoner.questions == ["When I was born,"]
    assert mem.store.timeline()[0]["utterances"][0]["command"] is True
    mem.store.close()
    ring.close()


def test_ordinary_speech_that_only_contains_the_words_is_not_a_command(tmp_path):
    reasoner = FakeReasoner()
    core, ring, ui_q, asr, mem = build(tmp_path, reasoner=reasoner)
    for text in ("They jarvis said it was fine.", "I told him hey jarvis is a funny name."):
        hear(core, ring, asr, text)
        assert not next_transcript(ui_q)["wake"]
    assert reasoner.questions == []
    mem.store.close()
    ring.close()


def test_a_transcript_starting_with_just_jarvis_is_rerouted_and_logged_as_a_wake_miss(tmp_path, caplog):
    reasoner = FakeReasoner()
    core, ring, ui_q, asr, mem = build(tmp_path, reasoner=reasoner)
    with caplog.at_level("WARNING", logger="kin.core"):
        hear(core, ring, asr, "Jarvis, what is on my list?", wake=False)
    msg = next_transcript(ui_q)
    assert msg["mode"] == "command" and msg["wake_miss"] and msg["command"] == "What is on my list?"
    assert any("wake_miss" in r.message for r in caplog.records)
    mem.store.close()
    ring.close()


def test_a_command_window_drops_the_end_of_an_ambient_sentence_before_the_phrase(tmp_path):
    reasoner = FakeReasoner()
    core, ring, ui_q, asr, mem = build(tmp_path, reasoner=reasoner)
    hear(core, ring, asr, "it was fine. Hey Jarvis, when is my birthday?", wake=True)
    assert next_transcript(ui_q)["command"] == "When is my birthday?"
    mem.store.close()
    ring.close()


def test_kins_own_voice_saying_the_phrase_never_wakes_it(tmp_path):
    voice, _ = make_voice()
    core, ring, ui_q, asr, mem = build(tmp_path, reasoner=FakeReasoner(), voice=voice)
    voice.speak("Say hey jarvis to wake me")
    hear(core, ring, asr, "Hey Jarvis, say hey jarvis to wake me")
    msg = next_transcript(ui_q)
    assert msg["speaker"] == "kin" and not msg["wake"]
    voice.close()
    mem.store.close()
    ring.close()


class FakeEngine:
    def __init__(self):
        self.loaded = True

    def unload(self):
        self.loaded = False


class FakeLLM:
    def __init__(self):
        self.loaded = True

    def unload_if_idle(self):
        self.loaded = False
        return True


def test_memory_pressure_releases_the_voice_and_language_models_before_anything_dies(tmp_path):
    reasoner = FakeReasoner()
    reasoner.llm = FakeLLM()
    voice, _ = make_voice()
    voice.engine = FakeEngine()
    core, ring, ui_q, asr, mem = build(tmp_path, reasoner=reasoner, voice=voice)
    assert core.shed_memory() == ["voice model", "language model"]
    assert not voice.engine.loaded and not reasoner.llm.loaded
    assert core.shed_memory() == []  # nothing left to give back
    voice.close()
    mem.store.close()
    ring.close()


def test_the_language_model_is_not_released_while_it_is_answering():
    llm = LocalLLM(threads=1)
    llm._llm = object()  # pretend it is loaded
    started, release = threading.Event(), threading.Event()

    def answering():
        with llm._lock:
            started.set()
            release.wait(5)

    t = threading.Thread(target=answering)
    t.start()
    started.wait(5)
    assert llm.unload_if_idle() is False and llm._llm is not None  # busy: leave it, and do not block
    release.set()
    t.join()
    llm._llm = None
