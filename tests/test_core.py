"""kin-core logic with a fake ASR: ordering, privacy wipe, overrun handling."""

import queue
import threading
import time

import numpy as np
import pytest

from kin.audio.ring import Ring
from kin.config import get_profile
from kin.core.process import Core
from kin.perception.asr import AsrResult


class FakeASR:
    def __init__(self):
        self.seen = []
        self.gate = threading.Event()
        self.gate.set()

    def warm_up(self):
        pass

    def transcribe(self, audio):
        self.gate.wait(5)
        self.seen.append(int(audio[0]))
        return AsrResult(f"the segment {int(audio[0])}", len(audio) / 16000, 1.0)


@pytest.fixture
def setup():
    ring = Ring.create(16000 * 10)
    ui_q, ctl_q = queue.Queue(), queue.Queue()
    asr = FakeASR()
    core = Core(ring, ui_q, ctl_q, get_profile("phone"), transcriber=asr)
    yield ring, ui_q, ctl_q, asr, core
    core.stop()
    ring.close()


def seg(start, end, wake=False):
    return {"type": "segment", "start": start, "end": end, "wake": wake, "t": time.time()}


def eos():
    return {"type": "eos", "t": time.time()}


def drain(q, until_type, timeout=5):
    out, deadline = [], time.time() + timeout
    while time.time() < deadline:
        try:
            m = q.get(timeout=0.1)
        except queue.Empty:
            continue
        out.append(m)
        if m["type"] == until_type:
            return out
    raise AssertionError(f"no {until_type} in {out}")


def test_transcribes_and_wipes_audio(setup):
    ring, ui_q, ctl_q, asr, core = setup
    ring.write(np.full(16000 * 3, 5, dtype=np.int16))
    core.start()
    core.handle(seg(16000, 32000))
    core.handle(eos())
    msgs = drain(ui_q, "eos")
    (tr,) = [m for m in msgs if m["type"] == "transcript"]
    assert tr["text"] == "the segment 5" and tr["command"] is None
    assert (ring.read(16000, 32000) == 0).all(), "decoded audio must be zeroed"
    assert (ring.read(0, 16000) == 5).all(), "audio outside the segment is untouched"
    assert ctl_q.get(timeout=1)["type"] == "eos"


def test_commands_jump_the_queue(setup):
    ring, ui_q, _, asr, core = setup
    for v in (1, 2, 3):
        ring.write(np.full(1000, v, dtype=np.int16))
    asr.gate.clear()  # hold the lane while the other segments queue up
    core.start()
    core.handle(seg(0, 1000))  # picked up first, then blocks
    time.sleep(0.2)
    core.handle(seg(1000, 2000))  # ambient
    core.handle(seg(2000, 3000, wake=True))  # command
    core.handle(eos())
    asr.gate.set()
    msgs = drain(ui_q, "eos")
    assert asr.seen == [1, 3, 2]
    cmd = [m for m in msgs if m["type"] == "transcript" and m["wake"]][0]
    assert cmd["command"] == "The segment 3"


def test_overrun_is_reported_not_fatal(setup):
    ring, ui_q, _, asr, core = setup
    ring.write(np.zeros(16000 * 12, dtype=np.int16))  # laps the 10 s ring
    core.start()
    core.handle(seg(0, 8000))
    core.handle(eos())
    assert [m["type"] for m in drain(ui_q, "eos")] == ["dropped", "eos"]


def test_status_passes_straight_through(setup):
    _, ui_q, _, _, core = setup
    core.handle({"type": "status", "gate": True})
    assert ui_q.get(timeout=1) == {"type": "status", "gate": True}
