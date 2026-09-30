"""The kin-audio pipeline in-process: ring -> gate -> VAD -> wake word -> endpoints."""

import numpy as np
import pytest

from kin.audio.capture import load_wav
from kin.audio.ring import Ring
from kin.config import FRAME, SAMPLE_RATE, model_path
from tests.conftest import DATA

pytestmark = pytest.mark.models


def run_detector(audio: np.ndarray, wake=True):
    from kin.audio.detector import Detector
    from kin.audio.vad import SileroVAD
    from kin.audio.wakeword import WakeWord

    ring = Ring.create(30 * SAMPLE_RATE)
    det = Detector(ring, SileroVAD(model_path("vad", "silero_vad.onnx")), WakeWord.from_spec() if wake else None)
    events = []
    try:
        for i in range(0, len(audio) - FRAME + 1, FRAME):
            ring.write(audio[i : i + FRAME])
            events += det.process(i, ring.read(i, i + FRAME))
        events += det.flush(ring.write_pos)
        return events, det
    finally:
        ring.close()


def pad(x, lead=1.0, tail=1.5):
    def z(s):
        return np.zeros(int(s * SAMPLE_RATE), np.int16)

    return np.concatenate([z(lead), x, z(tail)])


def test_silence_produces_nothing_and_skips_vad():
    rng = np.random.default_rng(2)
    quiet = (rng.standard_normal(SAMPLE_RATE * 10) * 20).astype(np.int16)  # about -64 dBFS room
    events, det = run_detector(quiet)
    assert events == []
    assert det.frames_vad / det.frames_total < 0.05  # the gate kept VAD off


def test_ambient_speech_segments_without_wake():
    speech = load_wav(DATA / "ambient.wav")
    events, _ = run_detector(pad(speech))
    segs = [e for e in events if e["type"] == "segment"]
    assert segs and not any(e["type"] == "wake" for e in events)
    assert not any(s["wake"] for s in segs)
    covered = sum(s["end"] - s["start"] for s in segs)
    assert covered > 0.7 * len(speech)


def test_wake_then_command_marks_both_segments():
    events, _ = run_detector(pad(load_wav(DATA / "command.wav")))
    wakes = [e for e in events if e["type"] == "wake"]
    segs = [e for e in events if e["type"] == "segment"]
    assert len(wakes) == 1
    assert segs and all(s["wake"] for s in segs)
    assert segs[0]["start"] <= wakes[0]["at"] <= segs[0]["end"] + int(0.3 * SAMPLE_RATE)


def test_wake_arming_expires():
    phrase = load_wav(DATA / "hey_jarvis.wav")
    later = load_wav(DATA / "ambient.wav")
    gap = np.zeros(8 * SAMPLE_RATE, np.int16)  # longer than the 5 s arm window
    events, _ = run_detector(pad(np.concatenate([phrase, gap, later])))
    segs = [e for e in events if e["type"] == "segment"]
    assert segs[0]["wake"]
    assert len(segs) > 1 and not any(s["wake"] for s in segs[1:])
