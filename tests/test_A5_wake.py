"""A5, the wake word, on its new edge E41: every 80 ms chunk from the ring, ungated, in parallel with
the gate and VAD. Real models; audio fed in 1280-sample blocks as the microphone delivers it."""

import numpy as np
import pytest

from kin.audio.capture import load_wav
from kin.audio.ring import Ring
from kin.config import CHUNK, FRAME, ROOT, SAMPLE_RATE, AudioSettings, model_path
from tests.conftest import DATA

pytestmark = pytest.mark.models

AUDIO = ROOT / "tests" / "audio"
S = SAMPLE_RATE


def detect(audio: np.ndarray, settings: AudioSettings = AudioSettings(), wake=None, block: int = CHUNK):
    from kin.audio.detector import Detector
    from kin.audio.vad import SileroVAD
    from kin.audio.wakeword import WakeWord

    ring = Ring.create(30 * S)
    det = Detector(ring, SileroVAD(model_path("vad", "silero_vad.onnx")),
                   wake if wake is not None else WakeWord.from_spec(settings.wake_threshold), settings)  # fmt: skip
    events = []
    try:
        for i in range(0, len(audio) - block + 1, block):
            ring.write(audio[i : i + block])
            events += det.process(i, ring.read(i, i + block))
        events += det.flush(ring.write_pos)
        return events, det
    finally:
        ring.close()


def silence(s: float) -> np.ndarray:
    return np.zeros(int(s * S), np.int16)


def trim(x: np.ndarray, level: int = 300) -> np.ndarray:
    """The clip without its leading and trailing silence, so the pauses a test adds are the real ones."""
    loud = np.flatnonzero(np.abs(x) > level)
    return x[loud[0] : loud[-1] + 1]


def kinds(events, t):
    return [e for e in events if e["type"] == t]


def test_A5_detects_wake_clip():
    clip = load_wav(AUDIO / "wake.wav")
    events, _ = detect(np.concatenate([silence(1), clip, silence(2)]))
    wakes = kinds(events, "wake")
    assert len(wakes) == 1 and wakes[0]["score"] >= 0.5
    segs = kinds(events, "segment")
    assert [s["mode"] for s in segs] == ["command"]
    assert segs[0]["start"] <= S + int(0.2 * S)  # the command starts at the phrase, not after it


def test_A5_wake_mid_utterance_sets_command_mode():
    ambient = load_wav(DATA / "ambient.wav")[: 3 * S]  # talking, and straight into the wake word
    phrase = trim(load_wav(AUDIO / "wake.wav"))
    question = trim(load_wav(DATA / "asr_ref" / "0.wav"))[: 2 * S]
    audio = np.concatenate([silence(1), ambient, phrase, silence(0.5), question, silence(3)])
    events, _ = detect(audio)

    wake = kinds(events, "wake")
    assert len(wake) == 1
    at = wake[0]["at"]
    phrase_start, question_end = S + 3 * S, S + 3 * S + len(phrase) + S // 2 + 2 * S
    segs = kinds(events, "segment")
    amb = [s for s in segs if s["mode"] == "ambient"]
    cmd = [s for s in segs if s["mode"] == "command"]
    assert amb and cmd and len(cmd) == 1
    # The ambient talk was cut where the wake phrase starts, before the detection point...
    assert amb[0]["end"] <= at and abs(amb[0]["end"] - phrase_start) < 0.6 * S
    # ...and everything after it, through the pause and the question, is one command.
    c = cmd[0]
    assert c["start"] == amb[0]["end"] and c["wake"] and c["wake_at"] == at
    assert c["end"] >= question_end - int(0.3 * S)
    assert c["end"] - at <= 8 * S  # the command window is capped


def test_A5_command_window_waits_1_5_s_for_the_command_and_caps_at_8_s():
    phrase = trim(load_wav(AUDIO / "wake.wav"))
    question = trim(load_wav(DATA / "asr_ref" / "0.wav"))[: 2 * S]
    # A 1.2 s pause after the wake word still belongs to the command.
    events, _ = detect(np.concatenate([silence(1), phrase, silence(1.2), question, silence(3)]))
    cmd = [s for s in kinds(events, "segment") if s["mode"] == "command"]
    assert len(cmd) == 1
    assert cmd[0]["end"] >= S + len(phrase) + int(1.2 * S) + int(1.7 * S)
    # A 2 s pause is too long: the wake word alone is a command, and it arms the next utterance.
    events, _ = detect(np.concatenate([silence(1), phrase, silence(2.0), question, silence(3)]))
    cmd = [s for s in kinds(events, "segment") if s["mode"] == "command"]
    assert len(cmd) == 2 and cmd[0]["end"] < S + len(phrase) + S
    # Talking on and on after it: the command is cut 8 s after the wake word.
    talk = load_wav(DATA / "asr_ref" / "1.wav")
    events, _ = detect(np.concatenate([silence(1), phrase, talk, silence(2)]))
    wake_at = kinds(events, "wake")[0]["at"]
    cmd = [s for s in kinds(events, "segment") if s["mode"] == "command"]
    assert cmd[0]["forced"] and cmd[0]["end"] - wake_at <= 8 * S + FRAME
    assert any(s["mode"] == "ambient" and s["start"] >= cmd[0]["end"] for s in kinds(events, "segment"))


def test_A5_no_trigger_on_ambient_clip():
    events, det = detect(np.concatenate([silence(1), load_wav(DATA / "ambient.wav"), silence(2)]))
    assert kinds(events, "wake") == []
    segs = kinds(events, "segment")
    assert segs and all(s["mode"] == "ambient" and not s["wake"] for s in segs)


@pytest.mark.parametrize("snr_db", [10, 5])
def test_A5_detects_wake_clip_over_recorded_background_noise(snr_db):
    """tests/audio/noise.wav is multi-talker babble mixed from real recordings (LibriSpeech clips)."""
    phrase, noise = load_wav(AUDIO / "wake.wav"), load_wav(AUDIO / "noise.wav")
    clean = np.concatenate([silence(1), phrase, silence(2)])
    # The noise runs through the silences too, as in a real room; SNR is measured against the phrase.
    p_speech = np.mean(phrase.astype(np.float64) ** 2)
    bed = np.resize(noise, clean.shape).astype(np.float64)
    bed *= np.sqrt(p_speech / (np.mean(bed**2) * 10 ** (snr_db / 10)))
    audio = np.clip(clean + bed, -32768, 32767).astype(np.int16)
    events, _ = detect(audio)
    assert len(kinds(events, "wake")) == 1


def test_E41_the_wake_word_scores_every_chunk_even_with_the_gate_shut():
    from kin.audio.wakeword import WakeWord

    class Counting(WakeWord):
        chunks = 0

        def score(self, chunk):
            Counting.chunks += 1
            assert chunk.shape[0] == CHUNK
            return super().score(chunk)

    audio = np.concatenate([silence(4), load_wav(DATA / "ambient.wav"), silence(4)])
    wake = Counting.from_spec()
    _, det = detect(audio, wake=wake)
    assert Counting.chunks == len(audio) // CHUNK  # all of them, silence included
    assert det.frames_vad < det.frames_total * 0.8  # while the gate still keeps VAD off in silence


def test_E41_blocks_of_any_size_give_the_same_events():
    audio = np.concatenate([silence(1), load_wav(DATA / "ambient.wav")[: 2 * S], load_wav(AUDIO / "wake.wav"),
                            silence(2)])  # fmt: skip
    a, _ = detect(audio, block=CHUNK)
    b, _ = detect(audio[: len(audio) // FRAME * FRAME], block=FRAME)
    strip = lambda ev: [{k: v for k, v in e.items() if k not in ("t", "score")} for e in ev]  # noqa: E731
    assert strip(a) == strip(b)


def test_E09_vad_end_of_speech_closes_the_command_window():
    phrase = trim(load_wav(AUDIO / "wake.wav"))
    events, _ = detect(np.concatenate([silence(1), phrase, silence(0.3), trim(load_wav(DATA / "asr_ref" / "0.wav"))[: 2 * S],
                                       silence(4)]))  # fmt: skip
    cmd = [s for s in kinds(events, "segment") if s["mode"] == "command"][0]
    end_of_speech = S + len(phrase) + int(0.3 * S) + 2 * S
    assert not cmd["forced"] and cmd["end"] - end_of_speech < S  # closed by VAD, not by the cap
