from kin.audio.endpoint import Endpointer, ms_to_samples
from kin.config import FRAME, AudioSettings


def run(probs, s=AudioSettings()):
    ep, segs = Endpointer(s), []
    for i, p in enumerate(probs):
        seg = ep.push(i * FRAME, p)
        if seg:
            segs.append(seg)
    tail = ep.flush(len(probs) * FRAME)
    return segs + ([tail] if tail else [])


def frames(ms):
    return round(ms_to_samples(ms) / FRAME)


def test_one_utterance_with_pre_roll():
    probs = [0.0] * 20 + [0.9] * frames(1000) + [0.0] * frames(1000)
    (seg,) = run(probs)
    speech_start = 20 * FRAME
    assert seg.start == speech_start - ms_to_samples(300)
    speech_end = speech_start + frames(1000) * FRAME
    assert speech_end <= seg.end <= speech_end + ms_to_samples(200)


def test_short_pause_does_not_split():
    probs = [0.9] * frames(500) + [0.1] * frames(300) + [0.9] * frames(500) + [0.0] * frames(1000)
    assert len(run(probs)) == 1


def test_long_pause_splits():
    probs = ([0.9] * frames(500) + [0.0] * frames(900)) * 2
    assert len(run(probs)) == 2


def test_hysteresis_keeps_speech_between_on_and_off():
    probs = [0.9] * frames(300) + [0.4] * frames(2000) + [0.0] * frames(1000)
    (seg,) = run(probs)
    assert seg.seconds > 2.2


def test_blips_are_ignored():
    assert run([0.0] * 10 + [0.9] + [0.0] * 50) == []  # one frame: never starts
    assert run([0.0] * 10 + [0.9] * 4 + [0.0] * 50) == []  # 128 ms: below min length


def test_max_length_forces_cut_without_losing_audio():
    s = AudioSettings(max_segment_ms=2000)
    probs = [0.9] * frames(5000) + [0.0] * frames(1000)
    segs = run(probs, s)
    assert len(segs) == 3
    assert segs[0].forced and segs[1].forced and not segs[2].forced
    for a, b in zip(segs, segs[1:], strict=False):
        assert a.end == b.start  # contiguous: no audio dropped at the cuts
