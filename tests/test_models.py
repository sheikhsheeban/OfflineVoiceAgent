"""VAD, wake word and ASR on real speech."""

import numpy as np
import pytest

from kin.audio.capture import load_wav
from kin.config import FRAME, model_path
from tests.conftest import ASR_WAVS, DATA, wer

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def vad():
    from kin.audio.vad import SileroVAD

    return SileroVAD(model_path("vad", "silero_vad.onnx"))


@pytest.fixture(scope="module")
def asr():
    from kin.perception.asr import Transcriber

    return Transcriber(threads=2)


def vad_probs(vad, audio):
    vad.reset()
    return np.array([vad(audio[i : i + FRAME]) for i in range(0, len(audio) - FRAME + 1, FRAME)])


def test_vad_silence_and_noise_are_not_speech(vad):
    rng = np.random.default_rng(1)
    assert vad_probs(vad, np.zeros(16000 * 2, np.int16)).max() < 0.1
    hiss = (rng.standard_normal(16000 * 2) * 300).astype(np.int16)
    assert vad_probs(vad, hiss).mean() < 0.1


def test_vad_detects_speech(vad):
    p = vad_probs(vad, load_wav(ASR_WAVS / "0.wav"))
    assert (p > 0.5).mean() > 0.6


def test_vad_rejects_wrong_frame_size(vad):
    with pytest.raises(ValueError):
        vad(np.zeros(480, np.int16))


def wake_scores(audio, pad=True):
    from kin.audio.wakeword import CHUNK, WakeWord

    ww = WakeWord.from_spec()
    if pad:
        audio = np.concatenate([np.zeros(16000, np.int16), audio, np.zeros(16000, np.int16)])
    return np.array([ww.feed(audio[i : i + CHUNK]) for i in range(0, len(audio) - CHUNK + 1, CHUNK)])


def test_wake_word_fires_on_phrase():
    assert wake_scores(load_wav(DATA / "hey_jarvis.wav")).max() > 0.9


@pytest.mark.parametrize("wav", [DATA / "ambient.wav", ASR_WAVS / "0.wav", ASR_WAVS / "1.wav"])
def test_wake_word_ignores_other_speech(wav):
    assert wake_scores(load_wav(wav)).max() < 0.1


def test_wake_word_matches_openwakeword_reference():
    oww = pytest.importorskip("openwakeword.model")
    from kin.config import MODELS_DIR

    d = MODELS_DIR / "wake_word"
    ref = oww.Model(
        wakeword_models=[str(d / "hey_jarvis_v0.1.onnx")],
        inference_framework="onnx",
        melspec_model_path=str(d / "melspectrogram.onnx"),
        embedding_model_path=str(d / "embedding_model.onnx"),
    )
    audio = np.concatenate([np.zeros(16000, np.int16), load_wav(DATA / "command.wav")])
    ours = wake_scores(audio, pad=False)
    theirs = np.array([ref.predict(audio[i : i + 1280])["hey_jarvis_v0.1"] for i in range(0, len(audio) - 1279, 1280)])
    assert ours.max() > 0.9 and ours.argmax() == theirs.argmax()
    assert np.abs(ours[16:] - theirs[16:]).max() < 0.01


def test_wake_prime_refreshes_context():
    from kin.audio.wakeword import WakeWord

    ww = WakeWord.from_spec()
    ww.feed(load_wav(DATA / "ambient.wav"))  # stale context from earlier speech
    phrase = load_wav(DATA / "hey_jarvis.wav")
    ww.prime(np.zeros(16000, np.int16))
    scores = [ww.feed(phrase[i : i + 1280]) for i in range(0, len(phrase) - 1279, 1280)]
    scores += [ww.feed(np.zeros(1280, np.int16)) for _ in range(8)]
    assert max(s for s in scores if s is not None) > 0.9


@pytest.mark.parametrize("idx", [0, 1])
def test_asr_accuracy_on_reference_speech(asr, idx):
    lines = (ASR_WAVS / "trans.txt").read_text().splitlines()
    refs = dict(line.split(" ", 1) for line in lines if line.strip())
    res = asr.transcribe(load_wav(ASR_WAVS / f"{idx}.wav"))
    assert wer(refs[f"{idx}.wav"], res.text) < 0.15, res.text
    assert res.rtf < 0.5  # comfortably faster than real time


def test_asr_on_command_fixture(asr):
    res = asr.transcribe(load_wav(DATA / "command.wav"))
    assert wer("hey jarvis remind me to call rohan about the site visit on thursday", res.text) < 0.2, res.text


def test_asr_silence_gives_no_words(asr):
    assert asr.transcribe(np.zeros(16000, np.int16)).text.strip(" .") == ""


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hey, Jarvis. Remind me to call Rohan.", "Remind me to call Rohan."),
        ("Hey Jarvis what's on today", "What's on today"),
        ("Jarvis, set a timer", "Set a timer"),
        ("A Jarvis. Hello", "Hello"),
        ("Hey, Jarvis.", ""),
        ("The budget moved to Friday", "The budget moved to Friday"),
    ],
)
def test_strip_wake_phrase(text, expected):
    from kin.perception.asr import strip_wake_phrase, wake_phrase_pattern

    assert strip_wake_phrase(text, wake_phrase_pattern("hey jarvis")) == expected
