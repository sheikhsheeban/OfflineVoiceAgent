"""Speech-model selection, every installed model, and an accuracy floor for the default."""

import glob
from pathlib import Path

import numpy as np
import pytest

from kin.audio.capture import load_wav
from kin.cli import installed_asr_models, verify_models
from kin.config import asr_model_name, load_model_spec
from kin.perception.asr import collapse_repeats
from tests.conftest import ASR_WAVS, DATA, wer

EVAL = DATA / "asr_eval"


def test_model_choice_precedence(monkeypatch):
    spec = load_model_spec()["asr"]
    monkeypatch.delenv("KIN_ASR", raising=False)
    assert asr_model_name("phone") == spec["phone"] == "moonshine-base"
    assert asr_model_name("laptop") == spec["laptop"]
    monkeypatch.setenv("KIN_ASR", "whisper-small")
    assert asr_model_name("phone") == "whisper-small"
    assert asr_model_name("phone", "moonshine-tiny") == "moonshine-tiny"  # --asr wins
    with pytest.raises(ValueError, match="unknown ASR model"):
        asr_model_name("phone", "gpt-9")


def test_every_listed_model_is_fully_pinned():
    for name, m in load_model_spec()["asr_models"].items():
        assert m["kind"] in {"moonshine", "moonshine_v2", "nemo_transducer", "whisper"}, name
        assert m["files"] and all(len(h) == 64 for h in m["files"].values()), name
        assert m["url"].startswith("https://github.com/k2-fsa/sherpa-onnx/releases/"), name


def test_verify_only_checks_requested_speech_models():
    installed = installed_asr_models()
    assert "moonshine-base" in installed
    assert verify_models() == []  # phone default only
    missing = [n for n in load_model_spec()["asr_models"] if n not in installed]
    if missing:
        assert any(missing[0] in p for p in verify_models([missing[0]]))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I am from I am from I am from I am from", "I am from"),
        ("the the the the end", "the end"),
        ("no no no", "no no no"),  # natural repetition is kept
        ("Remind me to call Rohan.", "Remind me to call Rohan."),
        ("", ""),
    ],
)
def test_collapse_repeats(text, expected):
    assert collapse_repeats(text) == expected


@pytest.mark.models
@pytest.mark.parametrize("model", load_model_spec()["asr_models"].keys())
def test_each_installed_model_transcribes(model):
    if model not in installed_asr_models():
        pytest.skip(f"{model} not installed (uv run kin fetch-models --asr {model})")
    from kin.perception.asr import Transcriber

    asr = Transcriber(threads=2, model=model)
    assert asr.model == model
    refs = dict(line.split(" ", 1) for line in (ASR_WAVS / "trans.txt").read_text().splitlines() if line.strip())
    res = asr.transcribe(load_wav(ASR_WAVS / "0.wav"))
    assert wer(refs["0.wav"], res.text) < 0.2, res.text
    assert asr.transcribe(np.zeros(16000, np.int16)).text.strip(" .") == ""


@pytest.mark.models
def test_default_model_accuracy_on_indian_names():
    """Regression floor for the phone default on the names eval set (clean speech).

    Moonshine Tiny scored 21.5% WER and 22/68 names here; v2 Base 14.1% and 35/68.
    """
    from kin.perception.asr import Transcriber

    asr = Transcriber(threads=2)
    refs = [line.strip() for line in (EVAL / "sentences.txt").read_text().splitlines() if line.strip()]
    errs = []
    for w in sorted(glob.glob(str(EVAL / "*.wav"))):
        idx = int(Path(w).stem.split("-")[1])
        audio = np.concatenate([np.zeros(4000, np.int16), load_wav(w), np.zeros(4000, np.int16)])
        errs.append(wer(refs[idx], asr.transcribe(audio).text))
    assert len(errs) == 24
    assert np.mean(errs) < 0.18, f"mean WER {np.mean(errs):.1%}"


def test_split_at_pauses_cuts_in_the_quiet_part():
    from kin.perception.asr import split_at_pauses

    sr = 16000
    loud = np.ones(sr * 7, np.float32) * 0.3
    x = np.concatenate([loud, np.zeros(sr // 2, np.float32), loud, np.zeros(sr // 2, np.float32), loud])
    pieces = split_at_pauses(x, 8 * sr)
    assert pieces[0][0] == 0 and pieces[-1][1] == len(x)
    assert all(b - a <= 8 * sr for a, b in pieces)
    assert all(a2 == b1 for (_, b1), (a2, _) in zip(pieces, pieces[1:], strict=False))  # nothing dropped
    first_cut = pieces[0][1]
    assert 7 * sr <= first_cut <= 7 * sr + sr // 2  # inside the first pause
    assert split_at_pauses(x[: 5 * sr], 8 * sr) == [(0, 5 * sr)]


@pytest.mark.models
@pytest.mark.parametrize("model", load_model_spec()["asr_models"].keys())
def test_long_utterances_are_not_lost(model):
    """Moonshine v2 returns nothing past ~9.5 s unless Kin splits the audio."""
    if model not in installed_asr_models():
        pytest.skip(f"{model} not installed")
    from kin.perception.asr import Transcriber

    refs = dict(line.split(" ", 1) for line in (ASR_WAVS / "trans.txt").read_text().splitlines() if line.strip())
    audio = load_wav(ASR_WAVS / "1.wav")  # 16.7 s
    assert len(audio) > 16 * 16000
    res = Transcriber(threads=2, model=model).transcribe(audio)
    assert wer(refs["1.wav"], res.text) < 0.25, res.text
