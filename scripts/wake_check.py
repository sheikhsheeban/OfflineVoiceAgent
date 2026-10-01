"""Feed a WAV straight to openWakeWord, 1280 int16 samples (80 ms) at a time, and print the scores.

Usage:  uv run python scripts/wake_check.py [path.wav] [--gain 0.5] [--noise noise.wav --snr 10]

It runs the reference `openwakeword` package and Kin's own reimplementation (kin.audio.wakeword) on
the same chunks, so a disagreement between the two shows up here and not in a live session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kin.audio.capture import load_wav  # noqa: E402
from kin.audio.wakeword import CHUNK, WakeWord  # noqa: E402
from kin.config import MODELS_DIR, SAMPLE_RATE, load_model_spec  # noqa: E402


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.resize(noise, clean.shape).astype(np.float64)
    c = clean.astype(np.float64)
    p_c, p_n = np.mean(c**2), max(np.mean(noise**2), 1e-9)
    scaled = noise * np.sqrt(p_c / (p_n * 10 ** (snr_db / 10)))
    return np.clip(c + scaled, -32768, 32767).astype(np.int16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", default=str(ROOT / "tests/audio/wake.wav"))
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--noise", help="a WAV to mix in")
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    spec = load_model_spec()["wake_word"]
    d = MODELS_DIR / spec["dir"]
    audio = load_wav(a.wav)
    if a.gain != 1.0:
        audio = np.clip(audio.astype(np.float64) * a.gain, -32768, 32767).astype(np.int16)
    if a.noise:
        audio = mix(audio, load_wav(a.noise), a.snr)
    # One second of silence either side, as a live stream would have.
    audio = np.concatenate([np.zeros(SAMPLE_RATE, np.int16), audio, np.zeros(SAMPLE_RATE, np.int16)])
    chunks = [audio[i : i + CHUNK] for i in range(0, len(audio) - CHUNK + 1, CHUNK)]

    print(f"file          {a.wav}  ({len(audio) / SAMPLE_RATE:.2f} s with padding, gain {a.gain})")
    print(
        f"sample rate   {SAMPLE_RATE} Hz;  chunk {CHUNK} samples ({CHUNK * 1000 // SAMPLE_RATE} ms);  "
        f"dtype fed {chunks[0].dtype}"
    )

    from openwakeword.model import Model

    oww = Model(
        wakeword_models=[str(d / spec["keyword"])],
        inference_framework="onnx",
        melspec_model_path=str(d / "melspectrogram.onnx"),
        embedding_model_path=str(d / "embedding_model.onnx"),
    )
    ref = []
    for c in chunks:
        pred = oww.predict(c)
        ref.append(pred)
    keys = list(ref[0].keys())
    key = keys[0]
    ref_scores = np.array([p[key] for p in ref])

    kin = WakeWord.from_spec(a.threshold)
    kin_scores = np.array([kin.feed(c) or 0.0 for c in chunks])

    above = int((ref_scores >= a.threshold).sum())
    print(f"model key     {key!r}  (all keys: {keys})")
    print(
        f"openWakeWord  max {ref_scores.max():.3f} at chunk {ref_scores.argmax()};  {above} chunk(s) >= {a.threshold}"
    )
    print(
        f"kin WakeWord  max {kin_scores.max():.3f} at chunk {kin_scores.argmax()};  "
        f"{int((kin_scores >= a.threshold).sum())} chunk(s) >= {a.threshold}"
    )
    print(f"max |diff|    {np.abs(ref_scores - kin_scores).max():.4f}")
    return 0 if ref_scores.max() >= a.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
