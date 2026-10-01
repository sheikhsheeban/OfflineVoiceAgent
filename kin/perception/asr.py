"""Speech to text on sherpa-onnx (C++), offline decoding per segment.

The model is chosen by name from `[asr_models]` in spec/models.toml; each
profile has a default there (Moonshine v2 Base for phone, Parakeet 0.6B for
laptop), and `--asr` / KIN_ASR override it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from kin.config import MODELS_DIR, SAMPLE_RATE, load_model_spec
from kin.native import preload_onnxruntime

# Longest input each model kind decodes reliably. The Moonshine v2 export fails
# (sherpa-onnx then returns an empty string) beyond ~9.5 s.
MAX_CHUNK_S = {"moonshine_v2": 8.0, "whisper": 28.0}
_WHISPER_TAGS = re.compile(r"\[[A-Z_ ]+\]?|\((?:music|applause|laughter|silence|inaudible)[^)]*\)", re.IGNORECASE)


def split_at_pauses(x: np.ndarray, max_len: int, search: int = 2 * SAMPLE_RATE) -> list[tuple[int, int]]:
    """Cut `x` into pieces of at most `max_len` samples, each cut placed at the
    quietest 100 ms in the last `search` samples before the limit (between words)."""
    if len(x) <= max_len:
        return [(0, len(x))]
    energy = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    win, hop = SAMPLE_RATE // 10, SAMPLE_RATE // 50
    out, start = [], 0
    while len(x) - start > max_len:
        lo, hi = start + max(max_len - search, win), start + max_len
        best, cut = None, hi
        for c in range(lo, hi - win + 1, hop):
            e = energy[c + win] - energy[c]
            if best is None or e < best:
                best, cut = e, c + win // 2
        out.append((start, cut))
        start = cut
    out.append((start, len(x)))
    return out


@dataclass
class AsrResult:
    text: str
    audio_s: float
    decode_ms: float

    @property
    def rtf(self) -> float:
        return (self.decode_ms / 1000) / self.audio_s if self.audio_s else 0.0


class Transcriber:
    def __init__(self, threads: int = 2, model: str | None = None):
        preload_onnxruntime()
        import sherpa_onnx

        from kin.config import asr_model_name

        self.model = model or asr_model_name("phone")
        spec = load_model_spec()["asr_models"][self.model]
        d = MODELS_DIR / spec["dir"]
        kind = self.kind = spec["kind"]
        self.max_chunk = int(MAX_CHUNK_S[kind] * SAMPLE_RATE) if kind in MAX_CHUNK_S else None
        R = sherpa_onnx.OfflineRecognizer
        if kind == "moonshine":
            self._rec = R.from_moonshine(
                preprocessor=str(d / "preprocess.onnx"),
                encoder=str(d / "encode.int8.onnx"),
                uncached_decoder=str(d / "uncached_decode.int8.onnx"),
                cached_decoder=str(d / "cached_decode.int8.onnx"),
                tokens=str(d / "tokens.txt"),
                num_threads=threads,
            )
        elif kind == "moonshine_v2":
            self._rec = R.from_moonshine_v2(
                encoder=str(d / spec["encoder"]),
                decoder=str(d / spec["decoder"]),
                tokens=str(d / "tokens.txt"),
                num_threads=threads,
            )
        elif kind == "nemo_transducer":
            self._rec = R.from_transducer(
                encoder=str(d / spec["encoder"]),
                decoder=str(d / spec["decoder"]),
                joiner=str(d / spec["joiner"]),
                tokens=str(d / "tokens.txt"),
                num_threads=threads,
                model_type="nemo_transducer",
            )
        elif kind == "whisper":
            self._rec = R.from_whisper(
                encoder=str(d / spec["encoder"]),
                decoder=str(d / spec["decoder"]),
                tokens=str(d / spec["tokens"]),
                num_threads=threads,
            )
        else:
            raise ValueError(f"unsupported ASR kind {kind!r}")

    def transcribe(self, audio: np.ndarray) -> AsrResult:
        """Decode int16 or float32 mono 16 kHz audio. The GIL is released inside sherpa-onnx."""
        if audio.dtype == np.int16:
            x = audio.astype(np.float32) * (1.0 / 32768.0)
        else:
            x = audio.astype(np.float32, copy=False)
        t0 = time.perf_counter()
        pieces = split_at_pauses(x, self.max_chunk) if self.max_chunk else [(0, len(x))]
        text = " ".join(t for t in (self._decode(x[a:b]) for a, b in pieces) if t)
        if self.kind == "whisper":
            text = " ".join(_WHISPER_TAGS.sub(" ", text).split())
        text = collapse_repeats(text)
        return AsrResult(text, len(x) / SAMPLE_RATE, (time.perf_counter() - t0) * 1000)

    def _decode(self, x: np.ndarray) -> str:
        stream = self._rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, x)
        self._rec.decode_stream(stream)
        return stream.result.text.strip()

    def warm_up(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))


def collapse_repeats(text: str, max_repeats: int = 3) -> str:
    """Cut decoder loops ("I am from I am from I am from ..."), which small
    autoregressive models such as Whisper produce on near-silent audio."""
    words = text.split()
    for n in range(1, 7):
        i = 0
        while i + n * (max_repeats + 1) <= len(words):
            gram = [w.lower().strip(",.") for w in words[i : i + n]]
            j = i + n
            while j + n <= len(words) and [w.lower().strip(",.") for w in words[j : j + n]] == gram:
                j += n
            if (j - i) // n > max_repeats:
                words = words[: i + n] + words[j:]
            i += 1
    return " ".join(words)


def wake_phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Match the wake phrase (loosely) at the start of a transcript."""
    words = [re.escape(w) for w in phrase.lower().split()]
    # ASR often drops or mishears the first word ("a jarvis", "jarvis").
    head = "".join(f"(?:{w}\\W*)?" for w in words[:-1])
    return re.compile(rf"^\W*(?:a\W+)?{head}{words[-1]}\W*", re.IGNORECASE)


def wake_text_pattern(phrase: str) -> re.Pattern[str]:
    """The wake phrase at the start of a transcript: "hey jarvis", "a jarvis" (as ASR often hears it) or
    just "jarvis". The safety net when the acoustic wake word missed."""
    words = [re.escape(w) for w in phrase.lower().split()]
    lead = r"\W+".join(words[:-1])
    return re.compile(rf"^\W*(?:(?:{lead}|a)\W+)?{words[-1]}\b", re.IGNORECASE)


def wake_anywhere_pattern(phrase: str) -> re.Pattern[str]:
    """The wake phrase anywhere, up to and including the punctuation after it."""
    words = [re.escape(w) for w in phrase.lower().split()]
    lead = r"\W+".join(words[:-1])
    return re.compile(rf"\b(?:(?:{lead}|a)\W+)?{words[-1]}\b\W*", re.IGNORECASE)


def strip_wake_phrase(text: str, pattern: re.Pattern[str], anywhere: re.Pattern[str] | None = None,
                      within: int = 40) -> str:  # fmt: skip
    """The command without its wake phrase. With `anywhere`, also drops a few words heard before the
    phrase (the end of an ambient sentence that fell inside the command window)."""
    m = pattern.match(text)
    if m is None and anywhere is not None:
        m = anywhere.search(text)
        if m is not None and m.start() > within:
            m = None
    rest = (text[m.end() :] if m else text).strip()
    return rest[:1].upper() + rest[1:] if rest else ""
