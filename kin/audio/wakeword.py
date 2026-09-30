"""Wake word detection: openWakeWord models run directly on ONNX Runtime.

Three small models per 80 ms chunk (1280 samples): a mel spectrogram, a
speech-embedding model over the last 76 mel frames, and the keyword classifier
over the last 16 embeddings (~1.28 s). This is the same streaming computation
as the `openwakeword` package, without importing it: its package pulls in
scipy and scikit-learn, which cost ~85 MB of RAM in kin-audio and are never
used at inference time.

Kin only feeds audio while the energy gate is open. When the gate reopens after
silence, `prime()` pushes the ring's pre-roll through first, which replaces the
whole 16-embedding context.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from kin.config import MODELS_DIR, load_model_spec

CHUNK = 1280
MEL_WINDOW = 76  # mel frames per embedding
N_EMBED = 16  # embeddings per classifier call
CONTEXT_SAMPLES = N_EMBED * CHUNK
_MEL_OVERLAP = 480  # extra samples so each chunk yields 8 fresh mel frames


def _session(path: str | Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class WakeWord:
    def __init__(
        self,
        keyword_path: str | Path,
        melspec_path: str | Path,
        embedding_path: str | Path,
        threshold: float = 0.5,
    ):
        self._mel = _session(melspec_path)
        self._emb = _session(embedding_path)
        self._clf = _session(keyword_path)
        self._clf_in = self._clf.get_inputs()[0].name
        self.threshold = threshold
        self.last_score = 0.0
        self.reset()

    @classmethod
    def from_spec(cls, threshold: float = 0.5) -> "WakeWord":
        spec = load_model_spec()["wake_word"]
        d = MODELS_DIR / spec["dir"]
        return cls(d / spec["keyword"], d / "melspectrogram.onnx", d / "embedding_model.onnx", threshold)

    def reset(self) -> None:
        self._raw = np.zeros(CHUNK + _MEL_OVERLAP, dtype=np.float32)
        self._mels = np.ones((MEL_WINDOW, 32), dtype=np.float32)
        self._embs = np.zeros((N_EMBED, 96), dtype=np.float32)
        self._fresh = 0  # embeddings computed from real audio since reset
        self._pending = np.empty(0, dtype=np.int16)

    def _chunk(self, chunk: np.ndarray) -> float:
        self._raw = np.concatenate([self._raw[CHUNK:], chunk.astype(np.float32)])
        mel = self._mel.run(None, {"input": self._raw[None, :]})[0].squeeze() / 10.0 + 2.0
        self._mels = np.vstack([self._mels, mel])[-MEL_WINDOW:]
        emb = self._emb.run(None, {"input_1": self._mels[None, :, :, None]})[0].reshape(1, 96)
        self._embs = np.vstack([self._embs[1:], emb])
        self._fresh += 1
        if self._fresh < N_EMBED:
            return 0.0  # context not yet filled with real audio
        return float(self._clf.run(None, {self._clf_in: self._embs[None, :, :]})[0][0, 0])

    def prime(self, audio: np.ndarray) -> None:
        """Replace stale context with recent audio; scores are discarded."""
        self.reset()
        audio = audio[-CONTEXT_SAMPLES:]
        for i in range(audio.shape[0] % CHUNK, audio.shape[0], CHUNK):
            self._chunk(audio[i : i + CHUNK])

    def feed(self, frame: np.ndarray) -> float | None:
        """Feed int16 audio; returns the max score if at least one chunk ran."""
        self._pending = np.concatenate([self._pending, frame])
        best = None
        while self._pending.shape[0] >= CHUNK:
            chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            score = self._chunk(chunk)
            best = score if best is None else max(best, score)
        if best is not None:
            self.last_score = best
        return best
