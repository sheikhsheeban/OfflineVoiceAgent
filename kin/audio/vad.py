"""Silero VAD v5 on ONNX Runtime, one 512-sample frame (32 ms) at a time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from kin.config import FRAME, SAMPLE_RATE

_CONTEXT = 64  # v5 expects the last 64 samples of the previous frame prepended


class SileroVAD:
    def __init__(self, model_path: str | Path, threads: int = 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one int16 frame of exactly FRAME samples."""
        if frame.shape[0] != FRAME:
            raise ValueError(f"VAD frame must be {FRAME} samples, got {frame.shape[0]}")
        x = (frame.astype(np.float32) * (1.0 / 32768.0))[None, :]
        inp = np.concatenate([self._context, x], axis=1)
        prob, self._state = self._sess.run(None, {"input": inp, "state": self._state, "sr": self._sr})
        self._context = inp[:, -_CONTEXT:]
        return float(prob[0, 0])
