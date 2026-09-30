"""Sentence embeddings: all-MiniLM-L6-v2 int8 on ONNX Runtime + Rust tokenizers.

This is what `fastembed` does for this model (tokenize, run, mean-pool,
normalise), without its extra dependencies in kin-core. Vectors are stored
as int8 (x127), which keeps cosine ranking intact at a quarter of the size.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from kin.config import MODELS_DIR, load_model_spec


def quantize(v: np.ndarray) -> np.ndarray:
    """Unit float vector(s) -> int8, the stored form."""
    return np.clip(np.rint(v * 127.0), -127, 127).astype(np.int8)


class Embedder:
    def __init__(self, threads: int = 1):
        from tokenizers import Tokenizer

        spec = load_model_spec()["embed"]
        d = MODELS_DIR / spec["dir"]
        self.dim = int(spec["dim"])
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(int(spec["max_tokens"]))
        self._tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        model = next(n for n in spec["files"] if n.endswith(".onnx"))
        self._sess = ort.InferenceSession(str(d / model), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length float32 vectors, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        enc = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._sess.run(None, feeds)[0]
        pooled = (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
        return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
