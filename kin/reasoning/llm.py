"""The local language model: Gemma 3 1B on llama.cpp.

Loaded on first use with `use_mmap=True` (weights stay in the page cache, not
private memory) and released after `idle_unload_s` of disuse, the same policy
as the phone. Before loading it checks free memory; below 1.5 GB it refuses,
and Kin answers in retrieval-only mode instead (sheet 8). On battery below 30 %
it uses half the threads.

Grammar-constrained decoding in llama.cpp scores every vocabulary entry per
token, which with Gemma's 262k vocabulary runs at ~8 tokens/s on a 2019 laptop
CPU. Tool calls therefore pre-fill the fixed JSON scaffolding in the prompt and
let the grammar constrain only the few variable tokens; free-text answers are
generated without a grammar and checked by the confidence gate.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from kin.config import MODELS_DIR, SPEC_DIR, load_model_spec

log = logging.getLogger("kin.llm")
MIN_AVAILABLE_MB = 1500


class LLMUnavailable(RuntimeError):
    """The model cannot be used right now (missing, or not enough free memory)."""


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    gen_tokens: int
    ms: float


def load_prompt(name: str) -> str:
    return (SPEC_DIR / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def load_grammar(name: str) -> str:
    return (SPEC_DIR / "grammars" / f"{name}.gbnf").read_text(encoding="utf-8")


def _clean(text: str) -> str:
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("�", "'")):
        text = text.replace(a, b)
    return text.strip()


def battery_low() -> bool:
    try:
        b = psutil.sensors_battery()
    except Exception:  # noqa: BLE001 - no battery sensor
        return False
    return bool(b and not b.power_plugged and b.percent < 30)


class LocalLLM:
    def __init__(self, threads: int, path: Path | None = None, idle_unload_s: float | None = None):
        spec = load_model_spec()["llm"]
        self.path = Path(path or MODELS_DIR / spec["dir"] / spec["file"])
        self.n_ctx = int(spec["n_ctx"])
        self.max_answer_tokens = int(spec["max_answer_tokens"])
        self.idle_unload_s = float(spec["idle_unload_s"] if idle_unload_s is None else idle_unload_s)
        self.threads = threads
        self._llm = None
        self._lock = threading.RLock()
        self.last_used = 0.0
        self.load_ms = 0.0

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def installed(self) -> bool:
        return self.path.exists()

    def load(self) -> None:
        with self._lock:
            if self._llm is not None:
                return
            if not self.installed():
                raise LLMUnavailable(f"model file missing: {self.path} (run `uv run kin fetch-models`)")
            avail = psutil.virtual_memory().available / 2**20
            if avail < MIN_AVAILABLE_MB:
                raise LLMUnavailable(f"only {avail:.0f} MB free; answering from memory without the model")
            from llama_cpp import Llama

            threads = max(1, self.threads // 2) if battery_low() else self.threads
            t0 = time.perf_counter()
            self._llm = Llama(
                model_path=str(self.path),
                n_ctx=self.n_ctx,
                n_threads=threads,
                n_threads_batch=threads,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            self.load_ms = (time.perf_counter() - t0) * 1000
            self.last_used = time.monotonic()
            log.info("language model loaded in %.0f ms (%d threads)", self.load_ms, threads)

    def unload(self) -> None:
        with self._lock:
            if self._llm is not None:
                self._llm.close()
                self._llm = None
                log.info("language model released")

    def maybe_unload(self, now: float | None = None) -> bool:
        """Release the model after `idle_unload_s` without use. Returns True if it did."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._llm is not None and now - self.last_used >= self.idle_unload_s:
                self.unload()
                return True
        return False

    def complete(
        self,
        user: str,
        *,
        prefill: str = "",
        grammar: str | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Completion:
        """One Gemma turn. `prefill` starts the model's reply (fixed JSON scaffolding);
        `grammar` (GBNF) constrains what follows it."""
        from llama_cpp import LlamaGrammar

        prompt = f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{prefill}"
        with self._lock:
            self.load()
            t0 = time.perf_counter()
            out = self._llm(
                prompt,
                max_tokens=max_tokens or self.max_answer_tokens,
                temperature=0.0,
                stop=(stop or []) + ["<end_of_turn>"],
                grammar=LlamaGrammar.from_string(grammar, verbose=False) if grammar else None,
            )
            ms = (time.perf_counter() - t0) * 1000
            self.last_used = time.monotonic()
        u = out["usage"]
        return Completion(_clean(out["choices"][0]["text"]), u["prompt_tokens"], u["completion_tokens"], ms)

    def warm(self, prefix: str) -> None:
        """Load and evaluate a fixed prompt prefix, so the first real answer skips both."""
        self.complete(prefix, max_tokens=1)
