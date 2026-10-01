"""O3 voice reply: Piper text-to-speech, entirely local.

Runs inside kin-core (spec CL6), never in the gateway. A worker thread
synthesises and plays one reply at a time; a new reply replaces one that is
still playing. While a reply plays, `speaking_between(t0, t1)` tells the
pipeline that anything heard in that window is Kin's own voice (E40), so it is
tagged `kin` and never stored or answered.

The voice model loads on first use and is released after five idle minutes,
which matters on a 4 GB Raspberry Pi.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time

import numpy as np

from kin.config import load_model_spec, model_path

log = logging.getLogger("kin.voice")

IDLE_UNLOAD_S = 300.0
MAX_VOICE_CHARS = 420  # about 120 tokens
LEAD_S = 0.4  # segments that began this long before playback may already contain it
TAIL_S = 1.2  # room echo and the recogniser's end-of-speech delay


def spoken_form(text: str) -> str:
    """The text as it should be read aloud: no episode ids or brackets, at most ~120 tokens."""
    t = re.sub(r"#?\be-\d+\b|\bcal-\d+\b", "", text)
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = t.replace("°C", " degrees")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= MAX_VOICE_CHARS:
        return t
    cut = t[:MAX_VOICE_CHARS]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[: end + 1] if end > 80 else cut.rsplit(" ", 1)[0] + ".").strip()


class PiperEngine:
    """Piper on ONNX Runtime; the same model file runs on the laptop and on a Raspberry Pi."""

    def __init__(self, path=None):
        spec = load_model_spec()["voice"]
        self.path = path or model_path("voice", spec["file"])
        self._voice = None
        self._lock = threading.Lock()  # a release must never land in the middle of a synthesis

    def installed(self) -> bool:
        return self.path.exists()

    def load(self) -> None:
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(str(self.path))

    def unload(self) -> None:
        if self._lock.acquire(blocking=False):
            try:
                self._voice = None
            finally:
                self._lock.release()

    @property
    def loaded(self) -> bool:
        return self._voice is not None

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        with self._lock:
            self.load()
            voice = self._voice
            chunks = list(voice.synthesize(text))
            audio = np.concatenate([c.audio_int16_array for c in chunks]) if chunks else np.zeros(0, dtype=np.int16)
            return audio, voice.config.sample_rate


def _play_sounddevice(audio: np.ndarray, rate: int, stop: threading.Event) -> None:
    import sounddevice as sd

    sd.play(audio, rate)
    end = time.monotonic() + len(audio) / rate + 0.5
    while time.monotonic() < end and not stop.is_set():
        time.sleep(0.02)
    sd.stop()


class Voice:
    def __init__(self, engine=None, player=None, enabled: bool = True, on_state=None, clock=time.time):
        self.engine = engine if engine is not None else PiperEngine()
        self._player = player or _play_sounddevice
        self.enabled = enabled
        self.on_state = on_state  # called with "speaking" / "idle"
        self._clock = clock
        self._q: queue.Queue = queue.Queue()
        self._stop_play = threading.Event()
        self._window: tuple[float, float] | None = None  # (start, end) of the reply now playing or just played
        self._last_used = time.monotonic()
        self.last_synth_ms = 0.0
        self._thread = threading.Thread(target=self._loop, name="voice", daemon=True)
        self._thread.start()

    @property
    def available(self) -> bool:
        return bool(getattr(self.engine, "installed", lambda: True)())

    def speak(self, text: str) -> bool:
        """Queue a reply (replacing any waiting one and cutting off one still playing)."""
        text = spoken_form(text)
        if not (self.enabled and text and self.available):
            return False
        self._stop_play.set()
        while True:  # only the newest reply is worth saying
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._last_used = time.monotonic()
        # The window opens now, before synthesis, so nothing Kin is about to say is stored.
        self._window = (self._clock(), self._clock() + 60.0)
        self._q.put(text)
        return True

    def stop(self) -> None:
        self._stop_play.set()

    def speaking_between(self, t0: float, t1: float) -> bool:
        """Did Kin's own voice overlap [t0, t1] (unix seconds)?"""
        w = self._window
        return bool(w and t1 >= w[0] - LEAD_S and t0 <= w[1] + TAIL_S)

    def maybe_unload(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if getattr(self.engine, "loaded", False) and now - self._last_used > IDLE_UNLOAD_S:
            self.engine.unload()
            log.info("voice model released")

    def _loop(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            self._stop_play.clear()
            try:
                t0 = time.perf_counter()
                audio, rate = self.engine.synth(text)
                self.last_synth_ms = (time.perf_counter() - t0) * 1000
                self._last_used = time.monotonic()
                if self._stop_play.is_set() or not len(audio):
                    self._window = (self._window[0], self._clock()) if self._window else None
                    continue
                start = self._clock()
                self._window = (start, start + len(audio) / rate)
                if self.on_state:
                    self.on_state("speaking")
                self._player(audio, rate, self._stop_play)
                self._window = (start, self._clock())
            except Exception:  # noqa: BLE001 - a broken speaker must not stop answers appearing on screen
                log.exception("could not speak")
                self._window = None
            finally:
                if self.on_state:
                    self.on_state("idle")

    def close(self) -> None:
        self._stop_play.set()
        self._q.put(None)
