"""Audio sources that write into the shared ring.

`MicSource` wraps a PortAudio input stream; its callback does one slice copy
into shared memory and nothing else. `WavSource` replays a demo file through
the same ring at real-time pace (or faster, for tests and benchmarks).
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

import numpy as np

from kin.audio.ring import Ring
from kin.config import FRAME, SAMPLE_RATE


def find_input_device(want: str | int | None) -> int | None:
    import sounddevice as sd

    if want is None or want == "":
        return None
    if isinstance(want, int) or str(want).isdigit():
        return int(want)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and str(want).lower() in d["name"].lower():
            return i
    raise LookupError(f"no input device matching {want!r}; run `kin devices`")


class MicSource:
    def __init__(self, ring: Ring, device: str | int | None = None):
        self.ring = ring
        self.device = find_input_device(device)
        self.overflows = 0
        self.finished = threading.Event()  # set if the device disappears
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:  # PortAudio thread
        if status.input_overflow:
            self.overflows += 1
        self.ring.write(indata[:, 0])

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._callback,
            finished_callback=self.finished.set,
        )
        self._stream.start()

    @property
    def device_name(self) -> str:
        import sounddevice as sd

        idx = self.device if self.device is not None else sd.default.device[0]
        return sd.query_devices(idx)["name"]

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def load_wav(path: str | Path) -> np.ndarray:
    """Load a WAV as 16 kHz mono int16 (demo input only; never written by Kin)."""
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: need 16-bit PCM, got {8 * width}-bit")
    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        n = int(len(x) * SAMPLE_RATE / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)
    return x


class WavSource:
    def __init__(self, ring: Ring, path: str | Path, speed: float = 1.0, lead_s: float = 1.0, tail_s: float = 1.5):
        self.ring = ring
        self.speed = speed
        pad = lambda s: np.zeros(int(s * SAMPLE_RATE), dtype=np.int16)  # noqa: E731
        self.audio = np.concatenate([pad(lead_s), load_wav(path), pad(tail_s)])
        self.device_name = f"replay:{Path(path).name}"
        self.overflows = 0
        self.finished = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="wav-source", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        period = FRAME / SAMPLE_RATE / self.speed if self.speed > 0 else 0.0
        t_next = time.perf_counter()
        for i in range(0, len(self.audio) - FRAME + 1, FRAME):
            if self._stop.is_set():
                break
            self.ring.write(self.audio[i : i + FRAME])
            if period:
                t_next += period
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        self.finished.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
