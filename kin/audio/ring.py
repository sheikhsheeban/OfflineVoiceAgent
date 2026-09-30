"""Single-writer audio ring buffer in shared memory.

kin-audio's PortAudio callback is the only writer. Readers (the kin-audio
detector thread and kin-core's ASR) address audio by absolute sample index,
so a speech event is just a pair of integers, never a copy of audio.

Layout: a 64-byte header (uint64 write position, uint64 capacity) followed by
`capacity` int16 samples. The write position counts every sample ever written;
sample `i` lives at slot `i % capacity` until it is overwritten.
"""

from __future__ import annotations

import os
from multiprocessing import shared_memory

import numpy as np

_HEADER = 64


class Overrun(Exception):
    """The requested samples were overwritten before they could be read."""


class Ring:
    def __init__(self, shm: shared_memory.SharedMemory, owner: bool):
        self._shm = shm
        self._owner = owner
        self._hdr = np.ndarray((2,), dtype=np.uint64, buffer=shm.buf[:16])
        self.capacity = int(self._hdr[1])
        self._data = np.ndarray((self.capacity,), dtype=np.int16, buffer=shm.buf[_HEADER:])

    @classmethod
    def create(cls, capacity: int, name: str | None = None) -> "Ring":
        shm = shared_memory.SharedMemory(name=name, create=True, size=_HEADER + capacity * 2)
        hdr = np.ndarray((2,), dtype=np.uint64, buffer=shm.buf[:16])
        hdr[0] = 0
        hdr[1] = capacity
        del hdr
        ring = cls(shm, owner=True)
        ring._data[:] = 0
        return ring

    @classmethod
    def attach(cls, name: str) -> "Ring":
        shm = shared_memory.SharedMemory(name=name, create=False)
        if os.name == "posix":
            # Python 3.11 registers attachments with the resource tracker, which
            # would unlink the segment when this process exits. Only the
            # supervisor (owner) may unlink it.
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        return cls(shm, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def write_pos(self) -> int:
        return int(self._hdr[0])

    def write(self, samples: np.ndarray) -> None:
        """Append samples. Called from the audio callback: no allocation beyond slicing."""
        n = samples.shape[0]
        pos = int(self._hdr[0])
        i = pos % self.capacity
        first = min(n, self.capacity - i)
        self._data[i : i + first] = samples[:first]
        if first < n:
            self._data[: n - first] = samples[first:]
        self._hdr[0] = pos + n  # publish only after the samples are in place

    def read(self, start: int, end: int) -> np.ndarray:
        """Copy samples [start, end). Raises Overrun if any were overwritten."""
        if end < start:
            raise ValueError("end < start")
        if end > self.write_pos:
            raise ValueError(f"samples up to {end} not written yet (write_pos={self.write_pos})")
        if end - start > self.capacity:
            raise Overrun("request longer than the ring")
        self._check_alive(start)
        i, n = start % self.capacity, end - start
        first = min(n, self.capacity - i)
        out = np.empty(n, dtype=np.int16)
        out[:first] = self._data[i : i + first]
        out[first:] = self._data[: n - first]
        self._check_alive(start)  # the writer may have lapped us during the copy
        return out

    def zero(self, start: int, end: int, keep_margin: int = 16_000) -> None:
        """Wipe samples after they have been decoded (privacy invariant).

        Slots the writer is about to reach (within `keep_margin` of being
        overwritten) are skipped: they are about to be replaced anyway, and
        zeroing them could race with fresh audio.
        """
        oldest_safe = self.write_pos - self.capacity + keep_margin
        start = max(start, oldest_safe)
        if end <= start:
            return
        i, n = start % self.capacity, end - start
        first = min(n, self.capacity - i)
        self._data[i : i + first] = 0
        self._data[: n - first] = 0

    def _check_alive(self, start: int) -> None:
        if start < self.write_pos - self.capacity:
            raise Overrun(f"sample {start} overwritten (write_pos={self.write_pos})")

    def close(self) -> None:
        # Drop numpy views first, or SharedMemory.close() raises BufferError.
        del self._hdr, self._data
        self._shm.close()
        if self._owner:
            self._shm.unlink()
