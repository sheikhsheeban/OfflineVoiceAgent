"""Energy gate: a cheap first filter so VAD and wake word only run on sound.

Tracks the noise floor with a slow-rising, fast-falling estimate and opens when
a frame is `margin_db` above it (and above an absolute minimum). A hangover
keeps it open through short pauses between words.

For the first second the gate stays closed and the floor follows the room
quickly in both directions. Without that, the few digital-silence frames a
device delivers as it starts drag the floor far below the real room, and room
noise then counts as sound for many seconds.
"""

from __future__ import annotations

import numpy as np

FLOOR_MIN_DBFS = -90.0  # dropouts of digital silence must not sink the floor further


def frame_dbfs(frame: np.ndarray) -> float:
    x = frame.astype(np.float32) * (1.0 / 32768.0)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return float(20.0 * np.log10(rms + 1e-9))


class EnergyGate:
    def __init__(
        self,
        margin_db: float = 9.0,
        min_dbfs: float = -55.0,
        hangover_frames: int = 25,
        initial_floor_dbfs: float = -60.0,
        calibration_frames: int = 31,  # ~1 s
    ):
        self.margin_db = margin_db
        self.min_dbfs = min_dbfs
        self.hangover_frames = hangover_frames
        self.floor = initial_floor_dbfs
        self._hang = 0
        self._calibrating = calibration_frames
        self.level = -120.0

    def update(self, frame: np.ndarray) -> bool:
        """Feed one frame; return True while the gate is open."""
        db = frame_dbfs(frame)
        self.level = db
        if self._calibrating > 0:
            self._calibrating -= 1
            self.floor = max(FLOOR_MIN_DBFS, self.floor + 0.3 * (db - self.floor))
            return False
        loud = db > max(self.floor + self.margin_db, self.min_dbfs)
        if loud:
            self._hang = self.hangover_frames
            # Rise slowly (~6 s time constant) so speech does not become "the
            # floor", but a noisier room is learned within a few seconds.
            self.floor += 0.005 * (db - self.floor)
        else:
            # Follow quiet frames quickly (down) and moderately (up).
            rate = 0.2 if db < self.floor else 0.02
            self.floor = max(FLOOR_MIN_DBFS, self.floor + rate * (db - self.floor))
            if self._hang > 0:
                self._hang -= 1
        return loud or self._hang > 0

    @property
    def is_open(self) -> bool:
        return self._hang > 0
