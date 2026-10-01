"""The kin-audio pipeline for each captured block.

Two consumers read every block, in parallel (sheet 2, E41 and E07):

* A5, the wake word, scores every 80 ms chunk (1280 samples). It is never gated:
  feeding it only "loud" or "speech" audio cut off the first syllable and reset
  its context at every pause, and wake words were missed live.
* A3 -> A4 -> endpoints: the energy gate and Silero VAD work in 512-sample
  windows taken from a carry-over buffer (1280 is not a multiple of 512). VAD
  only runs while the gate is open, which keeps idle listening cheap.

When the wake score is at or above the threshold for `wake_patience`
consecutive chunks, the segment being spoken is cut where the wake phrase
starts and a command window opens (see `Endpointer.open_command`).

Output is a list of small dict events for kin-core; audio stays in the ring and
is referenced by sample index.
"""

from __future__ import annotations

import time

import numpy as np

from kin.audio.endpoint import Endpointer, Segment, ms_to_samples
from kin.audio.gate import EnergyGate
from kin.audio.ring import Overrun, Ring
from kin.audio.vad import SileroVAD
from kin.audio.wakeword import WakeWord
from kin.config import CHUNK, FRAME, AudioSettings

VAD_PRIME_FRAMES = 8  # ~256 ms of pre-roll to warm the VAD state after silence


class Detector:
    def __init__(self, ring: Ring, vad: SileroVAD, wake: WakeWord | None, s: AudioSettings = AudioSettings()):
        self.ring = ring
        self.vad = vad
        self.wake = wake
        self.s = s
        self.gate = EnergyGate(s.gate_margin_db, s.gate_min_dbfs, max(1, ms_to_samples(s.gate_hangover_ms) // FRAME))
        self.endpoint = Endpointer(s)
        self._gate_was_open = False
        self._last_wake: int | None = None
        self._streak = 0  # consecutive chunks at or above the threshold
        self._armed_until: int | None = None  # a wake-only command arms the next utterance
        self._pos: int | None = None  # absolute index of the next sample expected
        self._vad_buf = np.empty(0, np.int16)
        self._vad_at = 0  # absolute index of _vad_buf[0]
        self._wake_buf = np.empty(0, np.int16)
        self._wake_at = 0
        self.last_prob = 0.0
        self.frames_total = 0
        self.frames_vad = 0

    def process(self, start: int, block: np.ndarray) -> list[dict]:
        """Feed captured audio that starts at ring index `start` (any length; 1280 live)."""
        if self._pos != start:  # first block, or the reader skipped ahead: start the buffers afresh
            self._vad_buf, self._vad_at = np.empty(0, np.int16), start
            self._wake_buf, self._wake_at = np.empty(0, np.int16), start
        self._pos = start + block.shape[0]
        self._vad_buf = np.concatenate([self._vad_buf, block])
        self._wake_buf = np.concatenate([self._wake_buf, block])

        events: list[dict] = []
        # Work through both in time order, so a wake word cuts the segment at the right sample.
        while True:
            vad_end = self._vad_at + FRAME if self._vad_buf.shape[0] >= FRAME else None
            wake_end = self._wake_at + CHUNK if self._wake_buf.shape[0] >= CHUNK and self.wake else None
            if vad_end is None and wake_end is None:
                break
            if wake_end is not None and (vad_end is None or wake_end <= vad_end):
                chunk, self._wake_buf = self._wake_buf[:CHUNK], self._wake_buf[CHUNK:]
                self._wake_at = wake_end
                events += self._wake_chunk(wake_end, chunk)
            else:
                frame, self._vad_buf = self._vad_buf[:FRAME], self._vad_buf[FRAME:]
                at, self._vad_at = self._vad_at, vad_end
                events += self._vad_frame(at, frame)
        if self.wake is None:
            self._wake_buf = np.empty(0, np.int16)
        return events

    def _wake_chunk(self, end: int, chunk: np.ndarray) -> list[dict]:
        assert self.wake is not None
        score = self.wake.score(chunk)
        self._streak = self._streak + 1 if score >= self.s.wake_threshold else 0
        if self._streak < self.s.wake_patience:
            return []
        refractory = ms_to_samples(self.s.wake_refractory_ms)
        if self._last_wake is not None and end - self._last_wake < refractory:
            return []
        self._last_wake = end
        events: list[dict] = [{"type": "wake", "at": end, "score": round(score, 3), "t": time.time()}]
        cut = self.endpoint.open_command(end - ms_to_samples(self.s.wake_span_ms), end)
        if cut is not None:
            events.append(self._segment_event(cut))
        self._armed_until = None
        return events

    def _vad_frame(self, start: int, frame: np.ndarray) -> list[dict]:
        self.frames_total += 1
        is_open = self.gate.update(frame)
        prob = 0.0
        if is_open:
            if not self._gate_was_open:
                self._prime(start)
            prob = self.vad(frame)
            self.frames_vad += 1
        self._gate_was_open = is_open
        self.last_prob = prob
        seg = self.endpoint.push(start, prob)
        return [self._segment_event(seg)] if seg is not None else []

    def flush(self, now: int) -> list[dict]:
        seg = self.endpoint.flush(now)
        return [self._segment_event(seg)] if seg else []

    def _prime(self, start: int) -> None:
        self.vad.reset()
        lo = max(0, start - VAD_PRIME_FRAMES * FRAME, self.ring.write_pos - self.ring.capacity)
        try:
            for i in range(lo, start - FRAME + 1, FRAME):
                self.vad(self.ring.read(i, i + FRAME))
        except Overrun:
            pass

    def _segment_event(self, seg: Segment) -> dict:
        mode = seg.mode
        if mode == "command":
            # The wake word with nothing after it: the command is the next utterance.
            if not seg.has_command:
                self._armed_until = seg.end + ms_to_samples(5000)
        elif self._armed_until is not None and seg.start <= self._armed_until:
            mode = "command"
            self._armed_until = None
        return {
            "type": "segment",
            "start": seg.start,
            "end": seg.end,
            "mode": mode,
            "wake": mode == "command",
            "wake_at": seg.wake_at,
            "forced": seg.forced,
            "t": time.time(),
        }

    def status(self) -> dict:
        return {
            "type": "status",
            "level_db": round(self.gate.level, 1),
            "floor_db": round(self.gate.floor, 1),
            "gate": self.gate.is_open,
            "vad": round(self.last_prob, 2),
            "speech": self.endpoint.in_speech,
            "mode": "command" if self.endpoint.in_command else "ambient",
            # Peak since the last status message, so a short wake word still shows on the meter.
            "wake_score": round(self.wake.take_peak(), 2) if self.wake else None,
            "wake_threshold": self.s.wake_threshold,
            "vad_duty": round(self.frames_vad / max(1, self.frames_total), 3),
        }
