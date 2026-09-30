"""The kin-audio pipeline for one frame: energy gate -> VAD -> wake word -> endpoints.

VAD and wake word only run while the gate is open, which keeps idle listening
cheap. Output is a list of small dict events for kin-core; audio stays in the
ring and is referenced by sample index.
"""

from __future__ import annotations

import time

import numpy as np

from kin.audio.endpoint import Endpointer, Segment, ms_to_samples
from kin.audio.gate import EnergyGate
from kin.audio.ring import Overrun, Ring
from kin.audio.vad import SileroVAD
from kin.audio.wakeword import CONTEXT_SAMPLES, WakeWord
from kin.config import FRAME, AudioSettings

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
        self._armed_until: int | None = None  # a wake-only utterance arms the next one
        self.last_prob = 0.0
        self.frames_total = 0
        self.frames_vad = 0

    def process(self, start: int, frame: np.ndarray) -> list[dict]:
        events: list[dict] = []
        self.frames_total += 1
        is_open = self.gate.update(frame)
        prob = 0.0
        if is_open:
            if not self._gate_was_open:
                self._prime(start)
            prob = self.vad(frame)
            self.frames_vad += 1
            if self.wake is not None:
                score = self.wake.feed(frame)
                if score is not None and score >= self.s.wake_threshold:
                    ev = self._on_wake(start + FRAME, score)
                    if ev:
                        events.append(ev)
        self._gate_was_open = is_open
        self.last_prob = prob

        seg = self.endpoint.push(start, prob)
        if seg is not None:
            events.append(self._segment_event(seg))
        return events

    def flush(self, now: int) -> list[dict]:
        seg = self.endpoint.flush(now)
        return [self._segment_event(seg)] if seg else []

    def _prime(self, start: int) -> None:
        self.vad.reset()
        lo = max(0, start - VAD_PRIME_FRAMES * FRAME, self.ring.write_pos - self.ring.capacity)
        try:
            for i in range(lo, start - FRAME + 1, FRAME):
                self.vad(self.ring.read(i, i + FRAME))
            if self.wake is not None:
                self.wake.prime(self.ring.read(max(0, start - CONTEXT_SAMPLES, lo), start))
        except Overrun:
            pass

    def _on_wake(self, at: int, score: float) -> dict | None:
        refractory = ms_to_samples(self.s.wake_refractory_ms)
        if self._last_wake is not None and at - self._last_wake < refractory:
            return None
        self._last_wake = at
        return {"type": "wake", "at": at, "score": round(score, 3), "t": time.time()}

    def _segment_event(self, seg: Segment) -> dict:
        wake = False
        lw = self._last_wake
        if lw is not None and seg.start <= lw <= seg.end + ms_to_samples(300):
            wake = True
            # Wake word with (almost) nothing after it: the command is the next utterance.
            if seg.end - lw < ms_to_samples(500):
                self._armed_until = seg.end + ms_to_samples(5000)
        elif self._armed_until is not None and seg.start <= self._armed_until:
            wake = True
            self._armed_until = None
        return {
            "type": "segment",
            "start": seg.start,
            "end": seg.end,
            "wake": wake,
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
            "wake_score": round(self.wake.last_score, 2) if self.wake else None,
            "vad_duty": round(self.frames_vad / max(1, self.frames_total), 3),
        }
