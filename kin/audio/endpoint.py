"""Turns per-frame speech probabilities into speech segments.

Works purely on absolute sample indices in the ring, so a finished segment is
handed to kin-core as `(start, end)` and the audio itself never leaves shared
memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from kin.config import FRAME, SAMPLE_RATE, AudioSettings


def ms_to_samples(ms: float) -> int:
    return int(ms * SAMPLE_RATE / 1000)


@dataclass
class Segment:
    start: int
    end: int
    forced: bool = False  # cut at max length while speech continued

    @property
    def seconds(self) -> float:
        return (self.end - self.start) / SAMPLE_RATE


class Endpointer:
    def __init__(self, s: AudioSettings = AudioSettings()):
        self.on, self.off = s.vad_on, s.vad_off
        self.start_frames = max(1, round(ms_to_samples(s.speech_start_ms) / FRAME))
        self.end_samples = ms_to_samples(s.speech_end_ms)
        self.pre_roll = ms_to_samples(s.pre_roll_ms)
        self.post_roll = ms_to_samples(200)
        self.min_samples = ms_to_samples(s.min_segment_ms)
        self.max_samples = ms_to_samples(s.max_segment_ms)
        self._run = 0  # consecutive frames above `on` while idle
        self._run_start = 0
        self._seg_start: int | None = None
        self._voice_first = 0
        self._voice_last = 0  # end index of the latest frame above `off`
        self._floor = 0  # segments never start before this (end of the previous one)

    @property
    def in_speech(self) -> bool:
        return self._seg_start is not None

    @property
    def segment_start(self) -> int | None:
        return self._seg_start

    def push(self, frame_start: int, prob: float) -> Segment | None:
        """Feed the probability for the frame starting at `frame_start`."""
        frame_end = frame_start + FRAME
        if self._seg_start is None:
            if prob >= self.on:
                if self._run == 0:
                    self._run_start = frame_start
                self._run += 1
                if self._run >= self.start_frames:
                    self._seg_start = max(self._floor, self._run_start - self.pre_roll, 0)
                    self._voice_first = self._run_start
                    self._voice_last = frame_end
                    self._run = 0
            else:
                self._run = 0
            return None

        if prob >= self.off:
            self._voice_last = frame_end
        if frame_end - self._seg_start >= self.max_samples:
            seg = Segment(self._seg_start, frame_end, forced=True)
            # Speech continues: the next segment starts right here, no pre-roll.
            self._seg_start = frame_end
            self._voice_first = frame_end
            self._floor = frame_end
            return seg
        if frame_end - self._voice_last >= self.end_samples:
            return self._close(min(self._voice_last + self.post_roll, frame_end))
        return None

    def flush(self, now: int) -> Segment | None:
        """Close any open segment (stream ended or gate closed for good)."""
        if self._seg_start is None:
            self._run = 0
            return None
        return self._close(min(self._voice_last + self.post_roll, now))

    def _close(self, end: int) -> Segment | None:
        start, voiced = self._seg_start, self._voice_last - self._voice_first
        self._seg_start = None
        self._run = 0
        self._floor = end
        assert start is not None
        if voiced < self.min_samples:
            return None
        return Segment(start, end)
