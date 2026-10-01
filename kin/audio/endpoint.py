"""Turns per-frame speech probabilities into speech segments.

Works purely on absolute sample indices in the ring, so a finished segment is
handed to kin-core as `(start, end)` and the audio itself never leaves shared
memory.

A segment is `ambient` unless the wake word opened a command window
(`open_command`): that cuts whatever was being said at the start of the wake
phrase, and the following audio is one `command` segment until speech ends.
The window allows a longer pause right after the wake word ("Hey Jarvis ...
when is my birthday") and never runs longer than `command_max_ms` past it.
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
    mode: str = "ambient"  # or "command": opened by the wake word
    wake_at: int | None = None  # where the wake word was detected (command segments)
    has_command: bool = False  # speech after the wake word (else it was the wake word alone)

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
        self.command_wait = ms_to_samples(s.command_wait_ms)
        self.command_max = ms_to_samples(s.command_max_ms)
        self._run = 0  # consecutive frames above `on` while idle
        self._run_start = 0
        self._seg_start: int | None = None
        self._voice_first = 0
        self._voice_last = 0  # end index of the latest frame above `off`
        self._floor = 0  # segments never start before this (end of the previous one)
        self._wake_at: int | None = None  # set while a command window is open
        self._heard_command = False  # speech after the wake word
        self._gap_after_wake = False  # a quiet frame since the wake word
        self.command_onset = ms_to_samples(400)  # speech running on this far past detection is the command

    @property
    def in_speech(self) -> bool:
        return self._seg_start is not None

    @property
    def in_command(self) -> bool:
        return self._wake_at is not None

    @property
    def segment_start(self) -> int | None:
        return self._seg_start

    def open_command(self, phrase_start: int, wake_at: int) -> Segment | None:
        """The wake word ended at `wake_at`. Close the ambient speech before the phrase (returned,
        if long enough to keep) and start a command segment at the phrase."""
        cut = None
        start = max(self._floor, phrase_start)
        if self._seg_start is not None:
            if self._seg_start < start and self._wake_at is None:
                cut = self._cut(start)
            else:
                start = self._seg_start  # the open segment began with the phrase: it becomes the command
        self._seg_start = start
        self._voice_first = start
        self._voice_last = wake_at  # the wake word itself was speech
        self._wake_at = wake_at
        self._heard_command = False
        self._gap_after_wake = False
        self._run = 0
        return cut

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
        if self._wake_at is not None and frame_start >= self._wake_at and not self._heard_command:
            # The detector fires before the phrase ends, so the rest of "...vis" is not the command yet.
            # The command is speech that starts after a pause, or speech that runs on well past it.
            if prob < self.off:
                self._gap_after_wake = True
            elif (self._gap_after_wake and prob >= self.on) or frame_end - self._wake_at >= self.command_onset:
                self._heard_command = True

        if self._wake_at is not None:
            if frame_end - self._wake_at >= self.command_max:
                return self._close(frame_end, forced=True)
            # Before the command starts, allow a pause after the wake word; then the usual end.
            quiet = self.end_samples if self._heard_command else self.command_wait
            if frame_end - self._voice_last >= quiet:
                return self._close(min(self._voice_last + self.post_roll, frame_end))
            return None

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

    def _cut(self, at: int) -> Segment | None:
        """End the open ambient segment at `at` (the wake phrase starts there)."""
        start, voiced = self._seg_start, min(self._voice_last, at) - self._voice_first
        assert start is not None
        self._seg_start = None
        self._floor = at
        if voiced < self.min_samples:
            return None
        return Segment(start, at)

    def _close(self, end: int, forced: bool = False) -> Segment | None:
        start, voiced = self._seg_start, self._voice_last - self._voice_first
        wake_at, heard = self._wake_at, self._heard_command
        self._seg_start = None
        self._wake_at = None
        self._heard_command = False
        self._run = 0
        self._floor = end
        assert start is not None
        if wake_at is not None:  # a command is always handed on, even the wake word alone
            return Segment(start, end, forced=forced, mode="command", wake_at=wake_at, has_command=heard)
        if voiced < self.min_samples:
            return None
        return Segment(start, end)
