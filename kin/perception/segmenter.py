"""Episode grouping policy: utterances become ~60 s episodes.

An utterance joins the open episode unless there has been a long pause or the
episode already spans a minute. An open episode closes once the pause since
its last speech, or its span, runs out; closing is what triggers its embedding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodePolicy:
    max_span_s: float = 60.0
    max_gap_s: float = 30.0

    def continues(self, started_at: float, ended_at: float, t_start: float) -> bool:
        return (t_start - ended_at) < self.max_gap_s and (t_start - started_at) < self.max_span_s

    def should_close(self, started_at: float, ended_at: float, now: float) -> bool:
        return (now - ended_at) >= self.max_gap_s or (ended_at - started_at) >= self.max_span_s
