"""Commitments heard in conversation ("I'll send it by Friday") become task candidates.

Rules plus the time parser, no model: a first-person promise followed by an
action. Candidates wait for the user to confirm them in the UI; nothing is added
to the to-do list on Kin's own judgement.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from kin.reasoning.timeparse import parse_when

_PROMISE = re.compile(
    r"\b(?:i'll|i will|i'm going to|i am going to|i'm gonna|i shall|i need to|i have to|i must|"
    r"i promise to|i promised to|i'll make sure to|remind me to|we'll|we will|we need to)\s+(?P<act>[^.?!;]+)",
    re.IGNORECASE,
)
# Not actions: states, feelings, vague futures.
_NOT_ACTION = re.compile(
    r"^(be|not|never|probably|maybe|see|think|try|have been|get back to you|let you know|go now|come|miss)\b", re.I
)
_FILLER = re.compile(r"\b(okay|ok|then|actually|definitely|surely|also|just)\b", re.I)


@dataclass
class Candidate:
    title: str
    due_at: dt.datetime | None
    has_time: bool
    sentence: str


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def extract(text: str, now: dt.datetime) -> list[Candidate]:
    out = []
    for sentence in _sentences(text):
        if sentence.endswith("?"):
            continue
        for m in _PROMISE.finditer(sentence):
            act = m.group("act").strip(" ,")
            if _NOT_ACTION.match(act):
                continue
            when = parse_when(act, now)
            title = when.strip_from(act) if when else act
            title = " ".join(_FILLER.sub(" ", title).split()).strip(" ,")
            words = title.split()
            if not 2 <= len(words) <= 14:
                continue
            out.append(Candidate(title[0].upper() + title[1:], when.at if when else None,
                                 bool(when and when.has_time), sentence))  # fmt: skip
    return out
