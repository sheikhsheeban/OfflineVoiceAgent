"""P5 signal extractor: people and claims from a closed episode, by rules.

Runs on an episode's text after the episode is embedded. It finds:
  - people: capitalised names, with a guard against words that are only
    capitalised because they start a sentence;
  - facts: "<Name> said / told me / mentioned (that) ..." becomes a claim with
    its subject, so the consolidator can merge repeats and the gateway knows
    which names to keep out of a search.
Commitments ("I'll send it by Friday") are extracted per utterance by
`kin.reasoning.commitments` (the `TaskCandidate` half of P5, E17).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kin.reasoning.commitments import extract as task_candidates  # noqa: F401 - part of P5's public surface

_CAP = re.compile(r"\b([A-Z][a-z]{2,})\b")
_SAYING = re.compile(
    r"\b(?P<who>[A-Z][a-z]{2,})\s+(?:said|says|told (?:me|us)|mentioned|promised|confirmed|asked)\s+"
    r"(?:that\s+)?(?P<what>[^.?!]+)"
)
# After one of these, a capitalised word at the start of a sentence is a person.
_NAME_VERBS = {
    "said", "says", "told", "asked", "mentioned", "will", "is", "was", "called", "wants", "has", "had", "thinks",
    "agreed", "promised", "confirmed", "sent", "came", "went", "left", "lives", "works",
}  # fmt: skip
_NOT_NAMES = set(
    "the and but she he they you your it its yes no okay hello hey what when where who why how which this that these "
    "those here there then also just well sure kin jarvis monday tuesday wednesday thursday friday saturday sunday "
    "january february march april may june july august september october november december today tomorrow yesterday "
    "please thanks thank sorry maybe actually because while after before until with about from for our their his "
    "her one two three python india nagpur maharashtra".split()
)  # fmt: skip
# A capitalised word is a person only with a cue next to it. Places ("the Hyderabad office", "at Butibori")
# are capitalised too, and the gateway keeps people's names out of searches, so a place must not look like one.
_BEFORE = {
    "with", "from", "to", "call", "called", "email", "emailed", "meet", "met", "ask", "asked", "tell", "told",
    "thanks", "hi", "hello", "dear", "by", "for", "and", "ping", "text", "message",
}  # fmt: skip


@dataclass
class Extracted:
    people: list[str] = field(default_factory=list)
    facts: list[tuple[str, str]] = field(default_factory=list)  # (subject, claim)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def extract(text: str) -> Extracted:
    people: dict[str, None] = {}
    facts: list[tuple[str, str]] = []
    for sent in _sentences(text):
        for m in _CAP.finditer(sent):
            name = m.group(1)
            if name.lower() in _NOT_NAMES:
                continue
            nxt = re.match(r"\s*([a-z']+)", sent[m.end():])
            before = re.search(r"([a-z']+)\s+$", sent[: m.start()], re.I)
            follows_verb = bool(nxt and nxt.group(1) in _NAME_VERBS)
            after_cue = bool(before and before.group(1).lower() in _BEFORE)
            if follows_verb or (m.start() > 0 and after_cue):
                people[name] = None
        for m in _SAYING.finditer(sent):
            who, what = m.group("who"), m.group("what").strip(" ,")
            if who.lower() in _NOT_NAMES or len(what.split()) < 2:
                continue
            facts.append((who, f"{who} said {what}"))
    return Extracted(list(people)[:12], facts[:8])
