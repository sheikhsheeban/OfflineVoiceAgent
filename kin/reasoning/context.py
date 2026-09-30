"""Context builder: retrieved episodes and calendar entries, labelled for citation.

Each item gets an id the answer can be traced to: `e-12` for episode #e-12,
`cal-1` for a calendar entry. Long episodes are cut down to the utterances that
share words with the question, so the prompt stays short on a laptop CPU.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from kin.reasoning.calendar import Event

_WORD = re.compile(r"[a-z0-9']+")
_STOP = set(
    "the a an and or of to in on at for with my me i is was what when where who did said say about "
    "it its that this be do does you your we our he she they them there".split()
)
MAX_ITEM_CHARS = 360


@dataclass
class Item:
    id: str
    kind: str  # "episode" | "calendar"
    when: dt.datetime
    text: str
    ref: dict = field(default_factory=dict)  # episode id, similarity, ... for the UI

    def line(self) -> str:
        if self.kind == "calendar":
            return f"[{self.id}] Calendar: {self.text}"
        return f"[{self.id}] Heard {self.when.strftime('%a %d %b %H:%M')}: {self.text}"


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def _trim(ep: dict, question: str) -> str:
    utts = ep["utterances"]
    text = " ".join(u["text"] for u in utts)
    if len(text) <= MAX_ITEM_CHARS:
        return text
    q = content_words(question)
    scored = [(len(q & content_words(u["text"])), i) for i, u in enumerate(utts)]
    keep: set[int] = set()
    for score, i in sorted(scored, reverse=True):
        if score == 0 and keep:
            break
        keep |= {i, i + 1} if i + 1 < len(utts) else {i}
        if len(" ".join(utts[j]["text"] for j in sorted(keep))) > MAX_ITEM_CHARS:
            break
    out = " ... ".join(utts[j]["text"] for j in sorted(keep))
    return out[:MAX_ITEM_CHARS]


def build(hits: list[dict], events: list[Event], question: str) -> list[Item]:
    items = [
        Item(f"e-{h['id']}", "episode", dt.datetime.fromtimestamp(h["started_at"]), _trim(h, question),
             {"episode_id": h["id"], "similarity": h.get("vector_similarity"), "keyword": h.get("keyword_match")})
        for h in hits
    ]  # fmt: skip
    items += [Item(f"cal-{i + 1}", "calendar", e.start, e.describe(), {"uid": e.uid}) for i, e in enumerate(events)]
    return items


def render(items: list[Item]) -> str:
    return "\n".join(it.line() for it in items)
