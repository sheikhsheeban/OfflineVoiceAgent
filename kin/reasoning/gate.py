"""Confidence gate: an answer is only spoken if memory backs it.

1. Retrieval: if no episode is close enough to the question, Kin says it has
   nothing, without calling the model.
2. Grounding: every name, weekday, month and number in the model's answer must
   appear in the content of a context item (or in the question). Items that
   contain them become the answer's citations. An ungrounded answer is replaced
   by the best remembered sentence.
3. Conflict: two relevant episodes that mention the same person or thing with
   different days, times or numbers are reported side by side, with a question.

The small model is not trusted to cite or to detect conflicts itself; these
checks are deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kin.reasoning.context import Item, content_words

MIN_SIMILARITY = 0.30  # meaning match alone
MIN_SIMILARITY_WITH_KEYWORD = 0.15  # a keyword match needs only a weak meaning match
NO_MEMORY = "I don't have anything about that."

_NUMWORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "noon": "12",
}  # fmt: skip
# Relative words ("tomorrow") are not checked: the model may correctly turn "Thursday" into one.
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_CAPWORD = re.compile(r"\b([A-Z][a-z]{2,})\b")
_NUMBER = re.compile(r"\b\d+(?::\d\d)?\b")
# Capitalised only because they start a sentence: not names.
_COMMON_CAPS = set(
    "kin the yes no and but she he they you your it its calendar heard here there this that these those what when "
    "where who why how which yeah okay sure not also then she'll he'll they'll will would should could can may might "
    "his her their our for from with about after before until at on in by one two".split()
)


def relevant(hits: list[dict]) -> list[dict]:
    out = []
    for h in hits:
        sim = h.get("vector_similarity") or 0.0
        if sim >= MIN_SIMILARITY or (h.get("keyword_match") and sim >= MIN_SIMILARITY_WITH_KEYWORD):
            out.append(h)
    return out


def _norm(text: str) -> str:
    t = text.lower()
    for w, d in _NUMWORDS.items():
        t = re.sub(rf"\b{w}\b", d, t)
    t = re.sub(r"(\d)\s*(am|pm|a\.m\.|p\.m\.)", lambda m: m.group(1), t)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: m.group(1), t)
    return re.sub(r"\b0(\d)\b", lambda m: m.group(1), t)


def key_terms(text: str) -> set[str]:
    """Facts a listener would act on: names, weekdays, months and numbers."""
    norm = _norm(text)
    terms = {m.lower() for m in _CAPWORD.findall(text) if m.lower() not in _COMMON_CAPS}
    terms -= set(_DAYS.split("|")) | set(_MONTHS.split("|"))
    terms |= set(re.findall(rf"\b({_DAYS}|{_MONTHS})\b", norm))
    terms |= {n.split(":")[0] if n.endswith(":00") else n for n in _NUMBER.findall(norm)}
    return terms


def _contains(haystack_norm: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", haystack_norm) is not None


@dataclass
class GateResult:
    ok: bool
    reason: str  # "grounded" | "declined" | "ungrounded" | "vague" | "retrieval_only"
    cites: list[Item] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def check(answer: str, items: list[Item], question: str) -> GateResult:
    if not answer or answer.lower().startswith("i don't have anything"):
        return GateResult(False, "declined")
    terms = key_terms(answer)
    q_norm = _norm(question)
    # Only content counts as evidence: an episode's recording date is not a fact about its subject.
    norms = {it.id: _norm(it.text) for it in items}
    support = {it.id: sum(_contains(norms[it.id], t) for t in terms) for it in items}
    unsupported = sorted(
        t for t in terms if not any(_contains(n, t) for n in norms.values()) and not _contains(q_norm, t)
    )
    if unsupported:
        return GateResult(False, "ungrounded", unsupported=unsupported)
    top = max(support.values(), default=0)
    # The items that carry the answer: the best-supporting ones, and any carrying at least two of its facts.
    cites = [it for it in sorted(items, key=lambda i: -support[i.id]) if support[it.id] and
             (support[it.id] == top or support[it.id] >= 2)]  # fmt: skip
    if not cites:
        # No names or numbers to check: accept only if the answer clearly echoes one item.
        a = content_words(answer) - content_words(question)
        overlap = sorted(((len(a & content_words(it.text)), n, it) for n, it in enumerate(items)), key=lambda x: -x[0])
        if not overlap or overlap[0][0] < 2:
            return GateResult(False, "vague")
        cites = [overlap[0][2]]
    return GateResult(True, "grounded", cites=cites)


def _values(text: str) -> set[str]:
    return {t for t in key_terms(text) if t.isdigit() or re.fullmatch(_DAYS, t)}


def find_conflict(items: list[Item]) -> tuple[Item, Item] | None:
    """Two episodes naming the same person/thing with different days, times or numbers."""
    eps = [it for it in items if it.kind == "episode"]
    for i, a in enumerate(eps):
        for b in eps[i + 1 :]:
            months = set(_MONTHS.split("|"))  # sharing a month name does not make two things the same subject
            shared = (key_terms(a.text) - _values(a.text) - months) & (key_terms(b.text) - _values(b.text) - months)
            if not shared:
                continue
            va, vb = _values(a.text), _values(b.text)
            if va and vb and va.isdisjoint(vb):
                return a, b
    return None


def best_sentence(items: list[Item], question: str) -> tuple[str, Item] | None:
    """Retrieval-only answer: the remembered sentence (or calendar entry) that best matches the question."""
    q = content_words(question)
    if q & {"birthday", "born", "birth"}:
        q |= {"birthday", "born", "birth"}  # "when is my birthday" is answered by "born on ..."
    best: tuple[int, int, str, Item] | None = None
    for rank, it in enumerate(items):
        for s in re.split(r"(?<=[.!?])\s+|\s\.\.\.\s", it.text):
            score = len(q & content_words(s))
            cand = (score, -rank, s.strip(), it)
            if s.strip() and (best is None or cand[:2] > best[:2]):
                best = cand
    return (best[2], best[3]) if best else None
