"""Intent router: rules first, then centroid match, then the model for close calls.

Each intent's centroid is the normalised mean of its example embeddings from
spec/intents.toml. A request clearly nearest one centroid is routed without the
language model; a close call goes to the model with a grammar that only allows
intent names. A close call between `recall.query` and a `fact.*` intent is kept
as both (the umbrella-and-site-visit question needs memory and a forecast).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field

import numpy as np

from kin.config import SPEC_DIR

FACT_INTENTS = ("fact.weather", "fact.search", "fact.news")

# High-precision patterns that beat embeddings on short, formulaic requests.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "meta.history",  # about the conversation itself: answered from the turns log, never by the model
        re.compile(
            r"\bwhat (was|is) my (last|previous|latest) (question|request)\b|\bwhat did i (just )?(ask|say)( you)?\b"
            r"|\b(repeat|say) (that|it)( again)?\b|\bwhat did you (just )?say\b|\bsay that again\b",
            re.I,
        ),
    ),
    (
        "reminder.set",
        re.compile(
            r"\b(remind|tell|ping|wake|alert)\s+me\b.*\b(in|after)\s+(\d+|a|an|half|one|two|three|five|ten|fifteen|twenty|thirty)\b"
            r"|^\s*(in|after)\s+\S+\s+(seconds?|minutes?|mins?|hours?)\b"
            r"|\b(remind|tell|ping)\s+me\s+at\s+\d|^\s*at\s+\d{1,2}(:\d\d)?\s*(am|pm)?\s*,?\s*(remind|tell|ping)\s+me"
            r"|\bset\s+(a\s+)?(reminder|timer)\b",
            re.I,
        ),
    ),
    (
        "task.list",
        re.compile(
            r"\b(what'?s|what is|show|read|list|any)\b.*\b(my|the)?\s*(to-?do|tasks?|list)\b"
            r"|\bwhat (do|else do) i (need|have) to do\b",
            re.I,
        ),
    ),
    (
        "digest.today",
        re.compile(r"\b(commit(ted)?|promised?|agreed?)\b.*\btoday\b|\bsummar(y|ise|ize)\b.*\b(day|today)\b", re.I),
    ),
    (
        "task.add",
        re.compile(
            r"^\s*(please\s+)?(remind me to|add|put|note down|make a note|don'?t let me forget)\b"
            r"|\b(to|on|onto) my (list|tasks|to-?do( list)?)\b|\badd (it|that|this)\b",
            re.I,
        ),
    ),
]

_QUESTION = re.compile(
    r"\?\s*$|^\s*(what|when|where|who|whom|whose|which|why|how|did|does|do|is|are|was|were|should|can|could|will)\b",
    re.I,
)
# A question that needs today's world, not built-in knowledge: it may go online instead.
_CURRENT = re.compile(
    r"\b(today|tonight|right now|currently|latest|current|this (week|month|year)|news|price of|stock|score|"
    r"who won|traffic|is it open|open now)\b",
    re.I,
)
_PERSONAL = re.compile(r"\b(my|mine|me|i|i'm|i've|we|our|us)\b", re.I)

# A fact question that also points back at something heard needs both sides.
_MEMORY_CUE = re.compile(r"\b(mentioned|said|told me|talked about|discussed|we agreed|remember|you heard)\b", re.I)
_FACT_CUE = {
    "fact.weather": re.compile(r"\b(umbrella|rain(ing|y)?|weather|forecast|temperature|jacket|sunny|storm)\b", re.I),
    "fact.news": re.compile(r"\b(news|headlines?)\b", re.I),
}


@dataclass
class Route:
    intent: str
    method: str  # "rules" | "centroid" | "model" | "mixed"
    scores: dict[str, float] = field(default_factory=dict)
    also: str | None = None  # a second intent to serve alongside (mixed recall + fact)
    q_ret: str | None = None  # the question with the one before it, for a follow-up

    @property
    def top(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)


class Router:
    def __init__(self, embedder, llm=None, memory_probe=None):
        """`memory_probe(text) -> bool`: does memory hold something clearly on this topic?"""
        self.memory_probe = memory_probe
        with open(SPEC_DIR / "intents.toml", "rb") as f:
            spec = tomllib.load(f)
        self.min_similarity = float(spec["router"]["min_similarity"])
        self.min_margin = float(spec["router"]["min_margin"])
        self.intents: dict[str, dict] = spec["intents"]
        self.embedder = embedder
        self.llm = llm
        self.names = list(self.intents)
        cents = []
        for name in self.names:
            v = embedder.embed(self.intents[name]["examples"]).mean(axis=0)
            cents.append(v / np.linalg.norm(v))
        self.centroids = np.stack(cents)

    def scores(self, text: str) -> dict[str, float]:
        sims = self.centroids @ self.embedder.embed_one(text)
        return {n: round(float(s), 4) for n, s in zip(self.names, sims, strict=True)}

    def route(self, text: str) -> Route:
        scores = self.scores(text)
        for intent, pattern in _RULES:
            if pattern.search(text):
                return Route(intent, "rules", scores)
        (i1, s1), (i2, s2) = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
        if i1 == "general.knowledge" and not _PERSONAL.search(text):
            if _CURRENT.search(text):
                return Route("fact.search", "centroid", scores)  # general knowledge would be out of date
            if self.memory_probe is not None and self.memory_probe(text):
                return Route("recall.query", "memory", scores)  # what the user said about it comes first
        if i1 in FACT_INTENTS and _MEMORY_CUE.search(text):
            return Route("recall.query", "mixed", scores, also=i1)
        if i1 == "recall.query":
            for fact, cue in _FACT_CUE.items():
                if cue.search(text):
                    return Route("recall.query", "mixed", scores, also=fact)
        unsure = s1 < self.min_similarity or s1 - s2 < self.min_margin
        if unsure and _QUESTION.search(text) and self.memory_probe is not None and self.memory_probe(text):
            return Route("recall.query", "memory", scores)  # memory clearly has something on it: recall first
        if s1 - s2 >= self.min_margin and (
            s1 >= self.min_similarity or i1 not in ("task.add", "reminder.set", "meta.history")
        ):
            # A clear winner, even at modest similarity, beats asking a 1B model. Tasks and
            # reminders change state, so they also need a strong match (or a rule).
            return Route(i1, "centroid", scores)
        pair = {i1, i2}
        if "recall.query" in pair and pair & set(FACT_INTENTS) and s2 >= self.min_similarity - 0.1:
            fact = (pair & set(FACT_INTENTS)).pop()
            return Route("recall.query", "mixed", scores, also=fact)

        # Close call. Narrow the choices before asking the small model:
        ranked = [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
        # Things Kin changes (tasks, reminders) need explicit wording, never a guess.
        allowed = set(ranked) - {"task.add", "reminder.set", "meta.history"}
        if _PERSONAL.search(text):
            allowed -= {"fact.search", "fact.news", "general.knowledge"}  # personal questions never go online
            if not _FACT_CUE["fact.weather"].search(text):
                allowed -= {"fact.weather"}  # "when is my birthday" is memory, not a forecast
        candidates = [i for i in ranked if i in allowed][:3]
        if _QUESTION.search(text) and _PERSONAL.search(text) and "recall.query" in candidates:
            return Route("recall.query", "personal", scores)  # "what is my ...": memory, or honestly nothing
        if len(candidates) == 1 or self.llm is None:
            return Route(candidates[0], "centroid", scores)
        try:
            return Route(self._ask_model(text, candidates), "model", scores)
        except Exception:  # noqa: BLE001 - model unavailable: best allowed centroid
            return Route(candidates[0], "centroid", scores)

    def _ask_model(self, text: str, ranked: list[str]) -> str:
        from kin.reasoning.llm import load_grammar, load_prompt

        listing = "\n".join(f"- {n}: {self.intents[n]['description']}" for n in ranked)
        prompt = load_prompt("route").replace("{{INTENTS}}", listing).replace("{{QUESTION}}", text)
        names = " | ".join(f'"{n}"' for n in ranked)
        grammar = load_grammar("route").replace("{{INTENTS}}", names)
        out = self.llm.complete(prompt, prefill='{"intent": "', grammar=grammar, max_tokens=12)
        return out.text.split('"')[0]
