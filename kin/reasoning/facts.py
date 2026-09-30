"""Typed fact requests (sheet 4): the only thing that may ever leave the laptop.

The model fills in only the variable part through a grammar; the location comes
from settings, rounded to 0.1 degree, and the question, names and transcript are
never included. Until kin-gateway exists (week 4) there is no way to send a
request, so every lookup reports "can't check that right now", which is also
Kin's behaviour with Wi-Fi off.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, conint, constr, field_validator

from kin.perception.filters import redact
from kin.reasoning.timeparse import parse_when

KIND_OF = {"fact.weather": "weather", "fact.search": "search", "fact.news": "news"}
SPOKEN = {"weather": "the weather", "search": "that online", "news": "the news"}


class FactRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["weather", "search", "news"]
    lat: float | None = None  # rounded to 0.1°
    lon: float | None = None
    hours: conint(ge=0, le=48) = 24
    q: constr(max_length=80) | None = None  # PII-scrubbed
    topic: Literal["local", "tech", "business", "sport"] | None = None

    @field_validator("lat", "lon")
    @classmethod
    def _round(cls, v: float | None) -> float | None:
        return None if v is None else round(v, 1)

    def payload(self) -> dict:
        """Exactly the fields that would be sent for this kind."""
        keep = {"weather": ("kind", "lat", "lon", "hours"), "search": ("kind", "q"), "news": ("kind", "topic")}[
            self.kind
        ]
        return {k: getattr(self, k) for k in keep}


class FactUnavailable(RuntimeError):
    pass


def home_location() -> tuple[float, float]:
    """Where "the weather" means. Default: Nagpur, as in the demo."""
    return float(os.environ.get("KIN_HOME_LAT", "21.1")), float(os.environ.get("KIN_HOME_LON", "79.1"))


_STOP = set("what whats is the a an of to for in on at me my please tell look up search find any about can you".split())


def _fallback(kind: str, question: str, now: dt.datetime) -> dict:
    if kind == "weather":
        when = parse_when(question, now)
        ahead = 24 if when is None else (when.at - now).total_seconds() / 3600 + (0 if when.has_time else 6)
        return {"hours": next((h for h in (6, 12, 24, 48) if h >= ahead), 48)}
    if kind == "news":
        low = question.lower()
        topic = next((t for t, cue in (("tech", "tech"), ("business", "business"), ("sport", "sport"),
                                       ("sport", "cricket"), ("sport", "match")) if cue in low), "local")  # fmt: skip
        return {"topic": topic}
    words = [w for w in re.findall(r"[A-Za-z0-9']+", question) if w.lower() not in _STOP]
    return {"q": " ".join(words)[:80] or question[:80]}


def build_request(
    intent: str, question: str, now: dt.datetime, llm=None, people: set[str] | None = None
) -> FactRequest:
    kind = KIND_OF[intent]
    fields = None
    if kind == "weather" and parse_when(question, now) is not None:
        fields = _fallback(kind, question, now)  # a named day: the hours are arithmetic, not language
    if fields is None and llm is not None:
        from kin.reasoning.llm import load_grammar, load_prompt

        prefill = {
            "weather": '{"kind": "weather", "hours": ',
            "search": '{"kind": "search", "q": "',
            "news": '{"kind": "news", "topic": "',
        }[kind]
        try:
            out = llm.complete(
                load_prompt("fact").replace("{{QUESTION}}", question),
                prefill=prefill,
                grammar=load_grammar("fact").replace("{{KIND}}", kind),
                max_tokens=40,
            )
            import json

            fields = {k: v for k, v in json.loads(prefill + out.text).items() if k != "kind"}
        except Exception:  # noqa: BLE001 - fall back to rules
            fields = None
    if fields is None:
        fields = _fallback(kind, question, now)
    if kind == "weather":
        fields["lat"], fields["lon"] = home_location()
    if "q" in fields:
        q = redact(fields["q"]).text
        for name in people or ():  # never send who the question is about
            q = re.sub(rf"\b{re.escape(name)}\b", "", q, flags=re.I)
        fields["q"] = " ".join(q.split())[:80]
    return FactRequest(kind=kind, **fields)


class FactService:
    """Week 3: no gateway yet, so nothing can be fetched. Week 4 replaces `fetch`."""

    available = False

    def fetch(self, req: FactRequest) -> dict:
        raise FactUnavailable(f"can't check {SPOKEN[req.kind]} right now")
