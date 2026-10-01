"""Typed fact requests (sheet 4): the only thing that may ever leave the laptop.

The model fills in only the variable part through a grammar; the location comes
from settings, rounded to 0.1 degree, and the question, names and transcript are
never included. Requests go to kin-gateway over local IPC; with Wi-Fi off, or
no gateway, every lookup reports "can't check that right now".
"""

from __future__ import annotations

import datetime as dt
import os
import re
from multiprocessing import AuthenticationError

from kin.contracts import FactRequest, FactResponse
from kin.perception.filters import redact
from kin.reasoning.timeparse import parse_when

KIND_OF = {"fact.weather": "weather", "fact.search": "search", "fact.news": "news"}
SPOKEN = {"weather": "the weather", "search": "that online", "news": "the news"}


class FactUnavailable(RuntimeError):
    pass


def home_location() -> tuple[float, float]:
    """Where "the weather" means. Default: Nagpur, as in the demo."""
    return float(os.environ.get("KIN_HOME_LAT", "21.1")), float(os.environ.get("KIN_HOME_LON", "79.1"))


_STOP = set("what whats is the a an of to for in on at me my please tell look up search find any about can you".split())


def _fallback(kind: str, question: str, now: dt.datetime, when_at: dt.datetime | None = None) -> dict:
    if kind == "weather":
        when = parse_when(question, now)
        if when is not None:
            ahead = (when.at - now).total_seconds() / 3600 + (0 if when.has_time else 6)
        elif when_at is not None:  # a plan remembered from earlier speech ("site visit Thursday at 4")
            ahead = (when_at - now).total_seconds() / 3600 + 3
        else:
            ahead = 24
        return {"hours": next((h for h in (6, 12, 24, 48) if h >= ahead), 48)}
    if kind == "news":
        low = question.lower()
        topic = next((t for t, cue in (("tech", "tech"), ("business", "business"), ("sport", "sport"),
                                       ("sport", "cricket"), ("sport", "match")) if cue in low), "local")  # fmt: skip
        return {"topic": topic}
    words = [w for w in re.findall(r"[A-Za-z0-9']+", question) if w.lower() not in _STOP]
    return {"q": " ".join(words)[:80] or question[:80]}


def build_request(
    intent: str,
    question: str,
    now: dt.datetime,
    llm=None,
    people: set[str] | None = None,
    when_at: dt.datetime | None = None,
) -> FactRequest:
    kind = KIND_OF[intent]
    fields = None
    if kind == "weather" and (parse_when(question, now) is not None or when_at is not None):
        fields = _fallback(kind, question, now, when_at)  # a named day: the hours are arithmetic, not language
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
        fields = _fallback(kind, question, now, when_at)
    if kind == "weather":
        fields["lat"], fields["lon"] = home_location()
    if "q" in fields:
        q = redact(fields["q"]).text
        for name in people or ():  # never send who the question is about
            q = re.sub(rf"\b{re.escape(name)}\b", "", q, flags=re.I)
        fields["q"] = " ".join(q.split())[:80]
    return FactRequest(kind=kind, **fields)


class FactService:
    """kin-core's side of E25: send a typed FactRequest to kin-gateway over local IPC.

    Always returns a FactResponse. If the gateway cannot be reached (not started,
    crashed) the response says "offline" without a ledger entry, because nothing
    was sent. `record` receives every response, so kin-core can store the
    LedgerEntry in `egress_log` and show it (spec CL3).
    """

    def __init__(self, session: str | None = None, authkey: bytes | None = None, record=None, timeout: float = 6.0):
        self.session, self.authkey, self.record, self.timeout = session, authkey, record, timeout

    @property
    def available(self) -> bool:
        return self.session is not None

    def fetch(self, req: FactRequest, names=()) -> FactResponse:
        resp = self._call(req, list(names))
        if self.record is not None:
            self.record(resp)
        return resp

    def _call(self, req: FactRequest, names: list[str]) -> FactResponse:
        offline = FactResponse(ok=False, kind=req.kind, data={}, error="offline")
        if self.session is None:
            return offline
        from kin import ipc

        try:
            conn = ipc.connect("gateway", self.session, self.authkey or b"")
            try:
                conn.send({"request": req.model_dump(), "names": names})
                if not conn.poll(self.timeout):
                    return FactResponse(ok=False, kind=req.kind, data={}, error="timeout")
                return FactResponse.model_validate(conn.recv())
            finally:
                conn.close()
        except (OSError, EOFError, ValueError, AuthenticationError):
            return offline


def unavailable_text(resp: FactResponse) -> str:
    """What Kin says when a lookup did not work (F5: plainly, not a guess)."""
    what = SPOKEN[resp.kind]
    if resp.error == "rejected" and resp.data.get("reason") == "search is not configured":
        return "Web search isn't set up on this laptop."
    if resp.error == "timeout":
        return f"I can't check {what} right now: the request timed out."
    return f"I can't check {what} right now."


def _fmt_hour(t: dt.datetime, now: dt.datetime) -> str:
    h = t.strftime("%I %p").lstrip("0")
    return f"{h} today" if t.date() == now.date() else f"{h} {t.strftime('%A')}"


def describe_weather(
    data: dict, now: dt.datetime, target: dt.datetime | None = None, has_time: bool = True
) -> str:
    rows = [(dt.datetime.fromisoformat(r["t"]), r["rain_pct"], r["temp_c"]) for r in data["hourly"]]
    if not rows:
        return "The forecast came back empty."
    if target is not None:
        if has_time:
            lo, hi = target - dt.timedelta(hours=1), target + dt.timedelta(hours=2)
            when = f"around {_fmt_hour(target, now)}"
        else:  # a day was named, not a time: the whole waking day
            day = target.date()
            lo, hi = dt.datetime.combine(day, dt.time(7)), dt.datetime.combine(day, dt.time(22))
            when = "today" if day == now.date() else f"on {target.strftime('%A')}"
        window = [r for r in rows if lo <= r[0] <= hi]
        if not window:
            return "That's further ahead than I can check the forecast."
    else:
        window = rows[: max(1, data.get("hours", 6))]
        when = f"in the next {data.get('hours', 6)} hours"
    peak = max(window, key=lambda r: r[1])
    lo_t, hi_t = min(r[2] for r in window), max(r[2] for r in window)
    temp = f"{round(lo_t)}°C" if round(lo_t) == round(hi_t) else f"{round(lo_t)} to {round(hi_t)}°C"
    if peak[1] >= 30:
        return f"The forecast shows a {peak[1]}% chance of rain {when}, and about {temp}."
    return f"The forecast shows little chance of rain {when} (at most {peak[1]}%), and about {temp}."


def describe_news(data: dict) -> str:
    heads = [h["title"] for h in data["headlines"][:3]]
    return ("Top headlines: " + "; ".join(heads) + ".") if heads else "I found no headlines."


def describe_search(data: dict) -> str:
    rows = data["results"][:3]
    if not rows:
        return "I found nothing online for that."
    return "Online I found: " + "; ".join(f"{r['title']}: {r['snippet']}" if r["snippet"] else r["title"] for r in rows)


def describe(resp: FactResponse, now: dt.datetime, target: dt.datetime | None = None, has_time: bool = True) -> str:
    if resp.kind == "weather":
        return describe_weather(resp.data, now, target, has_time)
    return describe_news(resp.data) if resp.kind == "news" else describe_search(resp.data)
