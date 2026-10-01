"""Answering a command: route it, run the tools it needs, and gate the result.

| intent        | handled by                                         |
| ------------- | -------------------------------------------------- |
| recall.query  | retrieval + calendar + language model + gate      |
| task.add      | rules; the model only when the due time is unclear |
| task.list     | rules                                              |
| reminder.set  | rules + scheduler                                  |
| digest.today  | stored tasks and episodes, no model                |
| fact.*        | typed fact request -> kin-gateway -> forecast/news |
| general.knowledge | the model alone, with a "may be out of date" note |
| meta.history  | the turns log (last question, repeat), no model    |
| unsupported   | honest decline, offer to save a note               |

Every model call happens on kin-core's heavy lane (the caller's thread).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from dataclasses import asdict, dataclass, field

from kin.output.voice import spoken_form
from kin.perception.filters import redact
from kin.reasoning import context as ctx
from kin.reasoning import gate
from kin.reasoning.calendar import Calendar
from kin.reasoning.facts import FactService, build_request, unavailable_text
from kin.reasoning.facts import describe as describe_fact
from kin.reasoning.llm import LLMUnavailable, load_grammar, load_prompt
from kin.reasoning.router import _CURRENT, Router
from kin.reasoning.timeparse import WEEKDAYS, describe, parse_when

log = logging.getLogger("kin.reasoning")

_TASK_LEAD = re.compile(
    r"^\s*(please\s+)?(can you\s+)?(remind me to|don'?t let me forget to|make a note to|note down that i have to|"
    r"note down|add a task to|add|put|i need to|i have to)\s+",
    re.I,
)
_TASK_TAIL = re.compile(r"\s*,?\s*(to|on|in)\s+(my|the)\s+(to-?do\s+)?(list|tasks?)\s*$|\s*,?\s*add it.*$", re.I)
_REMINDER_WORDS = re.compile(
    r"\b(remind|tell|ping|alert|wake)\s+me\b(\s+up)?(\s+to)?|\bset (a )?(reminder|timer)( for)?\b", re.I
)
_RELATIVE_DAY = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this (morning|afternoon|evening|week)|next|last|ago)\b", re.I
)
_VAGUE_WHEN = re.compile(r"\b(after|before|later|soon|this week|end of|when i|once)\b", re.I)
_FOLLOWUP = re.compile(r"^\s*(and|what about|how about|also|then|so)\b|^\s*(it|that|they|he|she|there)\b", re.I)
# "What is X" style questions that memory cannot answer and general knowledge can.
_GENERIC = re.compile(r"^\s*(what (is|are|does)|define|explain|how (does|do|is|are)|why (is|do|does))\b", re.I)
_PERSONAL_WORDS = re.compile(r"\b(my|mine|me|i|we|our|us|yesterday|last|earlier|said|told|mentioned)\b", re.I)
_REPEAT = re.compile(r"\b(repeat|say that again|what did you)", re.I)
CONVERSATION_S = 300.0  # the buffer of recent turns resets after five minutes of silence
BUFFER_TURNS = 4


@dataclass
class Answer:
    question: str
    text: str
    intent: str
    route: str
    gate: str = ""
    citations: list[dict] = field(default_factory=list)
    fact_request: dict | None = None
    task: dict | None = None
    reminder: dict | None = None
    offer_note: bool = False
    llm_used: bool = False
    timings: dict = field(default_factory=dict)
    voice_text: str = ""  # what is said aloud: no ids, at most ~120 tokens
    abstained: bool = False  # Kin declined or could not check: nothing was asserted
    fact: dict | None = None  # what the gateway did: kind, ok, error, host, bytes

    def to_dict(self) -> dict:
        return {"type": "answer", **asdict(self), "t": time.time()}


def _cite(item: ctx.Item) -> dict:
    return {"id": item.id, "kind": item.kind, "when": item.when.timestamp(), "text": item.text[:200], **item.ref}


def _clean_title(text: str) -> str:
    t = _TASK_TAIL.sub("", _TASK_LEAD.sub("", text.strip())).strip(" ,.")
    return t[:1].upper() + t[1:] if t else t


class Reasoner:
    def __init__(
        self, memory, llm, calendar: Calendar | None = None, facts: FactService | None = None, clock=None, on_state=None
    ):
        self.memory = memory
        self.llm = llm
        self.router = Router(memory.embedder, llm, memory_probe=self._memory_has)
        self.calendar = calendar or Calendar(None)
        self.facts = facts or FactService()
        self.clock = clock or dt.datetime.now
        self.on_state = on_state or (lambda reason, active: None)  # O1: thinking / fetching

    def warm(self) -> None:
        """Load the model and evaluate the fixed start of the answer prompt, so the first
        spoken answer pays for neither (the demo's "warm the model before going on stage")."""
        try:
            self.llm.warm(load_prompt("answer").split("{{EXTRA}}")[0])
        except LLMUnavailable as e:
            log.warning("cannot warm the model: %s", e)

    def _memory_has(self, q: str) -> bool:
        hits = self.memory.search(q, k=3, qvec=self.memory.embedder.embed_one(q))
        return any(h["keyword_match"] for h in gate.relevant(hits))

    # -- entry point -------------------------------------------------------------------

    def handle(self, question: str) -> Answer:
        t0 = time.perf_counter()
        question = question.strip()
        prev = self._recent_turns()
        effective = self._resolve_followup(question, prev)
        self.on_state("thinking", True)
        try:
            route = self.router.route(question)  # rules see only what was actually said
            if effective != question:
                base = prev[-1]["intent"].split("+")[0]
                if route.method != "rules" and base in ("recall.query", "fact.weather", "fact.news"):
                    # "and on Friday?" continues the last topic (a general-knowledge answer is not one to continue)
                    route.intent, route.method = base, "followup"
                route.q_ret = effective
            t_route = time.perf_counter()
            handler = {
                "recall.query": self._recall,
                "task.add": self._task_add,
                "task.list": self._task_list,
                "reminder.set": self._reminder,
                "digest.today": self._digest,
                "general.knowledge": self._general,
                "meta.history": self._history,
                "unsupported": self._unsupported,
            }.get(route.intent, self._fact)
            ans = handler(question, route)
        finally:
            self.on_state("thinking", False)
        ans.timings = {"route_ms": round((t_route - t0) * 1000, 1), **ans.timings,
                       "total_ms": round((time.perf_counter() - t0) * 1000, 1)}  # fmt: skip
        ans.voice_text = spoken_form(ans.text)
        ans.abstained = ans.abstained or ans.gate in ("no_memory", "fact_unavailable") or ans.intent == "unsupported"
        self._log_turn(question, ans)
        log.info("answer [%s/%s, %s] %s", ans.intent, ans.route, ans.gate or "-", ans.text)
        return ans

    # -- conversation (section 11: turns, R7) --------------------------------------------

    def _recent_turns(self) -> list[dict]:
        """The last few exchanges; empty after five minutes of silence."""
        return self.memory.store.turns(BUFFER_TURNS, since=self.clock().timestamp() - CONVERSATION_S)

    @staticmethod
    def _is_followup(q: str, prev: list[dict]) -> bool:
        """A short reply that leans on the last exchange ("and on Friday?", "what about Rohan?")."""
        return bool(prev) and bool(_FOLLOWUP.search(q))

    @staticmethod
    def _resolve_followup(q: str, prev: list[dict]) -> str:
        """A follow-up is searched together with the question before it."""
        return f"{q} {prev[-1]['question']}" if Reasoner._is_followup(q, prev) else q

    def _conversation(self, prev: list[dict]) -> str:
        if not prev:
            return ""
        lines = "\n".join(f"You asked: {t['question']}\nKin said: {t['answer'][:160]}" for t in prev)
        return f"Earlier in this conversation:\n{lines}\n"

    def _log_turn(self, question: str, ans: Answer) -> None:
        """E39: every exchange is kept (filtered, encrypted, seven days)."""
        try:
            self.memory.store.add_turn(
                redact(question).text, ans.intent, redact(ans.text).text, [c.get("id") for c in ans.citations],
                ts=self.clock().timestamp(),
            )  # fmt: skip
        except Exception:  # noqa: BLE001 - a logging failure must not lose the answer
            log.exception("could not log the turn")

    def _history(self, q: str, route) -> Answer:
        turns = self.memory.store.turns(1)
        ans = Answer(q, "", "meta.history", route.method, gate="history")
        if not turns:
            ans.text = "We haven't talked yet."
        elif _REPEAT.search(q):
            ans.text = turns[-1]["answer"]
        else:
            ans.text = f"You asked: {turns[-1]['question']}"
        return ans

    def _general(self, q: str, route) -> Answer:
        """Built-in knowledge, offline, with a note that it may be out of date."""
        ans = Answer(q, "", "general.knowledge", route.method)
        try:
            t = time.perf_counter()
            out = self.llm.complete(load_prompt("general").replace("{{QUESTION}}", q), max_tokens=70, stop=["\n\n"])
            ans.llm_used = True
            ans.timings["llm_ms"] = round((time.perf_counter() - t) * 1000, 1)
        except LLMUnavailable as e:
            log.warning("no model for a general question: %s", e)
            ans.gate = "retrieval_only"
            ans.text = "I can't answer general questions right now: my language model isn't loaded."
            ans.abstained = True
            return ans
        text = out.text.strip()
        if not text or text.lower().startswith("i'm not sure") or text.lower().startswith("i don't"):
            ans.gate, ans.abstained = "unsure", True
            ans.text = "I'm not sure about that, and I can't look it up offline."
            return ans
        ans.gate = "general"
        ans.text = f"{text} That's from general knowledge, so it may be out of date."
        return ans

    # -- recall --------------------------------------------------------------------------

    def _recall(self, q: str, route) -> Answer:
        now = self.clock()
        ans = Answer(q, "", "recall.query", route.method)
        q_ret = getattr(route, "q_ret", None) or q  # a follow-up is searched with the question before it
        if route.also:
            ans.intent = f"recall.query+{route.also}"

        t = time.perf_counter()
        qvec = self.memory.embedder.embed_one(q_ret)
        hits = gate.relevant(self.memory.search(q_ret, k=4, qvec=qvec))
        when = parse_when(q_ret, now)
        events = self.calendar.relevant(" ".join([q_ret] + [h["text"] for h in hits[:2]]), now,
                                        when.at.date() if when else None)  # fmt: skip
        ans.timings["retrieve_ms"] = round((time.perf_counter() - t) * 1000, 1)

        if not hits and not events and not route.also and self._generic(q):
            return self._general(q, route)  # "what is Python?" with nothing remembered about it

        fact_item, fact_note, fact_text = None, "", ""
        if route.also:
            target = when.at if when else self._target_from_memory(hits, events)
            fact_text, ok = self._lookup(q, route.also, ans, self._people(q), target)
            if ok:
                kind = "forecast" if route.also == "fact.weather" else "online"
                fact_item = ctx.Item(kind, "fact", now, fact_text, {"fact": kind})
            else:
                fact_note = fact_text

        if not hits and not events:
            ans.gate = "no_memory"
            ans.text = self._join(gate.NO_MEMORY, fact_text if fact_item else fact_note)
            return ans
        items = ctx.build(hits, events, q) + ([fact_item] if fact_item else [])

        conflict = gate.find_conflict(items)
        if conflict:
            a, b = conflict
            ans.gate = "conflict"
            ans.citations = [_cite(a), _cite(b)]
            ans.text = self._join(
                f'I heard two different things. {self._when(a)}: "{a.text[:140]}" '
                f'But {self._when(b).lower()}: "{b.text[:140]}" Which one is right?',
                fact_text if fact_item else fact_note,
            )
            return ans

        if route.also and fact_item is None:
            # The fact could not be fetched. The local part is read back directly and the
            # missing part explained plainly (flow F5).
            best = gate.best_sentence(items, q)
            if best is None:
                ans.gate = "no_memory"
                ans.text = self._join(gate.NO_MEMORY, fact_note)
                return ans
            sentence, item = best
            ans.gate = "local_part"
            ans.citations = [_cite(item)]
            ans.text = self._join(self._read_back(sentence, item), fact_note)
            return ans

        prev = self._recent_turns()
        prompt = (
            load_prompt("answer")
            .replace("{{EXTRA}}", self._conversation(prev))
            .replace("{{CONTEXT}}", ctx.render(items))
            # The date only when the question is relative to it: a small model otherwise
            # copies today's weekday into answers about other days.
            .replace("{{NOW}}", f"Now is {now.strftime('%A %d %B %Y, %H:%M')}." if _RELATIVE_DAY.search(q) else "")
            .replace("{{QUESTION}}", q)
        )
        try:
            t = time.perf_counter()
            loaded_before = self.llm.loaded
            out = self.llm.complete(prompt, stop=["\n\n"])
            ans.llm_used = True
            ans.timings["llm_ms"] = round((time.perf_counter() - t) * 1000, 1)
            if not loaded_before:
                ans.timings["llm_load_ms"] = round(self.llm.load_ms, 1)
            result = gate.check(out.text, items, q)
        except LLMUnavailable as e:
            log.warning("retrieval-only answer: %s", e)
            result = gate.GateResult(False, "retrieval_only")
            out = None

        if result.ok:
            ans.gate = "grounded"
            ans.citations = [_cite(c) for c in result.cites]
            ans.text = out.text
            return ans
        if result.reason == "declined":
            # The small model often says "nothing" for a plain "what is X" even when a memory
            # mentions X. If retrieval was confident, read the memory back instead.
            # (a calendar entry that matched the question counts as confident too)
            best = gate.best_sentence(items, q) if (self._confident(hits, q) or events) else None
            if best is None:
                ans.gate = "no_memory"
                ans.text = self._join(gate.NO_MEMORY, fact_text if fact_item else fact_note)
                return ans
            sentence, item = best
            ans.gate = "fallback"
            ans.citations = [_cite(item)]
            ans.text = self._join(self._read_back(sentence, item), fact_text if fact_item else "")
            return ans
        # Ungrounded, vague or no model: read back the best remembered sentence (and the fact, verbatim).
        best = gate.best_sentence([i for i in items if i.kind != "fact"] or items, q)
        ans.gate = result.reason if result.reason != "vague" else "fallback"
        if result.reason == "ungrounded":
            log.info("gate rejected %r: unsupported %s", out.text if out else "", result.unsupported)
        if best is None:
            ans.text = self._join(gate.NO_MEMORY, fact_text if fact_item else fact_note)
            return ans
        sentence, item = best
        ans.citations = [_cite(item)] + ([_cite(fact_item)] if fact_item else [])
        ans.text = self._join(self._read_back(sentence, item), fact_text if fact_item else "")
        return ans

    @staticmethod
    def _confident(hits: list[dict], q: str) -> bool:
        """Retrieval is sure enough to override a model that said "nothing": the top memory
        covers most of the question's content words, or matches its meaning strongly."""
        if not hits:
            return False
        top = hits[0]
        words = ctx.content_words(q)
        if words and len(words & ctx.content_words(top.get("text", ""))) / len(words) >= 0.6:
            return True
        return (top.get("vector_similarity") or 0.0) >= 0.55

    @staticmethod
    def _generic(q: str) -> bool:
        return bool(_GENERIC.search(q)) and not _PERSONAL_WORDS.search(q) and not _CURRENT.search(q)

    def _target_from_memory(self, hits: list[dict], events) -> dt.datetime | None:
        """The moment a remembered plan is for ("site visit Thursday at 4 pm"), for the forecast window."""
        for e in events:
            if not e.all_day and e.start > self.clock():
                return e.start
        for h in hits:
            when = parse_when(h["text"], dt.datetime.fromtimestamp(h["started_at"]))
            if when is not None and when.at > self.clock():
                return when.at
        return None

    def _read_back(self, sentence: str, item: ctx.Item) -> str:
        if item.kind == "calendar":
            return f"Your calendar has {sentence[0].lower() + sentence[1:]}."
        return f'On {self._when(item)}, I heard: "{sentence}"'

    def _when(self, item: ctx.Item) -> str:
        return item.when.strftime("%A at %I:%M %p").replace(" 0", " ")

    @staticmethod
    def _join(*parts: str) -> str:
        return " ".join(p for p in parts if p)

    def _people(self, q: str) -> set[str]:
        return {m for m in re.findall(r"\b[A-Z][a-z]{2,}\b", q) if m.lower() not in WEEKDAYS}

    # -- facts ---------------------------------------------------------------------------

    def _known_names(self, people: set[str]) -> list[str]:
        """Names the gateway must keep out of a search: people in the question and in memory."""
        return sorted(people | {p["name"] for p in self.memory.store.people()})

    def _lookup(self, q: str, intent: str, ans: Answer, people: set[str], target: dt.datetime | None = None):
        """One typed request through the gateway. Returns (sentence, ok); the sentence is the
        fact in words, or the plain reason it could not be checked."""
        now = self.clock()
        t = time.perf_counter()
        llm = self.llm if self.llm is not None and self.llm.installed() else None
        try:
            req = build_request(intent, q, now, llm=llm, people=people, when_at=target)
        except LLMUnavailable:
            req = build_request(intent, q, now, llm=None, people=people, when_at=target)
        ans.fact_request = req.payload()
        self.on_state("fetching", True)
        try:
            resp = self.facts.fetch(req, names=self._known_names(people))
        finally:
            self.on_state("fetching", False)
        ans.timings["fetch_ms"] = round((time.perf_counter() - t) * 1000, 1)
        led = resp.ledger
        ans.fact = {"kind": resp.kind, "ok": resp.ok, "error": resp.error, "host": led.host if led else None,
                    "bytes_out": led.bytes_out if led else 0, "status": led.status if led else "not sent"}  # fmt: skip
        if resp.ok:
            when = parse_when(q, now)
            has_time = when.has_time if when is not None else True  # a time remembered from speech is a time
            return describe_fact(resp, now, target, has_time), True
        return unavailable_text(resp), False

    def _fact(self, q: str, route) -> Answer:
        if route.intent == "fact.search":
            # Memory first: "When does the Hyderabad office open?" may have been said to the user.
            hits = self.memory.search(q, k=3, qvec=self.memory.embedder.embed_one(q))
            if any(h["keyword_match"] for h in gate.relevant(hits)):
                route.also, route.method = "fact.search", "mixed"
                return self._recall(q, route)
        ans = Answer(q, "", route.intent, route.method)
        when = parse_when(q, self.clock())
        text, ok = self._lookup(q, route.intent, ans, people=self._people(q), target=when.at if when else None)
        ans.text = text
        if ok:
            kind = {"fact.weather": "forecast", "fact.news": "news", "fact.search": "online"}[route.intent]
            ans.gate = "fact"
            ans.citations = [{"id": kind, "kind": "fact", "when": time.time(), "text": text[:200]}]
        else:
            ans.gate = "fact_unavailable"
        return ans

    # -- tasks and reminders ---------------------------------------------------------------

    def _due(self, text: str, now: dt.datetime) -> tuple[dt.datetime | None, bool, str]:
        """(due, has_time, title) for a task; asks the model only for vague wording."""
        when = parse_when(text, now)
        title = _clean_title(when.strip_from(text) if when else text)
        if when is not None:
            return when.at, when.has_time, title
        if not _VAGUE_WHEN.search(text) or self.llm is None:
            return None, False, title
        try:
            import json

            out = self.llm.complete(
                load_prompt("when").replace("{{NOW}}", now.strftime("%A %d %B %Y, %H:%M")).replace("{{TASK}}", text),
                prefill='{"day": "',
                grammar=load_grammar("when"),
                max_tokens=16,
            )
            v = json.loads('{"day": "' + out.text)
        except (LLMUnavailable, ValueError):
            return None, False, title
        if v["day"] == "none":
            return None, False, title
        day = now.date()
        if v["day"] == "tomorrow":
            day += dt.timedelta(days=1)
        elif v["day"] in WEEKDAYS:
            day += dt.timedelta(days=(WEEKDAYS.index(v["day"]) - now.weekday()) % 7)
        has_time = v["time"] != "none"
        hh, mm = (int(x) for x in (v["time"] if has_time else "18:00").split(":"))
        if not (0 <= hh < 24 and 0 <= mm < 60):
            return None, False, title
        return dt.datetime.combine(day, dt.time(hh, mm)), has_time, title

    def _task_add(self, q: str, route) -> Answer:
        now = self.clock()
        due, has_time, title = self._due(q, now)
        ans = Answer(q, "", "task.add", route.method, llm_used=False)
        if not title:
            ans.text = "What should I add?"
            return ans
        task_id = self.memory.store.add_task(title, source="command", due_at=due.timestamp() if due else None,
                                             due_has_time=has_time, now=now.timestamp())  # fmt: skip
        ans.task = self.memory.store.get_task(task_id)
        ans.text = f"Added: {title[0].lower() + title[1:]}" + (f", {describe(due, now, has_time)}." if due else ".")
        return ans

    def _task_list(self, q: str, route) -> Answer:
        now = self.clock()
        tasks = [t for t in self.memory.store.tasks(("open",))]
        ans = Answer(q, "", "task.list", route.method)
        if not tasks:
            ans.text = "Your list is empty."
            return ans
        parts = []
        for t in tasks[:5]:
            due = f" {describe(dt.datetime.fromtimestamp(t['due_at']), now, t['due_has_time'])}" if t["due_at"] else ""
            parts.append(t["title"][0].lower() + t["title"][1:] + due)
        more = f", and {len(tasks) - 5} more" if len(tasks) > 5 else ""
        n = len(tasks)
        ans.text = f"You have {n} thing{'s' if n != 1 else ''} to do: " + "; ".join(parts) + more + "."
        return ans

    def _reminder(self, q: str, route) -> Answer:
        now = self.clock()
        ans = Answer(q, "", "reminder.set", route.method)
        when = parse_when(q, now)
        if when is None:
            ans.text = "When should I remind you?"
            return ans
        text = _REMINDER_WORDS.sub(" ", when.strip_from(q))
        text = " ".join(text.split()).strip(" ,.") or "the time you asked for"
        text = re.sub(r"^(to|about|that)\s+", "", text, flags=re.I)
        rid = self.memory.store.add_reminder(text, when.at.timestamp(), now=now.timestamp())
        ans.reminder = {"id": rid, "text": text, "fire_at": when.at.timestamp()}
        ans.text = f"OK, I'll remind you to {text} {describe(when.at, now)}."
        return ans

    def _digest(self, q: str, route) -> Answer:
        now = self.clock()
        start = dt.datetime.combine(now.date(), dt.time()).timestamp()
        end = start + 86400
        created = self.memory.store.tasks_created_between(start, end)
        heard = [t for t in created if t["source"] == "heard" and t["status"] != "dismissed"]
        added = [t for t in created if t["source"] in ("command", "ui", "note")]
        due_today = [t for t in self.memory.store.tasks(("open",)) if t["due_at"] and start <= t["due_at"] < end]
        n_eps = self.memory.store.episodes_between(start, end)
        ans = Answer(q, "", "digest.today", route.method)
        parts = []
        if heard:
            parts.append(
                "You said you would " + "; ".join(t["title"][0].lower() + t["title"][1:] for t in heard[:5]) + "."
            )
        if added:
            parts.append(f"You added {len(added)} task{'s' if len(added) != 1 else ''}.")
        if due_today:
            parts.append("Due today: " + "; ".join(t["title"] for t in due_today[:5]) + ".")
        if not parts:
            parts.append("I didn't hear you commit to anything today.")
        parts.append(f"{n_eps} conversation{'s' if n_eps != 1 else ''} saved today.")
        ans.text = " ".join(parts)
        return ans

    def _unsupported(self, q: str, route) -> Answer:
        ans = Answer(q, "I can't do that. Do you want me to save it as a note?", "unsupported", route.method)
        ans.offer_note = True
        return ans
