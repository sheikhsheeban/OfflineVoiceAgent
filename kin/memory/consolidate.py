"""M3 consolidator: overnight housekeeping (sheet 6, flow F7).

Runs only on AC power, after 20 minutes of user idle, at night; or on demand
(`kin consolidate`, the Memory tab). It
  1. deletes transcripts older than 7 days (turns too);
  2. merges duplicate facts, keeping every source;
  3. writes one digest per day that has conversations, with the language model;
     the digest is checked like an answer (every name, day and number in it must
     come from that day's notes) and replaced by a plain extract if it is not;
  4. lets facts, people, digests and ledger rows go after a year.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time

from kin.contracts import ContextEvent
from kin.reasoning import gate
from kin.reasoning.llm import LLMUnavailable, load_prompt

log = logging.getLogger("kin.consolidate")

IDLE_REQUIRED_S = 20 * 60
NIGHT_HOURS = (22, 6)  # from 22:00 to 06:00
MIN_GAP_S = 20 * 3600
YEAR_S = 365 * 86400
MAX_NOTES_CHARS = 1400


def is_night(now: float) -> bool:
    h = dt.datetime.fromtimestamp(now).hour
    return h >= NIGHT_HOURS[0] or h < NIGHT_HOURS[1]


def _day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(t))


def _day_start(day: str) -> float:
    return dt.datetime.strptime(day, "%Y-%m-%d").timestamp()


class Consolidator:
    def __init__(self, memory, llm=None):
        self.memory = memory
        self.llm = llm
        self.last_run = 0.0

    def due(self, ctx: ContextEvent, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return ctx.on_ac and ctx.idle_s >= IDLE_REQUIRED_S and is_night(now) and now - self.last_run >= MIN_GAP_S

    # -- digests ------------------------------------------------------------------------

    @staticmethod
    def _notes(episodes: list[dict]) -> str:
        lines = [f"- {time.strftime('%H:%M', time.localtime(e['started_at']))} {e['text'].strip()}" for e in episodes]
        out, size = [], 0
        for line in lines:
            if size + len(line) > MAX_NOTES_CHARS:
                out.append(line[: max(0, MAX_NOTES_CHARS - size)])
                break
            out.append(line)
            size += len(line) + 1
        return "\n".join(out)

    @staticmethod
    def _extract(episodes: list[dict]) -> str:
        n = len(episodes)
        firsts = []
        for e in episodes:
            s = re.split(r"(?<=[.?!])\s+", e["text"].strip())[0]
            if s and s not in firsts:
                firsts.append(s)
        head = f"{n} conversation{'s' if n != 1 else ''}."
        return head + (" " + " ".join(firsts[:3]) if firsts else "")

    def digest_day(self, day: str) -> dict | None:
        t0 = _day_start(day)
        episodes = [e for e in self.memory.store.episodes_on(t0, t0 + 86400) if e["text"].strip()]
        if not episodes:
            return None
        notes = self._notes(episodes)
        text, source = None, "extract"
        if self.llm is not None:
            try:
                out = self.llm.complete(load_prompt("digest").replace("{{NOTES}}", notes), max_tokens=90, stop=["\n\n"])
                unsupported = [t for t in gate.key_terms(out.text) if not gate._contains(gate._norm(notes), t)]
                if out.text and not unsupported:
                    text, source = out.text, "model"
                else:
                    log.info("digest for %s rejected by the gate: %s", day, unsupported or "empty")
            except LLMUnavailable as e:
                log.info("digest without the model: %s", e)
        if text is None:
            text = self._extract(episodes)
        self.memory.store.set_digest(day, text, len(episodes))
        return {"day": day, "text": text, "n_episodes": len(episodes), "source": source}

    # -- the nightly run ---------------------------------------------------------------------

    def run(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        self.last_run = now
        store = self.memory.store
        result = {"expired_episodes": self.memory.purge_expired(now)}
        result["merged_facts"] = store.merge_duplicate_facts()
        today = _day(now)
        days = {_day(e["started_at"]) for e in store.timeline(limit=500)}
        digests = []
        for day in sorted(days):
            if day == today or store.get_digest(day) is None:  # today's is rewritten as the day grows
                d = self.digest_day(day)
                if d:
                    digests.append(d)
        result["digests"] = digests
        result.update({f"expired_{k}": v for k, v in store.expire_derived(now - YEAR_S).items()})
        log.info(
            "consolidated: %d digest(s), %d fact(s) merged, %d episode(s) expired",
            len(digests), result["merged_facts"], result["expired_episodes"],
        )  # fmt: skip
        return result
