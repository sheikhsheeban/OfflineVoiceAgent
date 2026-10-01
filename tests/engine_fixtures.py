"""Shared engine test fixtures (registered as a pytest plugin in conftest.py):
a scripted language model and a small memory with a real embedder and the demo calendar."""

import datetime as dt

import pytest

from kin.memory.store import Store
from kin.reasoning.llm import Completion, LLMUnavailable
from tests.conftest import DATA

NOW = dt.datetime(2026, 9, 30, 10, 40)


def ts(d, h, m):
    return dt.datetime(2026, 9, d, h, m).timestamp()


class ScriptedLLM:
    """Answers from a script keyed on what is being asked; records every call."""

    def __init__(self, answer="", unavailable=False):
        self.answer = answer
        self.unavailable = unavailable
        self.calls: list[tuple[str, str]] = []
        self.loaded = False
        self.load_ms = 5.0

    def installed(self):
        return True

    def complete(self, user, *, prefill="", grammar=None, max_tokens=None, stop=None):
        if self.unavailable:
            raise LLMUnavailable("only 900 MB free")
        self.loaded = True
        kind = (
            "route"
            if "intent" in prefill
            else "fact"
            if '"kind"' in prefill
            else "when"
            if '"day"' in prefill
            else "answer"
        )
        self.calls.append((kind, user))
        text = {"route": 'recall.query"}', "fact": "24}", "when": 'today", "time": "18:00"}'}.get(kind, self.answer)
        return Completion(text, 100, 10, 5.0)

    def warm(self, prefix):
        self.loaded = True


@pytest.fixture(scope="module")
def embedder():
    from kin.memory.embed import Embedder

    return Embedder()


@pytest.fixture
def make(tmp_path, embedder):
    from kin.memory.service import Memory
    from kin.reasoning.calendar import Calendar
    from kin.reasoning.engine import Reasoner

    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), embedder)
    for t, text in [
        (ts(29, 18, 12), "Rohan said the site visit is Thursday at 4 pm at Butibori."),
        (ts(29, 19, 0), "Priya will send the budget slides tomorrow morning."),
        (ts(30, 9, 5), "I had dal and rice for lunch with Aditya."),
        (ts(30, 9, 30), "Meera said the Hyderabad office opens in March."),
    ]:
        mem.add(t, t + 4, text, is_command=False, redactions=0)
        mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)

    def _make(llm):
        return Reasoner(mem, llm, Calendar(DATA / "demo_calendar.ics"), clock=lambda: NOW), mem

    yield _make
    mem.store.close()


