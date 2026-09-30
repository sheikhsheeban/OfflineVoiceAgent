"""The real language model (Gemma 3 1B on llama.cpp): grammars, lifecycle, guards, real answers."""

import datetime as dt
import json
import time
from collections import namedtuple

import pytest

from kin.reasoning.llm import LLMUnavailable, LocalLLM, load_grammar, load_prompt

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def llm():
    m = LocalLLM(threads=4)
    yield m
    m.unload()


def test_route_grammar_only_allows_listed_intents(llm):
    names = ["recall.query", "fact.weather", "unsupported"]
    grammar = load_grammar("route").replace("{{INTENTS}}", " | ".join(f'"{n}"' for n in names))
    prompt = load_prompt("route").replace("{{INTENTS}}", "\n".join(f"- {n}" for n in names))
    for q in ["Will it rain on Thursday?", "What did Priya say?", "Write a poem"]:
        out = llm.complete(prompt.replace("{{QUESTION}}", q), prefill='{"intent": "', grammar=grammar, max_tokens=12)
        assert json.loads('{"intent": "' + out.text)["intent"] in names


@pytest.mark.parametrize(
    "kind,prefill",
    [
        ("weather", '{"kind": "weather", "hours": '),
        ("search", '{"kind": "search", "q": "'),
        ("news", '{"kind": "news", "topic": "'),
    ],
)
def test_fact_grammar_yields_a_valid_request(llm, kind, prefill):
    from kin.reasoning.facts import FactRequest

    out = llm.complete(
        load_prompt("fact").replace("{{QUESTION}}", "Is the Wardha road closed because of rain? Any local news?"),
        prefill=prefill,
        grammar=load_grammar("fact").replace("{{KIND}}", kind),
        max_tokens=40,
    )
    FactRequest(**json.loads(prefill + out.text))  # validates against the sheet-4 schema


def test_when_grammar(llm):
    out = llm.complete(
        load_prompt("when")
        .replace("{{NOW}}", "Wednesday 30 September 2026, 10:40")
        .replace("{{TASK}}", "call the landlord after work"),
        prefill='{"day": "',
        grammar=load_grammar("when"),
        max_tokens=16,
    )
    v = json.loads('{"day": "' + out.text)
    assert v["day"] in ("today", "none", "wednesday") and (v["time"] == "none" or ":" in v["time"])


def test_idle_release(llm):
    llm.complete("Say ok", max_tokens=2)
    assert llm.loaded
    assert not llm.maybe_unload(time.monotonic())  # just used
    assert llm.maybe_unload(time.monotonic() + llm.idle_unload_s + 1)
    assert not llm.loaded


def test_low_memory_refuses_to_load(monkeypatch):
    import kin.reasoning.llm as mod

    VM = namedtuple("VM", "available")
    monkeypatch.setattr(mod.psutil, "virtual_memory", lambda: VM(900 * 2**20))
    with pytest.raises(LLMUnavailable, match="900 MB free"):
        LocalLLM(threads=2).load()


def test_battery_halves_threads(monkeypatch):
    import kin.reasoning.llm as mod

    Bat = namedtuple("Bat", "percent power_plugged")
    monkeypatch.setattr(mod.psutil, "sensors_battery", lambda: Bat(20, False))
    assert mod.battery_low()
    monkeypatch.setattr(mod.psutil, "sensors_battery", lambda: Bat(20, True))
    assert not mod.battery_low()


def test_missing_model_file(tmp_path):
    with pytest.raises(LLMUnavailable, match="missing"):
        LocalLLM(threads=2, path=tmp_path / "none.gguf").load()


def test_real_answers_on_demo_memory(tmp_path, llm):
    """Whatever the model says, what reaches the user is grounded or honestly declined."""
    from kin.memory.embed import Embedder
    from kin.memory.service import Memory
    from kin.memory.store import Store
    from kin.reasoning.calendar import Calendar
    from kin.reasoning.engine import Reasoner
    from tests.conftest import DATA

    mem = Memory(Store(tmp_path / "k.db", "ab" * 32), Embedder())
    for h, text in [(18, "Rohan said the site visit is Thursday at 4 pm at Butibori."),
                    (19, "Priya will send the budget slides tomorrow morning.")]:  # fmt: skip
        t = dt.datetime(2026, 9, 29, h, 0).timestamp()
        mem.add(t, t + 3, text, is_command=False, redactions=0)
        mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    r = Reasoner(mem, llm, Calendar(DATA / "demo_calendar.ics"), clock=lambda: dt.datetime(2026, 9, 30, 10, 40))
    try:
        a = r.handle("When is the site visit Rohan mentioned?")
        assert a.gate in ("grounded", "ungrounded") and "Thursday" in a.text and a.citations
        a = r.handle("What did Priya say about the budget?")
        assert "slides" in a.text and a.citations[0]["id"] == "e-2"
        a = r.handle("What is my passport number?")
        assert a.text == "I don't have anything about that."
        t0 = time.perf_counter()
        r.handle("Where is the site visit?")
        assert time.perf_counter() - t0 < 5.0  # warm model on a 2019 laptop CPU
    finally:
        mem.store.close()
