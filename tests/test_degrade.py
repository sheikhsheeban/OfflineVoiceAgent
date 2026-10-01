"""Sheet 8: graceful degradation, one test per row (`test_degrade_<situation>`).

The no-network row is in test_w4_ids.py, the clip-unplugged row too, and the crashed-process row is
`test_CL8_*` there."""

import pytest

pytestmark = pytest.mark.models


def test_degrade_nothing_in_memory_says_so_without_calling_the_model(make):
    from tests.engine_fixtures import ScriptedLLM

    llm = ScriptedLLM("must not be called")
    a = make(llm)[0].handle("What is my passport number?")
    assert a.text == "I don't have anything about that." and a.gate == "no_memory" and not a.llm_used
    assert [k for k, _ in llm.calls if k == "answer"] == []


def test_degrade_conflicting_memories_states_both_and_asks(make):
    from tests.engine_fixtures import ScriptedLLM, ts

    r, mem = make(ScriptedLLM("Thursday."))
    t = ts(30, 10, 0)
    mem.add(t, t + 3, "Rohan said the site visit moved to Friday at 3 pm.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    a = r.handle("When is Rohan's site visit?")
    assert a.gate == "conflict" and "Thursday" in a.text and "Friday" in a.text and a.text.endswith("Which one is right?")


def test_degrade_beyond_the_model_declines_plainly_and_offers_a_note(make):
    from tests.engine_fixtures import ScriptedLLM

    a = make(ScriptedLLM())[0].handle("Draft my tax return")
    assert a.intent == "unsupported" and a.offer_note and a.abstained and "can't do that" in a.text


def test_degrade_low_memory_unloads_the_model_and_reads_the_best_episode(make, monkeypatch):
    import psutil

    from kin.reasoning.calendar import Calendar
    from kin.reasoning.engine import Reasoner
    from kin.reasoning.llm import LocalLLM
    from tests.engine_fixtures import NOW

    _, mem = make(None)
    llm = LocalLLM(threads=2)
    if not llm.installed():
        pytest.skip("language model not fetched")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"available": 900 * 2**20})())
    a = Reasoner(mem, llm, Calendar(None), clock=lambda: NOW).handle("Where is the site visit?")
    assert a.gate == "retrieval_only" and "Butibori" in a.text and not llm.loaded


def test_degrade_battery_low_halves_the_model_threads(monkeypatch):
    import kin.reasoning.llm as llm_mod

    monkeypatch.setattr(
        llm_mod.psutil, "sensors_battery", lambda: type("B", (), {"power_plugged": False, "percent": 20.0})()
    )
    assert llm_mod.battery_low()
    monkeypatch.setattr(
        llm_mod.psutil, "sensors_battery", lambda: type("B", (), {"power_plugged": True, "percent": 20.0})()
    )
    assert not llm_mod.battery_low()
