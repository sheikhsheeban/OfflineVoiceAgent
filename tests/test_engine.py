"""The reasoning engine with a scripted language model: every gate path, every intent."""

import datetime as dt

import pytest

from tests.engine_fixtures import NOW, ScriptedLLM, ts  # noqa: F401

pytestmark = pytest.mark.models


def test_grounded_recall_cites_the_episode(make):
    llm = ScriptedLLM("Priya will send the budget slides tomorrow morning.")
    r, _ = make(llm)
    a = r.handle("What did Priya say about the budget?")
    assert (a.intent, a.gate) == ("recall.query", "grounded")
    assert a.text == "Priya will send the budget slides tomorrow morning."
    assert a.citations[0]["id"] == "e-2" and a.llm_used
    prompt = [u for k, u in llm.calls if k == "answer"][0]
    assert "[e-2] Heard" in prompt and "Priya will send" in prompt
    assert "Now is" not in prompt  # no date in the prompt unless the question is relative to it


def test_ungrounded_answer_is_replaced_by_what_was_heard(make):
    r, _ = make(ScriptedLLM("The site visit is on Wednesday at 4 pm."))
    a = r.handle("When is the site visit Rohan mentioned?")
    assert a.gate == "ungrounded"
    assert "Rohan said the site visit is Thursday at 4 pm at Butibori." in a.text
    assert a.citations[0]["id"] == "e-1"


def test_nothing_relevant_skips_the_model(make):
    llm = ScriptedLLM("should not be called")
    r, _ = make(llm)
    a = r.handle("What is my passport number?")
    assert a.text == "I don't have anything about that." and a.gate == "no_memory"
    assert [k for k, _ in llm.calls if k == "answer"] == []


def test_model_declining_means_no_memory(make):
    r, _ = make(ScriptedLLM("I don't have anything about that."))
    assert r.handle("What did Aditya say about the hall booking?").gate == "no_memory"


def test_low_memory_answers_from_retrieval_only(make):
    r, _ = make(ScriptedLLM(unavailable=True))
    a = r.handle("Where is the site visit?")
    assert a.gate == "retrieval_only" and "Butibori" in a.text and a.citations


def test_conflicting_memories_are_reported(make):
    r, mem = make(ScriptedLLM("Thursday."))
    t = ts(30, 10, 0)
    mem.add(t, t + 3, "Rohan said the site visit moved to Friday at 3 pm.", is_command=False, redactions=0)
    mem.store.close_all()
    for e in mem.store.pending_embeddings():
        mem.embed_episode(e)
    a = r.handle("When is Rohan's site visit?")
    assert a.gate == "conflict" and "Which one is right?" in a.text
    assert {c["id"] for c in a.citations} == {"e-1", "e-5"}


def test_umbrella_question_answers_the_local_part(make):
    r, _ = make(ScriptedLLM("unused"))
    a = r.handle("Should I take an umbrella to the site visit Rohan mentioned?")
    assert a.intent == "recall.query+fact.weather" and a.gate == "local_part"
    assert "Thursday at 4 pm at Butibori" in a.text and a.text.endswith("I can't check the weather right now.")
    assert a.fact_request == {"kind": "weather", "lat": 21.1, "lon": 79.1, "hours": 48}  # Thursday 4 pm is ~29 h away
    assert "Rohan" not in str(a.fact_request)  # sheet 4: no names, no question text


def test_search_checks_memory_first(make):
    a = make(ScriptedLLM())[0].handle("When does the Hyderabad office open?")
    assert "March" in a.text and "can't check that online" in a.text


def test_fact_without_gateway(make):
    a = make(ScriptedLLM())[0].handle("Will it rain tomorrow?")
    assert (a.intent, a.text) == ("fact.weather", "I can't check the weather right now.")
    assert a.fact_request["kind"] == "weather"


def test_calendar_answers(make):
    r, _ = make(ScriptedLLM("Monday 5 October at 11:00 AM at Sitabuldi."))
    a = r.handle("When is my dentist appointment?")
    assert a.gate == "grounded" and a.citations[0]["kind"] == "calendar"


def test_tasks_and_reminders(make):
    llm = ScriptedLLM()
    r, mem = make(llm)
    a = r.handle("Remind me to call the landlord after work")
    assert a.intent == "task.add" and a.text == "Added: call the landlord, at 6:00 pm."
    assert a.task["due_at"] == dt.datetime(2026, 9, 30, 18, 0).timestamp()
    a = r.handle("In 20 minutes, tell me to leave")
    assert a.text == "OK, I'll remind you to leave at 11:00 am."
    assert mem.store.reminders()[0]["fire_at"] == dt.datetime(2026, 9, 30, 11, 0).timestamp()
    assert r.handle("What's on my list?").text == "You have 1 thing to do: call the landlord at 6:00 pm."
    assert llm.calls == []  # all rules, no model


def test_unclear_due_time_asks_the_model(make):
    llm = ScriptedLLM()
    r, mem = make(llm)
    a = r.handle("Add renew the insurance after the meeting to my list")
    assert [k for k, _ in llm.calls] == ["when"]
    assert a.task["due_at"] == dt.datetime(2026, 9, 30, 18, 0).timestamp()


def test_digest_uses_stored_tasks_not_the_model(make):
    llm = ScriptedLLM()
    r, mem = make(llm)
    mem.store.add_task("Send the drawings", source="heard", status="candidate", now=NOW.timestamp())
    mem.store.add_task("Pay the bill", source="command", due_at=NOW.replace(hour=18).timestamp(), now=NOW.timestamp())
    a = r.handle("What did I commit to today?")
    assert "You said you would send the drawings." in a.text and "Due today: Pay the bill." in a.text
    assert "2 conversations saved today." in a.text and llm.calls == []


def test_unsupported_offers_a_note(make):
    a = make(ScriptedLLM())[0].handle("Draft my tax return")
    assert a.intent == "unsupported" and a.offer_note and "save it as a note" in a.text
