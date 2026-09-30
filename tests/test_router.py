"""Intent routing with the real embedder (no language model: close calls use the best allowed centroid)."""

import pytest

pytestmark = pytest.mark.models

HELD_OUT = [  # phrasings that are not in spec/intents.toml
    ("What did Suresh say about the invoice?", "recall.query"),
    ("when is the dentist appointment my wife mentioned", "recall.query"),
    ("Remind me to call the landlord after work", "task.add"),
    ("I have to buy milk, put it on my list", "task.add"),
    ("add renew insurance to my tasks", "task.add"),
    ("read me my to-do list", "task.list"),
    ("what do I need to do today", "task.list"),
    ("In 20 minutes, tell me to leave", "reminder.set"),
    ("remind me in 5 minutes to check the rice", "reminder.set"),
    ("at 7 pm remind me to call Anjali", "reminder.set"),
    ("What did I commit to today?", "digest.today"),
    ("what did I say I would do today", "digest.today"),
    ("is it going to rain this evening", "fact.weather"),
    ("how cold will it be tomorrow", "fact.weather"),
    ("is the highway to Wardha open", "fact.search"),
    ("any headlines about Nagpur", "fact.news"),
    ("what is the latest cricket news", "fact.news"),
    ("Draft my tax return", "unsupported"),
    ("write me a birthday song", "unsupported"),
]


@pytest.fixture(scope="module")
def router():
    from kin.memory.embed import Embedder
    from kin.reasoning.router import Router

    return Router(Embedder())


def test_held_out_accuracy(router):
    wrong = [(q, exp, router.route(q).intent) for q, exp in HELD_OUT if router.route(q).intent != exp]
    assert len(wrong) <= 1, wrong


def test_memory_plus_fact_is_mixed(router):
    r = router.route("Should I take an umbrella to the site visit Rohan mentioned?")
    assert (r.intent, r.also, r.method) == ("recall.query", "fact.weather", "mixed")


@pytest.mark.parametrize("q", ["What is my passport number?", "what is my bank balance", "where did I park my car"])
def test_personal_questions_never_go_online(router, q):
    assert router.route(q).intent not in ("fact.search", "fact.news")


def test_tasks_need_explicit_wording(router):
    # Close calls never create tasks or reminders.
    for q in ["Draft my tax return", "turn off the lights", "the plumber is coming", "What should I bring?"]:
        assert router.route(q).intent not in ("task.add", "reminder.set"), q


def test_memory_probe_wins_close_calls():
    from kin.memory.embed import Embedder
    from kin.reasoning.router import Router

    r = Router(Embedder(), memory_probe=lambda q: True)
    assert r.route("What should I bring to the site visit?").intent == "recall.query"
