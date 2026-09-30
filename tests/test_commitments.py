import datetime as dt

import pytest

from kin.reasoning.commitments import extract

NOW = dt.datetime(2026, 9, 30, 10, 40)


@pytest.mark.parametrize(
    "text,title,due",
    [
        ("I will send it by Friday.", "Send it", "2026-10-02 18:00"),
        (
            "Okay, I'll call Rohan tomorrow morning about the drawings.",
            "Call Rohan about the drawings",
            "2026-10-01 09:00",
        ),
        ("We need to book the hall before Monday.", "Book the hall", "2026-10-05 18:00"),
        ("I have to pay the electricity bill", "Pay the electricity bill", None),
        ("Sure. I'm going to email Kiran the invoice tonight.", "Email Kiran the invoice", "2026-09-30 20:00"),
    ],
)
def test_commitments_become_candidates(text, title, due):
    (c,) = extract(text, NOW)
    assert c.title == title
    assert (c.due_at.strftime("%Y-%m-%d %H:%M") if c.due_at else None) == due
    assert c.sentence in text


@pytest.mark.parametrize(
    "text",
    [
        "Priya will send the budget slides tomorrow.",  # someone else's promise
        "Will you send it?",  # a question
        "I'll let you know.",  # not an action
        "I think I will be there.",
        "I'll go now.",
        "The weather is nice today.",
    ],
)
def test_non_commitments_are_ignored(text):
    assert extract(text, NOW) == []


def test_several_in_one_utterance():
    found = extract("I'll send the drawings on Monday. And I will call the contractor.", NOW)
    assert [c.title for c in found] == ["Send the drawings", "Call the contractor"]
