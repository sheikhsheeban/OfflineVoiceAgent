import datetime as dt

from kin.reasoning import gate
from kin.reasoning.context import Item

WHEN = dt.datetime(2026, 9, 29, 18, 12)
ROHAN = Item("e-7", "episode", WHEN, "Rohan said the site visit is Thursday at 4 pm at Butibori.")
PRIYA = Item("e-9", "episode", WHEN, "Priya will send the budget slides tomorrow morning.")
CAL = Item("cal-1", "calendar", WHEN, "Dentist at Sitabuldi, Monday 5 October at 11:00 AM until 11:30 AM")
Q = "When is the site visit?"


def test_grounded_answer_cites_its_source():
    r = gate.check("Thursday at 4 pm at Butibori.", [ROHAN, PRIYA], Q)
    assert r.ok and [c.id for c in r.cites] == ["e-7"]


def test_number_words_and_ordinals_match_digits():
    assert gate.check("It's on Thursday at four.", [ROHAN], Q).ok
    assert gate.check("Monday the 5th of October at 11.", [CAL], "When is the dentist?").ok


def test_invented_day_is_rejected():
    r = gate.check("Wednesday at 4 pm.", [ROHAN], Q)
    assert not r.ok and r.reason == "ungrounded" and r.unsupported == ["wednesday"]


def test_invented_name_or_number_is_rejected():
    assert gate.check("Rohan said Thursday at 5 pm.", [ROHAN], Q).unsupported == ["5"]
    assert gate.check("Meera said Thursday.", [ROHAN], Q).unsupported == ["meera"]


def test_recording_date_is_not_evidence():
    # The episode was recorded in September; the answer's month must come from content.
    assert not gate.check("The dentist is on 5 September.", [ROHAN, CAL], "When is the dentist?").ok


def test_decline_and_vague_answers():
    assert gate.check("I don't have anything about that.", [ROHAN], Q).reason == "declined"
    assert gate.check("It is soon.", [ROHAN], Q).reason == "vague"
    assert gate.check("Butibori.", [ROHAN], "Where is the site visit?").ok  # a single name is checkable


def test_retrieval_threshold():
    hits = [
        {"id": 1, "vector_similarity": 0.55, "keyword_match": False},
        {"id": 2, "vector_similarity": 0.18, "keyword_match": True},
        {"id": 3, "vector_similarity": 0.18, "keyword_match": False},
        {"id": 4, "vector_similarity": 0.07, "keyword_match": True},
    ]
    assert [h["id"] for h in gate.relevant(hits)] == [1, 2]


def test_conflicting_memories_are_found():
    later = Item("e-12", "episode", WHEN + dt.timedelta(days=1), "Rohan said the site visit moved to Friday at 3 pm.")
    assert gate.find_conflict([ROHAN, later]) == (ROHAN, later)
    assert gate.find_conflict([ROHAN, PRIYA]) is None  # different subjects
    same = Item("e-13", "episode", WHEN, "Rohan confirmed Thursday at 4 again.")
    assert gate.find_conflict([ROHAN, same]) is None  # agreeing


def test_best_sentence_for_retrieval_only_mode():
    text, item = gate.best_sentence([PRIYA, ROHAN], "where is the site visit")
    assert item is ROHAN and "Butibori" in text
