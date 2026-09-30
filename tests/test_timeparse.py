import datetime as dt

import pytest

from kin.reasoning.timeparse import describe, parse_when

NOW = dt.datetime(2026, 9, 30, 10, 40)  # a Wednesday morning


@pytest.mark.parametrize(
    "text,expected,has_time,rest",
    [
        ("In 20 minutes, tell me to leave", "2026-09-30 11:00", True, "tell me to leave"),
        ("Remind me at 5 pm to call mom", "2026-09-30 17:00", True, "Remind me to call mom"),
        ("Tell me in an hour to check the oven", "2026-09-30 11:40", True, "Tell me to check the oven"),
        ("Ping me in half an hour", "2026-09-30 11:10", True, "Ping me"),
        ("At 6, remind me to take my medicine", "2026-09-30 18:00", True, "remind me to take my medicine"),
        ("I will send it by Friday", "2026-10-02 18:00", False, "I will send it"),
        ("call the landlord after work", "2026-09-30 18:00", True, "call the landlord"),
        ("pay the electricity bill tomorrow morning", "2026-10-01 09:00", True, "pay the electricity bill"),
        ("Remind me to call Rohan next week", "2026-10-05 18:00", False, "Remind me to call Rohan"),
        ("book the cab for 6 am on Monday", "2026-10-05 06:00", True, "book the cab"),
        ("dentist on 12 October at 11:30", "2026-10-12 11:30", True, "dentist"),
        ("next Wednesday", "2026-10-07 18:00", False, ""),
        ("in two days", "2026-10-02 10:40", True, ""),
        ("at 9 tomorrow", "2026-10-01 09:00", True, ""),
        ("call Rohan tomorrow morning about the drawings", "2026-10-01 09:00", True, "call Rohan about the drawings"),
    ],
)
def test_parse_when(text, expected, has_time, rest):
    w = parse_when(text, NOW)
    assert w is not None
    assert w.at.strftime("%Y-%m-%d %H:%M") == expected
    assert w.has_time is has_time
    assert w.strip_from(text) == rest


@pytest.mark.parametrize(
    "text",
    ["We need 4 chairs for the second meeting", "I may come later", "Room 204 is free", "nothing about time here"],
)
def test_ordinary_words_are_not_dates(text):
    assert parse_when(text, NOW) is None


def test_clock_time_already_past_today_moves_on():
    assert parse_when("at 9 am", NOW).at == dt.datetime(2026, 10, 1, 9, 0)  # 9 am today is gone
    assert parse_when("at 9", NOW).at == dt.datetime(2026, 9, 30, 21, 0)  # "at 9" means tonight


def test_describe():
    assert describe(dt.datetime(2026, 9, 30, 17, 0), NOW) == "at 5:00 pm"
    assert describe(dt.datetime(2026, 10, 1, 9, 0), NOW) == "tomorrow at 9:00 am"
    assert describe(dt.datetime(2026, 10, 2, 18, 0), NOW, has_time=False) == "Friday"
    assert describe(dt.datetime(2026, 10, 12, 11, 30), NOW) == "on 12 Oct at 11:30 am"
