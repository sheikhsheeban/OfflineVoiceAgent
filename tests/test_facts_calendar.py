"""Fact requests (sheet 4 schema, privacy) and the local .ics calendar."""

import datetime as dt

import pytest
from pydantic import ValidationError

from kin.reasoning.calendar import Calendar
from kin.reasoning.facts import FactRequest, FactService, FactUnavailable, build_request
from tests.conftest import DATA

NOW = dt.datetime(2026, 9, 30, 10, 40)


def test_fact_request_schema_matches_sheet_4():
    r = FactRequest(kind="weather", lat=21.1234, lon=79.0876, hours=48)
    assert (r.lat, r.lon) == (21.1, 79.1)  # rounded to 0.1 degree
    with pytest.raises(ValidationError):
        FactRequest(kind="email")
    with pytest.raises(ValidationError):
        FactRequest(kind="weather", hours=72)
    with pytest.raises(ValidationError):
        FactRequest(kind="search", q="x" * 81)
    with pytest.raises(ValidationError):
        FactRequest(kind="search", q="hi", transcript="leak")  # unknown fields are refused
    with pytest.raises(ValidationError):
        r.hours = 1  # frozen


def test_payload_holds_only_that_kinds_fields():
    assert FactRequest(kind="search", q="train timings to Pune").payload() == {
        "kind": "search",
        "q": "train timings to Pune",
    }
    assert set(FactRequest(kind="weather", lat=1, lon=2).payload()) == {"kind", "lat", "lon", "hours"}


def test_rules_build_requests_without_the_model():
    w = build_request("fact.weather", "Will it rain on Thursday?", NOW)
    assert w.kind == "weather" and w.lat is not None and w.hours == 48
    assert build_request("fact.weather", "Will it rain in 3 hours?", NOW).hours == 6
    assert build_request("fact.news", "any cricket news", NOW).topic == "sport"


def test_search_requests_drop_names_and_sensitive_numbers():
    r = build_request("fact.search", "Rohan asked if the Wardha road is closed, OTP 482913", NOW, people={"Rohan"})
    assert "rohan" not in r.q.lower() and "482913" not in r.q
    assert "Wardha" in r.q and len(r.q) <= 80


def test_no_gateway_yet():
    with pytest.raises(FactUnavailable, match="can't check the weather right now"):
        FactService().fetch(FactRequest(kind="weather", lat=21.1, lon=79.1))


def test_calendar_reads_events_and_finds_relevant_ones():
    cal = Calendar(DATA / "demo_calendar.ics")
    evs = cal.events()
    assert [e.title for e in evs] == ["Site visit", "Budget review with Priya", "Dentist"]
    assert evs[0].location == "Butibori" and evs[0].start == dt.datetime(2026, 10, 1, 16, 0)
    assert [e.title for e in cal.relevant("when is the site visit", NOW)] == ["Site visit"]
    assert [e.title for e in cal.relevant("anything on", NOW, day=dt.date(2026, 10, 5))] == ["Dentist"]
    assert cal.relevant("pasta for lunch", NOW) == []
    assert "Thursday 1 October at 4:00 PM" in evs[0].describe()


def test_missing_or_broken_calendar_is_empty(tmp_path):
    assert Calendar(None).events() == []
    assert Calendar(tmp_path / "none.ics").events() == []
    bad = tmp_path / "bad.ics"
    bad.write_text("this is not a calendar")
    assert Calendar(bad).events() == []


def test_calendar_reloads_when_the_file_changes(tmp_path):
    p = tmp_path / "c.ics"
    src = (DATA / "demo_calendar.ics").read_text()
    p.write_text(src)
    cal = Calendar(p)
    assert len(cal.events()) == 3
    import os
    import time

    p.write_text(src.replace("SUMMARY:Dentist", "SUMMARY:Dentist moved"))
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert cal.events()[-1].title == "Dentist moved"
