"""The user's local calendar: an .ics file exported from their calendar app.

Read-only and local. Timed events are converted to local time; all-day events
start at midnight. Recurring events are read as their first occurrence only.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("kin.calendar")
_WORD = re.compile(r"[a-z0-9']+")
_STOP = set("the a an and or of to in on at for with my me i is was what when where who did said about".split())


@dataclass(frozen=True)
class Event:
    uid: str
    title: str
    start: dt.datetime
    end: dt.datetime
    location: str
    all_day: bool

    def describe(self) -> str:
        when = self.start.strftime("%A %d %B").replace(" 0", " ")
        if not self.all_day:
            when += (
                " at "
                + self.start.strftime("%I:%M %p").lstrip("0")
                + " until "
                + self.end.strftime("%I:%M %p").lstrip("0")
            )
        loc = f" at {self.location}" if self.location else ""
        return f"{self.title}{loc}, {when}"


def _local(v) -> tuple[dt.datetime, bool]:
    if isinstance(v, dt.datetime):
        if v.tzinfo is not None:
            v = v.astimezone().replace(tzinfo=None)
        return v, False
    return dt.datetime.combine(v, dt.time()), True


class Calendar:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._mtime: float | None = None
        self._events: list[Event] = []

    def events(self) -> list[Event]:
        if self.path is None or not self.path.exists():
            self._events, self._mtime = [], None
            return []
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            self._events = self._load()
            self._mtime = mtime
        return self._events

    def _load(self) -> list[Event]:
        import icalendar

        try:
            cal = icalendar.Calendar.from_ical(self.path.read_bytes())
        except Exception as e:  # noqa: BLE001 - a broken file must not break answers
            log.warning("cannot read calendar %s: %s", self.path, e)
            return []
        out = []
        for c in cal.walk("VEVENT"):
            if "DTSTART" not in c:
                continue
            start, all_day = _local(c.decoded("DTSTART"))
            if "DTEND" in c:
                end, _ = _local(c.decoded("DTEND"))
            else:
                end = start + (dt.timedelta(days=1) if all_day else dt.timedelta(hours=1))
            out.append(
                Event(
                    str(c.get("UID", "")),
                    str(c.get("SUMMARY", "")).strip(),
                    start,
                    end,
                    str(c.get("LOCATION", "")).strip(),
                    all_day,
                )  # fmt: skip
            )
        out.sort(key=lambda e: e.start)
        return out

    def between(self, start: dt.datetime, end: dt.datetime) -> list[Event]:
        return [e for e in self.events() if e.start < end and e.end > start]

    def relevant(self, text: str, now: dt.datetime, day: dt.date | None = None, limit: int = 3) -> list[Event]:
        """Events that bear on a question: on the day it names, or sharing a word with it
        within the next two weeks."""
        if day is not None:
            start = dt.datetime.combine(day, dt.time())
            return self.between(start, start + dt.timedelta(days=1))[:limit]
        words = {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}
        if not words:
            return []
        soon = self.between(now - dt.timedelta(hours=2), now + dt.timedelta(days=14))
        hits = [e for e in soon if words & set(_WORD.findall(f"{e.title} {e.location}".lower()))]
        return hits[:limit]
