"""When-phrases in everyday speech: "in 20 minutes", "at 5 pm", "tomorrow morning",
"by Friday", "after work", "next week".

Deterministic rules cover what people actually say to a reminder app; `dateparser`
is only consulted for calendar dates such as "12 October" (it misreads words like
"we" and "may" as dates, and takes seconds to import, so it is loaded lazily).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
    "a couple of": 2, "a few": 3, "few": 3,
}  # fmt: skip
_NUMWORD = "|".join(sorted((re.escape(k) for k in _NUM), key=len, reverse=True))
_UNIT = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 604800}

# Default clock times for parts of the day.
DAY_PARTS = {
    "morning": (9, 0), "noon": (12, 0), "midday": (12, 0), "lunch": (13, 0), "afternoon": (14, 0),
    "evening": (18, 0), "after work": (18, 0), "tonight": (20, 0), "night": (21, 0), "midnight": (23, 59),
}  # fmt: skip
DEFAULT_DUE_TIME = (18, 0)  # a task due "Friday" is due by Friday evening

_RELATIVE = re.compile(
    rf"\b(?:in|after)\s+(?:(half an? hour|an? hour and a half)|(\d+|{_NUMWORD})\s*"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?))\b",
    re.IGNORECASE,
)
_CLOCK = re.compile(
    r"\b(?:at\s+|by\s+|around\s+|for\s+)?(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?|o'?clock)?(?=\W|$)",
    re.IGNORECASE,
)
_DAY = re.compile(
    r"\b(?:(?:on|by|this|next|coming|before)\s+)*(today|tonight|tomorrow|day after tomorrow|"
    + "|".join(WEEKDAYS)
    + r"|next week|this weekend|weekend)\b",
    re.IGNORECASE,
)
_PART = re.compile(r"\b(?:this\s+|in the\s+|tomorrow\s+)?(" + "|".join(DAY_PARTS) + r")\b", re.IGNORECASE)
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_CAL_DATE = re.compile(
    rf"\b(?:on\s+|by\s+)?(?:(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})[a-z]*|(?:{_MONTHS})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?)\b",
    re.IGNORECASE,
)


@dataclass
class When:
    at: dt.datetime
    has_time: bool
    spans: list[tuple[int, int]] = field(default_factory=list)  # where the phrase was found

    def strip_from(self, text: str) -> str:
        """The text with the when-phrase removed (for titles like "call the landlord")."""
        merged: list[list[int]] = []
        for a, b in sorted(self.spans):  # "tomorrow" and "tomorrow morning" overlap
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out = text
        for a, b in reversed(merged):
            out = out[:a] + " " + out[b:]
        return " ".join(out.split()).strip(" ,.;")


def _num(s: str) -> int:
    s = s.lower()
    return int(s) if s.isdigit() else _NUM[s]


def _next_weekday(now: dt.datetime, name: str, force_next: bool) -> dt.date:
    target = WEEKDAYS.index(name)
    days = (target - now.weekday()) % 7
    if days == 0 and force_next:
        days = 7
    return (now + dt.timedelta(days=days)).date()


def parse_when(text: str, now: dt.datetime) -> When | None:
    """Find the first when-phrase in `text`, resolved against `now` (naive local time)."""
    low = text.lower()

    m = _RELATIVE.search(low)
    if m:
        if m.group(1):
            secs = 1800 if m.group(1).startswith("half") else 5400
        else:
            unit = m.group(3).rstrip("s")
            secs = _num(m.group(2)) * _UNIT[unit]
        return When(now + dt.timedelta(seconds=secs), True, [m.span()])

    spans: list[tuple[int, int]] = []
    day: dt.date | None = None
    dm = _DAY.search(low)
    if dm:
        word = dm.group(1)
        spans.append(dm.span())
        if word in ("today", "tonight"):
            day = now.date()
        elif word == "tomorrow":
            day = (now + dt.timedelta(days=1)).date()
        elif word == "day after tomorrow":
            day = (now + dt.timedelta(days=2)).date()
        elif word == "next week":
            day = (now + dt.timedelta(days=7 - now.weekday())).date()  # next Monday
        elif word in ("weekend", "this weekend"):
            day = _next_weekday(now, "saturday", force_next=False)
        else:
            day = _next_weekday(now, word, force_next="next" in dm.group(0))
    else:
        cm = _CAL_DATE.search(low)
        if cm:
            parsed = _dateparser(text[cm.start() : cm.end()], now)
            if parsed:
                day = parsed.date()
                spans.append(cm.span())

    hm: tuple[int, int] | None = None
    for c in _CLOCK.finditer(low):
        h, mnt, suffix = int(c.group(1)), int(c.group(2) or 0), (c.group(3) or "").replace(".", "")
        phrase = c.group(0).lower()
        explicit = suffix or c.group(2) or phrase.startswith(("at", "by", "around"))
        if not explicit or h > 23 or mnt > 59:
            continue  # a bare number ("4 chairs") is not a time
        if suffix.startswith("p") and h < 12:
            h += 12
        elif suffix.startswith("a") and h == 12:
            h = 0
        elif not suffix.startswith(("a", "p")) and h <= 12:
            # "at 6" with no am/pm: the next 6 o'clock that is still ahead (or in the evening for a later day)
            if day in (None, now.date()):
                if (h, mnt) <= (now.hour, now.minute) and h + 12 < 24:
                    h += 12
            elif h < 7:
                h += 12
        hm = (h, mnt)
        spans.append(c.span())
        break
    if hm is None:
        pm = _PART.search(low)
        if pm:
            hm = DAY_PARTS[pm.group(1)]
            spans.append(pm.span())

    if day is None and hm is None:
        return None
    if day is None:
        day = now.date()
        if dt.datetime.combine(day, dt.time(*hm)) <= now:
            day = day + dt.timedelta(days=1)
    has_time = hm is not None
    at = dt.datetime.combine(day, dt.time(*(hm or DEFAULT_DUE_TIME)))
    return When(at, has_time, spans)


def _dateparser(phrase: str, now: dt.datetime) -> dt.datetime | None:
    import dateparser

    return dateparser.parse(
        phrase,
        languages=["en"],
        settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now, "RETURN_AS_TIMEZONE_AWARE": False},
    )


def describe(at: dt.datetime, now: dt.datetime, has_time: bool = True) -> str:
    """Short spoken form: "at 5:00 pm", "tomorrow at 9:00 am", "Friday", "on 12 Oct"."""
    t = at.strftime("%I:%M %p").lstrip("0").lower()
    days = (at.date() - now.date()).days
    if days == 0:
        d = "today"
    elif days == 1:
        d = "tomorrow"
    elif 1 < days < 7:
        d = at.strftime("%A")
    else:
        d = "on " + at.strftime("%d %b").lstrip("0")
    if not has_time:
        return d
    return f"at {t}" if days == 0 else f"{d} at {t}"
