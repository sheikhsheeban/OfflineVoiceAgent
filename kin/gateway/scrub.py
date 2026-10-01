"""The gateway's own privacy scrub (sheet 4), applied again to whatever kin-core sends.

kin-core already builds requests without names, but the gateway does not trust
its caller: names from the `people` table, phone numbers, e-mail addresses and
digit runs are removed from a search query before it can leave.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_DIGITS = re.compile(r"\d{4,}")  # long digit runs: codes, account and card numbers


def scrub_query(q: str, names: list[str] | None = None) -> str:
    q = _EMAIL.sub(" ", q)
    q = _PHONE.sub(" ", q)
    q = _DIGITS.sub(" ", q)
    for name in names or ():
        if name.strip():
            q = re.sub(rf"\b{re.escape(name.strip())}\b", " ", q, flags=re.I)
    return " ".join(q.split())[:80]
