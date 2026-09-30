"""Write-path filters: remove one-time codes, card and account numbers before anything is stored.

Runs on every transcript before it reaches the store or the UI. Speech
recognisers write digits either as numerals ("4 5 2 1") or as words
("four five two one"), so both forms are handled.

A digit run (4+ digits) is removed when:
  - it is a complete card number (13-19 digits passing the Luhn check), anywhere;
  - the words around it announce a card ("card number", "card ending in", CVV,
    expiry), whatever its length, since people often say partial numbers;
  - the words around it announce an account or ID number (account number,
    Aadhaar, PAN, UPI PIN, IFSC);
  - the words around it announce a code (OTP, PIN, password, verification code).
Plain amounts, times, years and phone numbers are kept. "I paid 4500 with my
card" keeps the amount: the context must name the number, not just the card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Bump when the rules change: kin-core then re-filters everything already stored.
FILTER_VERSION = 2

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}  # fmt: skip
_DIGIT = r"(?:\d|zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_SEP = r"[\s,.\-/]*"

# A run of spoken or written digits, e.g. "4521", "4 5 2 1", "four five two one".
_DIGIT_RUN = re.compile(rf"(?<![\w]){_DIGIT}(?:{_SEP}{_DIGIT})+(?![\w])|(?<![\w])\d{{4,}}(?![\w])", re.IGNORECASE)
_CODE_CONTEXT = re.compile(
    r"\b(otp|one[\s-]?time|passcode|pass code|password|pin|verification|security code|code)\b",
    re.IGNORECASE,
)
_CARD_CONTEXT = re.compile(
    r"\b(card\s*(number|no|num|ending(\s+in|\s+with)?|digits|details|is)|cvv|cvc|expiry|expiration)\b",
    re.IGNORECASE,
)
_ACCOUNT_CONTEXT = re.compile(
    r"\b(account\s*(number|no|num|is)|a/c|aadhaa?r|pan\s*(number|no|card)|upi\s*pin|ifsc|bank\s*details)\b",
    re.IGNORECASE,
)

CODE_TOKEN = "[code removed]"
CARD_TOKEN = "[card number removed]"
CARD_DETAILS_TOKEN = "[card details removed]"
ACCOUNT_TOKEN = "[account number removed]"


@dataclass
class Filtered:
    text: str
    redactions: int


def _digits(s: str) -> str:
    return "".join(_DIGIT_WORDS.get(w.lower(), w) for w in re.findall(r"\d|[a-zA-Z]+", s))


def luhn_ok(num: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def redact(text: str) -> Filtered:
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        digits = _digits(m.group(0))
        if not digits.isdigit() or len(digits) < 3:
            return m.group(0)
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            count += 1
            return CARD_TOKEN
        near = text[max(0, m.start() - 40) : m.start()] + " … " + text[m.end() : m.end() + 25]
        if _CARD_CONTEXT.search(near):  # includes 3-digit CVVs and partial numbers
            count += 1
            return CARD_DETAILS_TOKEN
        if len(digits) < 4:
            return m.group(0)
        if _ACCOUNT_CONTEXT.search(near):
            count += 1
            return ACCOUNT_TOKEN
        if len(digits) <= 8 and _CODE_CONTEXT.search(near):
            count += 1
            return CODE_TOKEN
        return m.group(0)

    out = _DIGIT_RUN.sub(repl, text)
    return Filtered(out, count)
