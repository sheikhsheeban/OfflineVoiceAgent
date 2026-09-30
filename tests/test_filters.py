import pytest

from kin.perception.filters import CARD_TOKEN, CODE_TOKEN, luhn_ok, redact


@pytest.mark.parametrize(
    "text",
    [
        "Your OTP is 482913.",
        "The verification code is 4 8 2 9 1 3",
        "OTP four eight two nine one three, don't share it",
        "It's 7731, that's the one time password",
        "my pin is 0420",
    ],
)
def test_codes_are_removed(text):
    out = redact(text)
    assert CODE_TOKEN in out.text and out.redactions == 1
    assert not any(ch.isdigit() for ch in out.text)


@pytest.mark.parametrize(
    "text",
    [
        "Card number 4111 1111 1111 1111 expires soon",
        "it's 4111-1111-1111-1111",
        "5500 0055 5555 5559",
    ],
)
def test_valid_card_numbers_are_removed(text):
    out = redact(text)
    assert CARD_TOKEN in out.text and "1111" not in out.text and "5555" not in out.text


@pytest.mark.parametrize(
    "text",
    [
        "The site visit is Thursday at 4 pm",
        "We moved to Butibori in 2019",
        "Call me on 9823 4567 12",  # not a code context, not a valid card
        "Order 1234 5678 9012 3456 is late",  # 16 digits but fails the Luhn check
        "one or two people will come",
        "Someone said the budget is 45000",
    ],
)
def test_ordinary_numbers_are_kept(text):
    out = redact(text)
    assert out.text == text and out.redactions == 0


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")


@pytest.mark.parametrize(
    "text,token",
    [
        ("My credit card number is 1234567", "[card details removed]"),  # partial, announced
        ("my card ending in 4521", "[card details removed]"),
        ("card number one two three four five six", "[card details removed]"),
        ("the CVV is 123", "[card details removed]"),
        ("My account number is 50100234567812", "[account number removed]"),
        ("Aadhaar 1234 5678 9012", "[account number removed]"),
        ("UPI PIN is 4 4 5 5", "[account number removed]"),
    ],
)
def test_announced_card_and_account_numbers_are_removed(text, token):
    out = redact(text)
    assert token in out.text and out.redactions == 1
    assert sum(ch.isdigit() for ch in out.text) == 0


@pytest.mark.parametrize(
    "text",
    [
        "I paid 4500 with my credit card",  # an amount near "card" is not a card number
        "The card game starts at 1130",
        "Room 204 is free",
    ],
)
def test_card_word_alone_does_not_remove_amounts(text):
    assert redact(text).text == text
