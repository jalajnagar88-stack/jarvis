"""Interpreting a spoken yes or no.

Deliberately strict. Anything not clearly affirmative is treated as a refusal,
because the cost of the two mistakes is not symmetric: a missed "yes" means
asking again, a missed "no" means running a command the user did not want.

Whisper output is lowercase-ish but punctuated and sometimes padded with filler,
so matching is on words rather than on the whole string.
"""

from __future__ import annotations

import re

_AFFIRMATIVE = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "aye",
        "ok",
        "okay",
        "sure",
        "certainly",
        "please",
        "affirmative",
        "confirm",
        "confirmed",
        "go",
        "proceed",
        "do",
        "run",
        "correct",
        "right",
        "indeed",
        "absolutely",
        "definitely",
    }
)

_NEGATIVE = frozenset(
    {
        "no",
        "nope",
        "nah",
        "negative",
        "cancel",
        "stop",
        "abort",
        "don't",
        "dont",
        "never",
        "wait",
        "hold",
        "skip",
        "forget",
        "nevermind",
    }
)

_WORD = re.compile(r"[a-z']+")


def interpret_confirmation(text: str) -> bool | None:
    """Decide what the user meant.

    Returns:
        True for a clear yes, False for a clear no, and None when it is neither
        -- which the caller should treat as "ask again" rather than as consent.
    """
    words = _WORD.findall(text.lower())
    if not words:
        return None

    # A negation anywhere wins. "yes, don't do that" is a refusal, and treating
    # it as consent because it starts with "yes" would be exactly the wrong
    # failure.
    if any(word in _NEGATIVE for word in words):
        return False
    if any(word in _AFFIRMATIVE for word in words):
        return True
    return None


def is_affirmative(text: str) -> bool:
    """True only for an unambiguous yes. Anything else is a refusal."""
    return interpret_confirmation(text) is True
