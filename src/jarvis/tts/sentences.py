"""Splitting streamed text into speakable sentences.

Milestone 3 starts speaking as soon as the first sentence is complete rather
than waiting for the whole reply, which is most of the difference between an
assistant that feels immediate and one that feels slow. That needs a splitter
that can tell a real sentence end from a decimal point or an abbreviation,
working on a stream where the next character has not arrived yet.

It lives in the tts package because sentence boundaries are a speech concern,
but it is pure text and has no dependencies.
"""

from __future__ import annotations

import re

_TERMINATORS = ".!?"
_CLOSERS = "\"')]}\u201d\u2019"

# Things that end in a full stop but do not end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "rev",
        "hon",
        "e.g",
        "i.e",
        "etc",
        "vs",
        "approx",
        "est",
        "no",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "mon",
        "tue",
        "tues",
        "wed",
        "thu",
        "thur",
        "thurs",
        "fri",
        "sat",
        "sun",
        "a.m",
        "p.m",
        "u.k",
        "u.s",
        "ph.d",
    }
)

_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


class SentenceSplitter:
    """Accumulates streamed text and emits complete sentences.

    Feed it fragments with :meth:`feed`; it returns whichever sentences became
    complete. Call :meth:`flush` at the end of the stream for the remainder.
    """

    def __init__(self, *, min_length: int = 2) -> None:
        self._buffer = ""
        self._min_length = min_length

    def feed(self, fragment: str) -> list[str]:
        """Add streamed text; return any sentences that are now complete."""
        self._buffer += fragment
        sentences: list[str] = []

        while True:
            index = self._find_boundary(self._buffer)
            if index is None:
                break
            sentence = self._buffer[: index + 1].strip()
            self._buffer = self._buffer[index + 1 :]
            if len(sentence) >= self._min_length:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str:
        """Return whatever is left, and clear the buffer."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder

    @property
    def pending(self) -> str:
        """Text buffered but not yet a complete sentence."""
        return self._buffer

    def _find_boundary(self, text: str) -> int | None:
        """Index of the last character belonging to the first complete sentence.

        The index points past any closing quote or bracket, so that
        ``He said "hello."`` keeps its quote instead of handing it to the next
        sentence.
        """
        for index, char in enumerate(text):
            if char not in _TERMINATORS:
                continue
            if char == "." and self._is_abbreviation(text[: index + 1]):
                continue
            if char == "." and self._is_decimal(text, index):
                continue

            end = index
            while end + 1 < len(text) and text[end + 1] in _CLOSERS:
                end += 1

            # Require a following character, so a terminator arriving at the very
            # end of a fragment waits for the next one. Without this, "3." is
            # emitted before the "14" arrives.
            if end + 1 >= len(text):
                return None
            if not text[end + 1].isspace():
                continue
            return end
        return None

    def _is_abbreviation(self, text_up_to_dot: str) -> bool:
        match = _TRAILING_WORD.search(text_up_to_dot.strip())
        if match is None:
            return False
        word = match.group(1).rstrip(".").lower()
        if word in _ABBREVIATIONS:
            return True
        # A single letter followed by a dot is an initial: "J. Smith".
        return len(word) == 1

    def _is_decimal(self, text: str, index: int) -> bool:
        before = text[index - 1] if index > 0 else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        return before.isdigit() and after.isdigit()


def split_sentences(text: str) -> list[str]:
    """Split a complete piece of text. Convenience wrapper for the non-streaming case."""
    splitter = SentenceSplitter()
    sentences = splitter.feed(text)
    remainder = splitter.flush()
    if remainder:
        sentences.append(remainder)
    return sentences
