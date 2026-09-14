"""Streaming sentence splitting.

Milestone 3 starts speaking on the first complete sentence rather than waiting
for the whole reply. That only feels good if the boundaries are right: cutting
after "Dr." or mid-decimal is worse than waiting.
"""

from __future__ import annotations

import pytest

from jarvis.tts.sentences import SentenceSplitter, split_sentences


class TestBasicSplitting:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("One. Two. Three.", ["One.", "Two.", "Three."]),
            ("Is it? Yes! Quite.", ["Is it?", "Yes!", "Quite."]),
            ("No terminator here", ["No terminator here"]),
            ("", []),
            ("   ", []),
        ],
    )
    def test_simple_cases(self, text: str, expected: list[str]) -> None:
        assert split_sentences(text) == expected

    def test_quotes_and_brackets_close_a_sentence(self) -> None:
        assert split_sentences('He said "hello." Then left.') == [
            'He said "hello."',
            "Then left.",
        ]

    def test_whitespace_is_normalised_away(self) -> None:
        assert split_sentences("One.    Two.") == ["One.", "Two."]


class TestFalseBoundaries:
    """The cases that make early speech sound broken."""

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Watson will see you now.",
            "Meet me at 3 p.m. tomorrow.",
            "That is 21.5 degrees outside.",
            "Bring milk, eggs, etc. from the shop.",
            "J. R. R. Tolkien wrote it.",
            "The U.K. is rather damp.",
        ],
    )
    def test_these_are_all_one_sentence(self, text: str) -> None:
        assert split_sentences(text) == [text]

    def test_a_decimal_does_not_split(self) -> None:
        assert split_sentences("It is 3.14159 exactly. Truly.") == [
            "It is 3.14159 exactly.",
            "Truly.",
        ]

    def test_an_abbreviation_still_ends_a_real_sentence(self) -> None:
        assert split_sentences("See you at 3 p.m. Goodbye.") == ["See you at 3 p.m. Goodbye."]


class TestStreaming:
    def test_sentences_are_emitted_as_soon_as_they_complete(self) -> None:
        splitter = SentenceSplitter()
        assert splitter.feed("Good evening") == []
        assert splitter.feed(", sir.") == []  # no following char yet
        assert splitter.feed(" Shall I") == ["Good evening, sir."]

    def test_a_terminator_at_the_fragment_edge_waits_for_more(self) -> None:
        """Without this, "3." is spoken before the "14" arrives."""
        splitter = SentenceSplitter()
        assert splitter.feed("The answer is 3.") == []
        assert splitter.feed("14 exactly.") == []
        assert splitter.feed(" Done.") == ["The answer is 3.14 exactly."]

    def test_flush_returns_the_unterminated_remainder(self) -> None:
        splitter = SentenceSplitter()
        splitter.feed("All done. And one more thing")
        assert splitter.flush() == "And one more thing"

    def test_flush_empties_the_buffer(self) -> None:
        splitter = SentenceSplitter()
        splitter.feed("Leftover")
        splitter.flush()
        assert splitter.flush() == ""

    def test_pending_exposes_what_is_still_buffered(self) -> None:
        splitter = SentenceSplitter()
        splitter.feed("One. Two")
        assert splitter.pending.strip() == "Two"

    def test_character_at_a_time_matches_whole_text(self) -> None:
        """The worst case a streaming API can produce: one token per character."""
        text = "Good evening, sir. The time is 3.14. Shall I proceed?"
        splitter = SentenceSplitter()
        streamed: list[str] = []
        for char in text:
            streamed.extend(splitter.feed(char))
        remainder = splitter.flush()
        if remainder:
            streamed.append(remainder)
        assert streamed == split_sentences(text)

    def test_several_sentences_in_one_fragment(self) -> None:
        splitter = SentenceSplitter()
        assert splitter.feed("One. Two. Three. ") == ["One.", "Two.", "Three."]

    def test_very_short_fragments_are_not_emitted_as_sentences(self) -> None:
        """A stray "." should not become a unit of speech."""
        splitter = SentenceSplitter(min_length=2)
        assert splitter.feed(". . ") == []
