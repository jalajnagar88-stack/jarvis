"""A brain that says back whatever it heard.

This is what milestone 2 ran on, kept because it is the only way to test the
audio path in isolation: if echo sounds right but the real brain does not, the
problem is reasoning, not microphones. Reachable as ``jarvis run --no-brain``,
and it needs no API key.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from jarvis.interfaces.agent import (
    AgentEvent,
    Brain,
    SentenceComplete,
    TextDelta,
    TurnFinished,
)
from jarvis.tts.sentences import split_sentences


class EchoBrain(Brain):
    """Repeats the user, one sentence at a time."""

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages.clear()

    def confirm(self, call_id: str, approved: bool) -> None:
        raise NotImplementedError("The echo brain never asks for confirmation.")

    def respond(self, user_text: str) -> Iterator[AgentEvent]:
        text = user_text.strip()
        if not text:
            yield TurnFinished(text="", stop_reason="empty")
            return

        yield TextDelta(text=text)
        for sentence in split_sentences(text):
            yield SentenceComplete(text=sentence)

        self._messages.append({"role": "user", "content": text})
        self._messages.append({"role": "assistant", "content": text})
        yield TurnFinished(text=text, stop_reason="echo")
