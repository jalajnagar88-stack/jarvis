"""The brain interface.

This is the only place in JARVIS that talks to Anthropic. Everything else deals
in the event stream defined here, so the voice loop, the text REPL, and the
status window all consume the same thing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A fragment of the spoken reply, as it streams in."""

    text: str


@dataclass(frozen=True, slots=True)
class SentenceComplete:
    """A sentence boundary. This is the cue to start speaking.

    Emitted by the agent rather than the TTS layer because only the agent can
    tell a genuine sentence end from an abbreviation or a decimal point.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolCallFinished:
    name: str
    call_id: str
    ok: bool
    summary: str
    """One line, suitable for the status window. Not the full tool output."""


@dataclass(frozen=True, slots=True)
class ConfirmationRequired:
    """A tool needs explicit approval before it runs.

    The loop must present ``prompt`` verbatim — for ``run_shell`` that includes
    the full command — and call ``respond`` with the user's decision.
    """

    name: str
    call_id: str
    prompt: str
    details: str


@dataclass(frozen=True, slots=True)
class TurnFinished:
    """End of the assistant's turn."""

    text: str
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentFailed:
    """Something went wrong. ``spoken`` is what JARVIS should say out loud."""

    spoken: str
    detail: str


AgentEvent = (
    TextDelta
    | SentenceComplete
    | ToolCallStarted
    | ToolCallFinished
    | ConfirmationRequired
    | TurnFinished
    | AgentFailed
)

Role = Literal["user", "assistant"]


class Brain(ABC):
    """Turns user text into a stream of events."""

    @abstractmethod
    def respond(self, user_text: str) -> Iterator[AgentEvent]:
        """Handle one user turn.

        Yields events as they happen. The caller drives the iterator, which is
        what lets it stop consuming — and so cancel the turn — the moment the
        user interrupts.
        """

    @abstractmethod
    def confirm(self, call_id: str, approved: bool) -> None:
        """Answer a :class:`ConfirmationRequired` raised during :meth:`respond`."""

    @abstractmethod
    def reset(self) -> None:
        """Drop the working conversation. Stored memory is unaffected."""

    @property
    @abstractmethod
    def history(self) -> list[dict[str, Any]]:
        """The working conversation, in Anthropic message format."""
