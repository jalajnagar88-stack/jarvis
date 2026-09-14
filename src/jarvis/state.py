"""What the assistant is doing right now.

Kept in its own module with no dependencies so that the voice loop, the text
REPL, and the milestone 7 status window can all agree on the vocabulary without
importing each other.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum


class State(StrEnum):
    """The assistant's observable state."""

    IDLE = "idle"
    """Listening for the wake word, doing nothing else."""

    LISTENING = "listening"
    """Recording a command; stops on silence."""

    THINKING = "thinking"
    """Transcribing, and later reasoning and running tools."""

    SPEAKING = "speaking"
    """Playing a reply. Interruptible."""

    STOPPED = "stopped"
    """Shut down."""


StateListener = Callable[[State], None]
"""Called on every state transition. Must not block — it runs on the loop thread."""
