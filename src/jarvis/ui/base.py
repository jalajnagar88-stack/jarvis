"""The status display interface.

A small thing that shows what JARVIS is doing -- idle, listening, thinking,
speaking -- and a running transcript. Two implementations sit behind this: a
terminal panel and an always-on-top window.

Everything here is called from the voice loop's thread and must not block. A
display that stalls the loop would delay speech, which is worse than no display
at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from jarvis.state import State


class StatusDisplay(ABC):
    """Shows the assistant's state and transcript."""

    @abstractmethod
    def start(self) -> None:
        """Begin displaying. Idempotent."""

    @abstractmethod
    def stop(self) -> None:
        """Tear down. Idempotent, and safe to call from any thread."""

    @abstractmethod
    def set_state(self, state: State) -> None:
        """Update the displayed state. Must return immediately."""

    @abstractmethod
    def add_line(self, speaker: str, text: str) -> None:
        """Append a finished line of transcript."""

    @abstractmethod
    def set_pending(self, speaker: str, text: str) -> None:
        """Show a line still being produced. Replaces any previous pending line."""

    def note(self, text: str) -> None:
        """Show a transient aside -- a tool running, an interruption."""
        self.add_line("", text)

    def __enter__(self) -> StatusDisplay:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


class NullDisplay(StatusDisplay):
    """Shows nothing. The default, and what the tests use."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def set_state(self, state: State) -> None: ...

    def add_line(self, speaker: str, text: str) -> None: ...

    def set_pending(self, speaker: str, text: str) -> None: ...
