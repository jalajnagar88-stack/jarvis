"""The transcript model, shared by both displays.

Keeping this separate from either display means the bounded history, the
speaker labels, and the pending-line handling are written once and tested
without a terminal or a window.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Line:
    """One line of transcript."""

    speaker: str
    text: str
    at: datetime

    @property
    def is_aside(self) -> bool:
        """True for notes that are not speech -- tool calls, interruptions."""
        return not self.speaker


class Transcript:
    """A bounded, thread-safe conversation log for display.

    Bounded because a display that grows forever eventually costs more to
    render than the assistant costs to run.
    """

    def __init__(self, max_lines: int = 12) -> None:
        self._lines: deque[Line] = deque(maxlen=max(1, max_lines))
        self._pending: Line | None = None
        self._lock = threading.Lock()

    def add(self, speaker: str, text: str) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        with self._lock:
            self._lines.append(Line(speaker, cleaned, datetime.now()))
            self._pending = None

    def set_pending(self, speaker: str, text: str) -> None:
        """Show a line still arriving. Cleared when a finished line is added."""
        cleaned = " ".join(text.split())
        with self._lock:
            self._pending = Line(speaker, cleaned, datetime.now()) if cleaned else None

    def clear_pending(self) -> None:
        with self._lock:
            self._pending = None

    def lines(self) -> list[Line]:
        """Everything to display, pending line last."""
        with self._lock:
            visible = list(self._lines)
            if self._pending is not None:
                visible.append(self._pending)
            return visible

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._pending = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._lines)
