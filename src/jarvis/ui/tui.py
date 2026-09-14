"""A terminal status panel, drawn with rich.

Uses ``rich.Live`` to redraw a small panel in place: the current state, a
coloured indicator, and the last few lines of conversation. Works over SSH and
in any terminal, which is why it is the default of the two.

Refresh is capped rather than driven by every update, so a fast token stream
redraws a handful of times a second instead of hundreds.
"""

from __future__ import annotations

import contextlib
import threading

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from jarvis.state import State
from jarvis.ui.base import StatusDisplay
from jarvis.ui.transcript import Transcript

REFRESH_PER_SECOND = 8

STATE_STYLE: dict[State, tuple[str, str]] = {
    State.IDLE: ("idle", "dim"),
    State.LISTENING: ("listening", "bold cyan"),
    State.THINKING: ("thinking", "bold yellow"),
    State.SPEAKING: ("speaking", "bold green"),
    State.STOPPED: ("stopped", "dim"),
}

SPEAKER_STYLE = {"you": "cyan", "jarvis": "green"}


class TerminalDisplay(StatusDisplay):
    """A live-updating panel at the bottom of the terminal."""

    def __init__(
        self, *, name: str = "JARVIS", max_lines: int = 12, console: Console | None = None
    ) -> None:
        self._name = name
        self._transcript = Transcript(max_lines)
        self._state = State.IDLE
        self._console = console or Console()
        self._live: Live | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=REFRESH_PER_SECOND,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        with self._lock:
            live, self._live = self._live, None
        if live is not None:
            # A failing teardown must not mask whatever is actually going wrong.
            with contextlib.suppress(Exception):
                live.stop()

    def set_state(self, state: State) -> None:
        self._state = state
        self._refresh()

    def add_line(self, speaker: str, text: str) -> None:
        self._transcript.add(speaker, text)
        self._refresh()

    def set_pending(self, speaker: str, text: str) -> None:
        self._transcript.set_pending(speaker, text)
        self._refresh()

    def _refresh(self) -> None:
        with self._lock:
            if self._live is not None:
                self._live.update(self._render())

    def _render(self) -> RenderableType:
        label, style = STATE_STYLE[self._state]
        header = Text()
        header.append("* ", style=style)
        header.append(label, style=style)

        body: list[RenderableType] = [header, Text("")]
        for line in self._transcript.lines():
            body.append(_render_line(line))

        return Panel(
            Group(*body),
            title=self._name,
            title_align="left",
            border_style=style,
            padding=(0, 1),
        )


def _render_line(line: object) -> Text:
    speaker = getattr(line, "speaker", "")
    text = getattr(line, "text", "")
    if not speaker:
        return Text(f"  {text}", style="dim italic")
    rendered = Text()
    rendered.append(f"{speaker} ", style=SPEAKER_STYLE.get(speaker.lower(), "white"))
    rendered.append(text)
    return rendered
