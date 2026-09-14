"""A small always-on-top status window, drawn with tkinter.

tkinter is in the standard library, so this costs no dependency. It does come
with one hard constraint: every widget call must happen on the thread that
created the window. The voice loop runs elsewhere, so updates are pushed onto a
queue and drained by the window's own event loop.

Getting that wrong produces intermittent crashes on macOS that look like audio
bugs, which is why it is worth the queue rather than calling widgets directly.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from jarvis.logging_setup import get_logger
from jarvis.state import State
from jarvis.ui.base import StatusDisplay
from jarvis.ui.transcript import Transcript

log = get_logger("ui.window")

STATE_COLOUR: dict[State, str] = {
    State.IDLE: "#6b7280",
    State.LISTENING: "#22d3ee",
    State.THINKING: "#facc15",
    State.SPEAKING: "#4ade80",
    State.STOPPED: "#6b7280",
}

BACKGROUND = "#111317"
FOREGROUND = "#e5e7eb"
MUTED = "#9ca3af"
POLL_MILLISECONDS = 80


class WindowDisplay(StatusDisplay):
    """An always-on-top window showing state and transcript."""

    def __init__(
        self, *, name: str = "JARVIS", max_lines: int = 12, always_on_top: bool = True
    ) -> None:
        self._name = name
        self._transcript = Transcript(max_lines)
        self._always_on_top = always_on_top
        self._updates: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._failed = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="jarvis-ui")
        self._thread.start()
        # Wait briefly for the window to appear, but never block the assistant
        # on it: a display that will not start is not a reason not to listen.
        self._ready.wait(timeout=3.0)

    def stop(self) -> None:
        self._stopping.set()
        self._updates.put(("quit", None))
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def set_state(self, state: State) -> None:
        self._updates.put(("state", state))

    def add_line(self, speaker: str, text: str) -> None:
        self._transcript.add(speaker, text)
        self._updates.put(("transcript", None))

    def set_pending(self, speaker: str, text: str) -> None:
        self._transcript.set_pending(speaker, text)
        self._updates.put(("transcript", None))

    # -- the UI thread ------------------------------------------------------ #

    def _run(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            log.warning(
                "tkinter is not available, so the status window cannot open. "
                "JARVIS will carry on without it."
            )
            self._failed = True
            self._ready.set()
            return

        try:
            root = tk.Tk()
        except Exception as exc:
            # No display attached: over SSH, or a headless machine.
            log.warning("Could not open the status window (%s); carrying on without it.", exc)
            self._failed = True
            self._ready.set()
            return

        root.title(self._name)
        root.configure(bg=BACKGROUND)
        root.geometry("420x260")
        root.attributes("-topmost", self._always_on_top)

        state_label = tk.Label(
            root,
            text="idle",
            font=("Helvetica", 15, "bold"),
            bg=BACKGROUND,
            fg=STATE_COLOUR[State.IDLE],
            anchor="w",
            padx=14,
            pady=10,
        )
        state_label.pack(fill="x")

        transcript = tk.Text(
            root,
            bg=BACKGROUND,
            fg=FOREGROUND,
            font=("Helvetica", 11),
            bd=0,
            highlightthickness=0,
            wrap="word",
            padx=14,
            pady=4,
            state="disabled",
        )
        transcript.pack(fill="both", expand=True)
        transcript.tag_configure("you", foreground="#22d3ee")
        transcript.tag_configure("jarvis", foreground="#4ade80")
        transcript.tag_configure("aside", foreground=MUTED)

        def drain() -> None:
            try:
                while True:
                    kind, payload = self._updates.get_nowait()
                    if kind == "quit":
                        root.destroy()
                        return
                    if kind == "state":
                        label, colour = payload.value, STATE_COLOUR[payload]
                        state_label.configure(text=label, fg=colour)
                    elif kind == "transcript":
                        _fill(transcript, self._transcript)
            except queue.Empty:
                pass
            except Exception as exc:
                log.debug("Status window update failed: %r", exc)
            if not self._stopping.is_set():
                root.after(POLL_MILLISECONDS, drain)

        root.protocol("WM_DELETE_WINDOW", lambda: self._stopping.set())
        root.after(POLL_MILLISECONDS, drain)
        self._ready.set()

        try:
            root.mainloop()
        except Exception as exc:
            log.debug("Status window closed unexpectedly: %r", exc)


def _fill(widget: Any, transcript: Transcript) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    for line in transcript.lines():
        if line.is_aside:
            widget.insert("end", f"{line.text}\n", "aside")
        else:
            tag = line.speaker.lower() if line.speaker.lower() in {"you", "jarvis"} else ""
            widget.insert("end", f"{line.speaker} ", tag)
            widget.insert("end", f"{line.text}\n")
    widget.configure(state="disabled")
    widget.see("end")
