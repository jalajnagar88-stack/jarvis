"""Status display: a terminal panel or an always-on-top window.

Both implement :class:`~jarvis.ui.base.StatusDisplay`, and which one is used is
decided by ``ui.mode`` in config.yaml. The default is neither -- the assistant
is a voice interface, and a window is opt-in.
"""

from __future__ import annotations

from jarvis.ui.base import NullDisplay, StatusDisplay
from jarvis.ui.transcript import Line, Transcript

__all__ = ["Line", "NullDisplay", "StatusDisplay", "Transcript"]
