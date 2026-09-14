"""JARVIS — a local-first voice assistant.

The package is deliberately layered so that any single piece can be swapped
without touching the rest:

    audio_in -> wake_word -> stt -> agent -> [tools] -> tts -> audio_out
                                      |
                                    memory

Each stage is declared as an abstract base class in :mod:`jarvis.interfaces`
and implemented in its own subpackage. No implementation imports another
implementation's internals; they communicate through the interfaces and the
plain dataclasses defined alongside them.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
