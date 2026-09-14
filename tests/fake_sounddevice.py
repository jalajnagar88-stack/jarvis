"""A stand-in for the ``sounddevice`` module.

The real one cannot be imported without the native PortAudio library, which is
exactly the situation CI is in. This fake implements the small surface
:mod:`jarvis.audio.sounddevice_io` uses -- ``query_devices``, ``default.device``,
``InputStream``, ``OutputStream`` -- and lets tests drive the audio callback by
hand, so the callback logic is genuinely exercised rather than mocked away.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

DEVICES: list[dict[str, Any]] = [
    {
        "name": "MacBook Pro Microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {
        "name": "MacBook Pro Speakers",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    {
        "name": "Jabra Evolve 65",
        "max_input_channels": 1,
        "max_output_channels": 2,
        "default_samplerate": 16000.0,
    },
]


class _Stream:
    """Common behaviour for both stream directions."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback: Callable[..., None] = kwargs["callback"]
        self.samplerate = kwargs.get("samplerate")
        self.blocksize = kwargs.get("blocksize", 1024)
        self.device = kwargs.get("device")
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class InputStream(_Stream):
    def deliver(self, samples: np.ndarray, status: Any = None) -> None:
        """Simulate PortAudio handing us a block of captured audio."""
        block = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
        self.callback(block, len(block), None, status)


class OutputStream(_Stream):
    def pull(self, frames: int | None = None, status: Any = None) -> np.ndarray:
        """Simulate PortAudio asking for the next block to play."""
        count = frames if frames is not None else self.blocksize
        out = np.zeros((count, 1), dtype=np.float32)
        self.callback(out, count, None, status)
        return out[:, 0].copy()


def query_devices() -> list[dict[str, Any]]:
    return [dict(device) for device in DEVICES]


default = SimpleNamespace(device=(0, 1))


class PortAudioError(Exception):
    """Mirrors sounddevice.PortAudioError."""
