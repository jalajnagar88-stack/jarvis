"""Audio capture and playback interfaces.

The common currency between stages is a mono ``numpy`` array of ``float32``
samples in ``[-1.0, 1.0]``, tagged with its sample rate. Every implementation
converts to that on the way in and from it on the way out, so the wake word
detector never needs to know whether the source was a microphone, a WAV file,
or a test fixture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType

import numpy as np
import numpy.typing as npt

Samples = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AudioClip:
    """A finite stretch of mono audio."""

    samples: Samples
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples)) / self.sample_rate if self.sample_rate else 0.0

    def rms(self) -> float:
        """Root-mean-square amplitude — the level used for silence detection."""
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One audio device, as reported by the backend."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False
    is_default_output: bool = False


class AudioInput(ABC):
    """A microphone, or anything that can pretend to be one.

    Implementations must be usable as a context manager and must yield
    fixed-size blocks so the wake word detector sees the chunk length it was
    trained on.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def block_size(self) -> int:
        """Frames per block yielded by :meth:`blocks`."""

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin buffering.

        Raises:
            AudioDeviceError: no usable input device, or it could not be opened.
        """

    @abstractmethod
    def stop(self) -> None:
        """Close the device. Must be idempotent."""

    @abstractmethod
    def blocks(self) -> Iterator[Samples]:
        """Yield successive blocks of ``block_size`` mono float32 frames.

        Blocks until audio is available. The iterator ends when :meth:`stop`
        is called.
        """

    def __enter__(self) -> AudioInput:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


class AudioOutput(ABC):
    """A speaker, or anything that can pretend to be one.

    :meth:`play` must not block the caller's event loop for the duration of the
    clip — milestone 6 needs to cut playback the instant the user speaks, and it
    cannot do that if the loop is parked inside a blocking write.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @abstractmethod
    def start(self) -> None:
        """Open the device and prepare for playback.

        Raises:
            AudioDeviceError: no usable output device, or it could not be opened.
        """

    @abstractmethod
    def stop(self) -> None:
        """Close the device. Must be idempotent."""

    @abstractmethod
    def play(self, clip: AudioClip) -> None:
        """Queue a clip for playback and return immediately."""

    @abstractmethod
    def cancel(self) -> None:
        """Stop playing *now* and discard anything queued.

        This is the barge-in path. It must take effect within one audio block,
        not at the end of the current sentence.
        """

    @abstractmethod
    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Block until the queue drains or ``timeout`` elapses.

        Returns:
            True if playback finished, False if it timed out or was cancelled.
        """

    @property
    @abstractmethod
    def is_playing(self) -> bool: ...

    def __enter__(self) -> AudioOutput:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
