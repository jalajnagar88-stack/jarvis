"""Wake word detection interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jarvis.interfaces.audio import Samples


@dataclass(frozen=True, slots=True)
class WakeEvent:
    """A wake word firing."""

    model: str
    score: float
    """Confidence in [0, 1] reported by the detector."""


class WakeWordDetector(ABC):
    """Streaming wake word detector.

    Fed one audio block at a time; returns an event on the block where the
    phrase completes. Implementations own their own cooldown so that a single
    spoken "hey Jarvis" produces exactly one event.
    """

    @abstractmethod
    def process(self, block: Samples) -> WakeEvent | None:
        """Consume one block of audio; return an event if the wake word fired."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state. Call after handling a wake so scores don't linger."""

    @property
    @abstractmethod
    def expected_block_size(self) -> int:
        """Block length in frames this detector expects. Feeding a different
        size is allowed but costs accuracy."""
