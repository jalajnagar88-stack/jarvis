"""Speech-to-text interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jarvis.interfaces.audio import AudioClip


@dataclass(frozen=True, slots=True)
class Transcript:
    """The result of transcribing one utterance."""

    text: str
    language: str | None = None
    confidence: float | None = None
    """Average token log-probability, when the engine reports one. Not a
    probability — more negative means less certain."""
    duration_seconds: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class Transcriber(ABC):
    """Turns an audio clip into text, locally.

    Loading is deferred: constructing a transcriber must be cheap, so the health
    check can ask about it without pulling a model into memory. :meth:`load`
    does the expensive part.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Idempotent.

        Raises:
            ModelMissingError: the model has not been downloaded.
            DependencyMissingError: the engine's package is not installed.
        """

    @abstractmethod
    def transcribe(self, clip: AudioClip) -> Transcript:
        """Transcribe one utterance. Calls :meth:`load` if needed."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...
