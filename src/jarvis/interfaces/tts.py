"""Text-to-speech interface.

Two implementations sit behind this: Piper (offline, the default) and
ElevenLabs (cloud, opt-in). The agent never learns which one it is talking to —
that is the whole point of the interface. Note the privacy asymmetry, which is
stated in the README: Piper sends nothing anywhere; ElevenLabs necessarily
sends the reply text to a third party.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from jarvis.interfaces.audio import AudioClip


class SpeechSynthesizer(ABC):
    """Turns text into audio.

    :meth:`synthesize` is the simple path. :meth:`stream` exists because
    milestone 3 starts speaking on the first complete sentence rather than
    waiting for the model to finish — the caller feeds sentences in as they
    arrive and plays each clip as it comes back.
    """

    @abstractmethod
    def load(self) -> None:
        """Prepare the engine. Idempotent.

        Raises:
            ModelMissingError: the voice has not been downloaded.
            DependencyMissingError: the engine's package is not installed.
            AuthError: a cloud engine has no API key.
        """

    @abstractmethod
    def synthesize(self, text: str) -> AudioClip:
        """Render ``text`` to a single clip."""

    def stream(self, sentences: Iterator[str]) -> Iterator[AudioClip]:
        """Render each sentence as it arrives.

        The default implementation synthesises one sentence at a time, which is
        already enough to start playback early. An engine with a native
        streaming API should override this.
        """
        for sentence in sentences:
            stripped = sentence.strip()
            if stripped:
                yield self.synthesize(stripped)

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate of the audio this engine produces."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def sends_text_off_machine(self) -> bool:
        """True for cloud engines. The health check reports this prominently so
        the privacy posture is never a surprise."""
