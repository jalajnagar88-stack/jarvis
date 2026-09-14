"""Detecting that the user has started talking over JARVIS.

Harder than it looks, for one reason: while JARVIS is speaking, its own voice is
coming out of the speakers and straight back into the microphone. A plain
level threshold would hear that and cut itself off mid-word, every time.

Three things keep that from happening:

* **A raised threshold.** Interrupting is judged against a higher level than
  ordinary speech detection, because the microphone is already picking up the
  reply.
* **A minimum duration.** The level must hold for a fraction of a second, so a
  door closing or a single loud syllable of JARVIS's own output does not count.
* **A grace period.** The first moments of playback are ignored entirely
  (``interrupt.grace_seconds``), which covers the sharp onset of a sentence.

None of this is echo cancellation. It is the pragmatic version: with the speaker
at a sensible volume it works, and it fails in the safe direction -- a missed
interruption means waiting for the sentence to end, not JARVIS talking over
itself forever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from jarvis.audio.recorder import block_rms
from jarvis.config import InterruptConfig
from jarvis.interfaces.audio import Samples
from jarvis.logging_setup import get_logger

log = get_logger("audio.barge_in")


@dataclass(slots=True)
class BargeInDetector:
    """Decides whether the user is talking over the assistant.

    Fed audio blocks while JARVIS speaks. Returns True on the block where the
    interruption becomes certain, once and only once per utterance.
    """

    config: InterruptConfig
    sample_rate: int
    block_size: int

    _loud_seconds: float = 0.0
    _started_at: float | None = None
    _fired: bool = False

    def begin(self) -> None:
        """Called when playback starts."""
        self._loud_seconds = 0.0
        self._started_at = time.monotonic()
        self._fired = False

    def end(self) -> None:
        """Called when playback stops."""
        self._started_at = None
        self._loud_seconds = 0.0

    @property
    def armed(self) -> bool:
        """False during the grace period, or when interruption is switched off."""
        if not self.config.enabled or self._started_at is None or self._fired:
            return False
        return (time.monotonic() - self._started_at) >= self.config.grace_seconds

    def feed(self, block: Samples) -> bool:
        """Consume one block of microphone audio while JARVIS is speaking."""
        if not self.armed:
            return False

        duration = len(block) / self.sample_rate if self.sample_rate else 0.0
        if block_rms(block) >= self.config.threshold:
            self._loud_seconds += duration
        else:
            # Must be sustained, not intermittent. Anything else is the room.
            self._loud_seconds = 0.0

        if self._loud_seconds >= self.config.min_duration_seconds:
            self._fired = True
            log.info("Interrupted by the user after %.2fs of speech.", self._loud_seconds)
            return True
        return False
