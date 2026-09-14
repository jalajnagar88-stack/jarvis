"""Record one utterance, stopping when the speaker stops.

This is the part of a voice assistant that most obviously feels wrong when it
is wrong: too eager and it truncates you mid-sentence, too patient and it sits
there while you wait. The thresholds are all in ``config.yaml`` under
``audio.silence`` because the right values depend on the room.

The recorder is fed one block at a time and is deliberately free of any audio
backend, so the tests drive it with synthetic arrays.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from jarvis.config import SilenceConfig
from jarvis.interfaces.audio import AudioClip, Samples
from jarvis.logging_setup import get_logger

log = get_logger("recorder")


class RecordingOutcome(StrEnum):
    """Why the recorder stopped."""

    RECORDING = "recording"
    """Still going."""

    COMPLETE = "complete"
    """Speech, then a clear pause. The normal, happy ending."""

    NO_SPEECH = "no_speech"
    """The wake word fired but nothing was said. Usually a false wake."""

    TOO_LONG = "too_long"
    """Hit the hard ceiling. The audio is kept -- a long question is still a
    question -- but the caller may want to mention it was cut off."""


@dataclass(frozen=True, slots=True)
class Recording:
    clip: AudioClip
    outcome: RecordingOutcome
    speech_seconds: float

    @property
    def usable(self) -> bool:
        return self.outcome in (RecordingOutcome.COMPLETE, RecordingOutcome.TOO_LONG)


def block_rms(block: Samples) -> float:
    """Root-mean-square amplitude of one block."""
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


class UtteranceRecorder:
    """Accumulates audio blocks until the speaker falls silent.

    Feed it with :meth:`feed`, which returns the current outcome. Anything other
    than ``RECORDING`` means it is finished and :meth:`result` is ready.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        silence: SilenceConfig,
        preroll: list[Samples] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._silence = silence
        self._blocks: list[Samples] = list(preroll or [])
        self._speech_seconds = 0.0
        self._trailing_silence = 0.0
        self._elapsed = 0.0
        self._heard_speech = False
        self._outcome = RecordingOutcome.RECORDING

        # Pre-roll audio is kept in the clip but never counted as the user
        # speaking: it predates the wake word, and is often the wake word itself.
        self._preroll_seconds = sum(len(b) for b in self._blocks) / sample_rate

    @property
    def outcome(self) -> RecordingOutcome:
        return self._outcome

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed

    def feed(self, block: Samples) -> RecordingOutcome:
        """Consume one block and report whether recording should continue."""
        if self._outcome is not RecordingOutcome.RECORDING:
            return self._outcome

        self._blocks.append(block)
        duration = len(block) / self._sample_rate
        self._elapsed += duration

        if block_rms(block) >= self._silence.threshold:
            self._heard_speech = True
            self._speech_seconds += duration
            self._trailing_silence = 0.0
        else:
            self._trailing_silence += duration

        if self._elapsed >= self._silence.max_utterance_seconds:
            self._outcome = (
                RecordingOutcome.TOO_LONG if self._heard_speech else RecordingOutcome.NO_SPEECH
            )
            return self._outcome

        if not self._heard_speech:
            # Nothing yet. Give up early rather than making the user stand
            # through the full max_utterance_seconds after a false wake.
            if self._trailing_silence >= self._silence.no_speech_timeout_seconds:
                self._outcome = RecordingOutcome.NO_SPEECH
            return self._outcome

        if self._trailing_silence >= self._silence.duration_seconds:
            # Measure the speech, not the wall clock. Counting elapsed time here
            # would let an 80 ms cough plus the 800 ms pause that follows it
            # clear a 400 ms minimum, which is exactly backwards.
            if self._speech_seconds >= self._silence.min_utterance_seconds:
                self._outcome = RecordingOutcome.COMPLETE
            else:
                # Too short to be a command: a cough, a door, a stray consonant.
                self._outcome = RecordingOutcome.NO_SPEECH
        return self._outcome

    def result(self) -> Recording:
        """The recording so far. Safe to call before it has finished."""
        samples = (
            np.concatenate(self._blocks).astype(np.float32)
            if self._blocks
            else np.zeros(0, dtype=np.float32)
        )
        clip = AudioClip(samples=samples, sample_rate=self._sample_rate)
        return Recording(clip=clip, outcome=self._outcome, speech_seconds=self._speech_seconds)


class PreRollBuffer:
    """A short rolling window of the most recent audio.

    The wake word fires at the *end* of "hey Jarvis", by which point a quick
    speaker is already a syllable into the command. Seeding the recorder with
    the preceding fraction of a second recovers that.
    """

    def __init__(self, *, sample_rate: int, seconds: float, block_size: int) -> None:
        blocks = max(0, round(seconds * sample_rate / block_size)) if block_size else 0
        self._buffer: deque[Samples] = deque(maxlen=blocks) if blocks else deque(maxlen=1)
        self._enabled = blocks > 0

    def push(self, block: Samples) -> None:
        if self._enabled:
            self._buffer.append(block)

    def drain(self) -> list[Samples]:
        """Return and clear the buffered blocks, oldest first."""
        if not self._enabled:
            return []
        blocks = list(self._buffer)
        self._buffer.clear()
        return blocks
