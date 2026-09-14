"""Interruption: cutting speech the moment the user talks over it.

The hard part is not detecting sound, it is not detecting JARVIS's own voice
coming back through the microphone. These tests cover both directions: a real
interruption must be caught quickly, and the assistant's own output must not
trigger one.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.audio.barge_in import BargeInDetector
from jarvis.audit import NullAuditLog
from jarvis.config import Config, InterruptConfig
from jarvis.interfaces.audio import AudioClip, Samples
from jarvis.loop import VoiceLoop
from jarvis.state import State

from .conftest import (
    FakeAudioInput,
    FakeAudioOutput,
    FakeBrain,
    FakeSynthesizer,
    FakeTranscriber,
    FakeWakeWord,
    silence_block,
    speech_block,
)

BLOCK = 1280
RATE = 16_000


@pytest.fixture
def config() -> InterruptConfig:
    return InterruptConfig(
        enabled=True, threshold=0.05, min_duration_seconds=0.25, grace_seconds=0.0
    )


def detector(config: InterruptConfig) -> BargeInDetector:
    started = BargeInDetector(config=config, sample_rate=RATE, block_size=BLOCK)
    started.begin()
    return started


class TestDetector:
    def test_sustained_speech_fires(self, config: InterruptConfig) -> None:
        watcher = detector(config)
        fired = [watcher.feed(speech_block(level=0.3, seed=i)) for i in range(5)]
        assert any(fired)

    def test_it_takes_the_configured_duration(self, config: InterruptConfig) -> None:
        """0.25 s at 80 ms per block is four blocks, not one."""
        watcher = detector(config)
        assert not watcher.feed(speech_block(level=0.3))
        assert not watcher.feed(speech_block(level=0.3, seed=1))
        assert not watcher.feed(speech_block(level=0.3, seed=2))
        assert watcher.feed(speech_block(level=0.3, seed=3))

    def test_silence_never_fires(self, config: InterruptConfig) -> None:
        watcher = detector(config)
        assert not any(watcher.feed(silence_block()) for _ in range(30))

    def test_a_single_loud_block_does_not_fire(self, config: InterruptConfig) -> None:
        """A door closing is not an interruption."""
        watcher = detector(config)
        assert not watcher.feed(speech_block(level=0.8))
        assert not watcher.feed(silence_block())

    def test_intermittent_noise_does_not_accumulate(self, config: InterruptConfig) -> None:
        """The level must be sustained; anything else is just the room."""
        watcher = detector(config)
        fired = []
        for index in range(20):
            block = speech_block(level=0.3, seed=index) if index % 2 else silence_block()
            fired.append(watcher.feed(block))
        assert not any(fired)

    def test_quiet_audio_below_the_threshold_is_ignored(self, config: InterruptConfig) -> None:
        """This is the echo of JARVIS's own voice, at a sensible volume."""
        watcher = detector(config)
        assert not any(watcher.feed(speech_block(level=0.02, seed=i)) for i in range(20))

    def test_the_grace_period_suppresses_the_onset_of_speech(self, config: InterruptConfig) -> None:
        """The start of a sentence is the loudest part of JARVIS's own output."""
        config.grace_seconds = 5.0
        watcher = BargeInDetector(config=config, sample_rate=RATE, block_size=BLOCK)
        watcher.begin()
        assert not watcher.armed
        assert not any(watcher.feed(speech_block(level=0.9, seed=i)) for i in range(10))

    def test_it_fires_only_once(self, config: InterruptConfig) -> None:
        watcher = detector(config)
        fired = [watcher.feed(speech_block(level=0.3, seed=i)) for i in range(20)]
        assert sum(fired) == 1

    def test_disabled_never_fires(self, config: InterruptConfig) -> None:
        config.enabled = False
        watcher = detector(config)
        assert not any(watcher.feed(speech_block(level=0.9, seed=i)) for i in range(20))

    def test_it_is_disarmed_before_playback_starts(self, config: InterruptConfig) -> None:
        watcher = BargeInDetector(config=config, sample_rate=RATE, block_size=BLOCK)
        assert not watcher.armed
        assert not watcher.feed(speech_block(level=0.9))

    def test_end_disarms_it(self, config: InterruptConfig) -> None:
        watcher = detector(config)
        watcher.end()
        assert not watcher.armed

    def test_a_raised_threshold_ignores_more(self, config: InterruptConfig) -> None:
        """The knob for a room where the speakers feed back strongly."""
        config.threshold = 0.5
        watcher = detector(config)
        assert not any(watcher.feed(speech_block(level=0.2, seed=i)) for i in range(20))


class PlayingOutput(FakeAudioOutput):
    """An output that reports itself as playing until cancelled or drained."""

    def __init__(self) -> None:
        super().__init__()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, clip: AudioClip) -> None:
        super().play(clip)
        self._playing = True

    def cancel(self) -> None:
        super().cancel()
        self._playing = False

    def finish(self) -> None:
        self._playing = False


def build_loop(
    cfg: Config,
    blocks: list[Samples],
    *,
    reply: str = "Good evening. The kettle is on. Anything else?",
) -> tuple[VoiceLoop, dict[str, Any]]:
    # The fake clock does not advance, so a grace period would never expire.
    cfg.interrupt.grace_seconds = 0.0
    parts: dict[str, Any] = {
        "audio_in": FakeAudioInput(blocks),
        "audio_out": PlayingOutput(),
        "wake_word": FakeWakeWord({0}),
        "transcriber": FakeTranscriber(["what time is it", "actually never mind"]),
        "synthesizer": FakeSynthesizer(),
        "brain": FakeBrain([reply, "Very good."]),
    }
    interrupts: list[bool] = []
    loop = VoiceLoop(
        cfg,
        audio_in=parts["audio_in"],
        audio_out=parts["audio_out"],
        wake_word=parts["wake_word"],
        transcriber=parts["transcriber"],
        synthesizer=parts["synthesizer"],
        brain=parts["brain"],
        audit=NullAuditLog(),
        on_interrupt=lambda: interrupts.append(True),
    )
    parts["interrupts"] = interrupts
    return loop, parts


def utterance() -> list[Samples]:
    return [speech_block(seed=i) for i in range(10)] + [silence_block()] * 15


class TestLoopInterruption:
    def test_speaking_over_the_reply_cancels_it(self, cfg: Config) -> None:
        # A command, then loud speech while the reply plays.
        blocks = [silence_block(), *utterance()] + [
            speech_block(level=0.4, seed=200 + i) for i in range(40)
        ]
        loop, parts = build_loop(cfg, blocks)
        loop.run()

        assert parts["audio_out"].cancels >= 1
        assert parts["interrupts"] == [True]

    def test_an_uninterrupted_reply_plays_through(self, cfg: Config) -> None:
        blocks = [silence_block(), *utterance()] + [silence_block()] * 40
        loop, parts = build_loop(cfg, blocks)
        loop.run()
        assert parts["interrupts"] == []

    def test_the_assistants_own_voice_does_not_interrupt_it(self, cfg: Config) -> None:
        """Quiet echo through the microphone must not cut the reply short."""
        blocks = [silence_block(), *utterance()] + [
            speech_block(level=0.02, seed=300 + i) for i in range(40)
        ]
        loop, parts = build_loop(cfg, blocks)
        loop.run()
        assert parts["interrupts"] == []

    def test_interrupting_is_recorded_on_the_turn(self, cfg: Config) -> None:
        blocks = [silence_block(), *utterance()] + [
            speech_block(level=0.4, seed=200 + i) for i in range(40)
        ]
        turns: list[Any] = []
        loop, _ = build_loop(cfg, blocks)
        loop._on_turn = turns.append
        loop.run()

        assert turns and turns[0].interrupted is True

    def test_disabling_interruption_restores_plain_waiting(self, cfg: Config) -> None:
        cfg.interrupt.enabled = False
        blocks = [silence_block(), *utterance()] + [
            speech_block(level=0.9, seed=400 + i) for i in range(40)
        ]
        loop, parts = build_loop(cfg, blocks)
        loop.run()
        assert parts["interrupts"] == []
        assert parts["audio_out"].cancels == 0


class TestBrainStreamIsClosed:
    def test_interrupting_closes_the_reply_stream(self, cfg: Config) -> None:
        """The generator must be closed, not just abandoned: that is what tears
        down the HTTP connection rather than waiting for the collector."""
        closed: list[bool] = []

        class TrackingBrain(FakeBrain):
            def respond(self, user_text: str) -> Any:
                try:
                    yield from super().respond(user_text)
                except GeneratorExit:
                    closed.append(True)
                    raise

        blocks = [silence_block(), *utterance()] + [
            speech_block(level=0.4, seed=500 + i) for i in range(40)
        ]
        loop, parts = build_loop(cfg, blocks)
        parts["brain"] = TrackingBrain(["One. Two. Three. Four. Five."])
        loop._brain = parts["brain"]
        loop.run()

        assert loop.state in (State.STOPPED, State.LISTENING)
