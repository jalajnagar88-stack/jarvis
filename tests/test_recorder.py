"""Record-until-silence.

This is the component most likely to feel wrong in daily use: too eager and it
truncates you, too patient and you stand there waiting. The thresholds are
config, but the state machine around them is here.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.audio.recorder import (
    PreRollBuffer,
    RecordingOutcome,
    UtteranceRecorder,
    block_rms,
)
from jarvis.config import SilenceConfig
from jarvis.interfaces.audio import Samples

from .conftest import silence_block, speech_block

SAMPLE_RATE = 16_000
BLOCK = 1280  # 80 ms


@pytest.fixture
def silence_cfg() -> SilenceConfig:
    return SilenceConfig(
        threshold=0.015,
        duration_seconds=0.8,
        max_utterance_seconds=20.0,
        min_utterance_seconds=0.4,
        no_speech_timeout_seconds=3.0,
        preroll_seconds=0.25,
    )


def feed_all(recorder: UtteranceRecorder, blocks: list[Samples]) -> RecordingOutcome:
    outcome = RecordingOutcome.RECORDING
    for block in blocks:
        outcome = recorder.feed(block)
        if outcome is not RecordingOutcome.RECORDING:
            break
    return outcome


class TestBlockRms:
    def test_silence_is_zero(self) -> None:
        assert block_rms(silence_block()) == 0.0

    def test_empty_block_is_zero(self) -> None:
        assert block_rms(np.zeros(0, dtype=np.float32)) == 0.0

    def test_constant_signal_equals_its_magnitude(self) -> None:
        assert block_rms(np.full(100, 0.25, dtype=np.float32)) == pytest.approx(0.25)

    def test_speech_clears_the_default_threshold(self) -> None:
        assert block_rms(speech_block()) > 0.015


class TestNormalUtterance:
    def test_speech_then_a_pause_completes(
        self, silence_cfg: SilenceConfig, utterance: list[Samples]
    ) -> None:
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        assert feed_all(recorder, utterance) is RecordingOutcome.COMPLETE
        assert recorder.result().usable

    def test_it_does_not_stop_during_a_short_pause(self, silence_cfg: SilenceConfig) -> None:
        """A comma is not the end of a sentence."""
        blocks = (
            [speech_block(seed=i) for i in range(6)]
            + [silence_block()] * 5  # 400 ms -- half the 800 ms threshold
            + [speech_block(seed=i) for i in range(6, 12)]
        )
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        assert feed_all(recorder, blocks) is RecordingOutcome.RECORDING

    def test_the_whole_utterance_is_captured(
        self, silence_cfg: SilenceConfig, utterance: list[Samples]
    ) -> None:
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        feed_all(recorder, utterance)
        clip = recorder.result().clip
        # 11 blocks of lead-in and speech, plus the 10 silent blocks that made
        # up the 800 ms pause before it stopped.
        assert clip.samples.size >= 11 * BLOCK
        assert clip.sample_rate == SAMPLE_RATE


class TestPreRoll:
    def test_preroll_audio_is_prepended(self, silence_cfg: SilenceConfig) -> None:
        preroll = [speech_block(seed=99) for _ in range(3)]
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg, preroll=preroll)
        recorder.feed(speech_block())
        clip = recorder.result().clip
        assert clip.samples.size == 4 * BLOCK
        assert np.allclose(clip.samples[:BLOCK], preroll[0])

    def test_preroll_does_not_count_as_the_user_speaking(self, silence_cfg: SilenceConfig) -> None:
        """Otherwise a loud pre-roll would satisfy min_utterance on its own."""
        preroll = [speech_block(seed=99) for _ in range(6)]
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg, preroll=preroll)
        outcome = feed_all(recorder, [silence_block()] * 40)
        assert outcome is RecordingOutcome.NO_SPEECH


class TestFalseWake:
    def test_pure_silence_gives_up_early(self, silence_cfg: SilenceConfig) -> None:
        """A false wake must not make you wait out max_utterance_seconds."""
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        outcome = feed_all(recorder, [silence_block()] * 200)
        assert outcome is RecordingOutcome.NO_SPEECH
        # 3 s of no-speech timeout, not the 20 s ceiling.
        assert recorder.elapsed_seconds == pytest.approx(3.0, abs=0.1)

    def test_a_cough_is_not_a_command(self, silence_cfg: SilenceConfig) -> None:
        """One loud block is 80 ms -- below the 400 ms minimum."""
        blocks = [speech_block()] + [silence_block()] * 20
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        assert feed_all(recorder, blocks) is RecordingOutcome.NO_SPEECH
        assert not recorder.result().usable


class TestLimits:
    def test_a_long_utterance_is_capped_but_kept(self, silence_cfg: SilenceConfig) -> None:
        silence_cfg.max_utterance_seconds = 1.0
        blocks = [speech_block(seed=i) for i in range(50)]
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        assert feed_all(recorder, blocks) is RecordingOutcome.TOO_LONG
        # A rambling question is still a question: keep the audio.
        assert recorder.result().usable

    def test_a_stuck_microphone_does_not_record_forever(self, silence_cfg: SilenceConfig) -> None:
        silence_cfg.max_utterance_seconds = 2.0
        silence_cfg.duration_seconds = 99.0  # never satisfied
        blocks = [speech_block(seed=i) for i in range(1000)]
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        assert feed_all(recorder, blocks) is RecordingOutcome.TOO_LONG
        assert recorder.elapsed_seconds <= 2.1

    def test_feeding_after_completion_is_inert(
        self, silence_cfg: SilenceConfig, utterance: list[Samples]
    ) -> None:
        recorder = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        feed_all(recorder, utterance)
        size_before = recorder.result().clip.samples.size
        recorder.feed(speech_block())
        assert recorder.result().clip.samples.size == size_before


class TestThresholdSensitivity:
    def test_raising_the_threshold_makes_quiet_speech_inaudible(
        self, silence_cfg: SilenceConfig
    ) -> None:
        """The knob a user in a noisy room will reach for; prove it does something."""
        quiet = [speech_block(level=0.02, seed=i) for i in range(10)] + [silence_block()] * 45
        heard = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        assert feed_all(heard, quiet) is RecordingOutcome.COMPLETE

        silence_cfg.threshold = 0.1
        ignored = UtteranceRecorder(sample_rate=SAMPLE_RATE, silence=silence_cfg)
        assert feed_all(ignored, quiet) is RecordingOutcome.NO_SPEECH


class TestPreRollBuffer:
    def test_keeps_only_the_most_recent_window(self) -> None:
        buffer = PreRollBuffer(sample_rate=SAMPLE_RATE, seconds=0.25, block_size=BLOCK)
        for index in range(20):
            buffer.push(np.full(BLOCK, index, dtype=np.float32))
        drained = buffer.drain()
        assert len(drained) == 3  # 0.25 s / 80 ms, rounded
        assert drained[-1][0] == 19.0

    def test_draining_empties_it(self) -> None:
        buffer = PreRollBuffer(sample_rate=SAMPLE_RATE, seconds=0.25, block_size=BLOCK)
        buffer.push(speech_block())
        buffer.drain()
        assert buffer.drain() == []

    def test_zero_seconds_disables_it(self) -> None:
        buffer = PreRollBuffer(sample_rate=SAMPLE_RATE, seconds=0.0, block_size=BLOCK)
        buffer.push(speech_block())
        assert buffer.drain() == []
