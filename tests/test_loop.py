"""The voice loop, end to end.

Driven entirely through the interfaces with synthetic audio, so this exercises
the real state machine -- wake, record, transcribe, reply, speak, recover --
without a microphone, a model, or an API key.
"""

from __future__ import annotations

import pytest

from jarvis.audio.recorder import RecordingOutcome
from jarvis.audit import NullAuditLog
from jarvis.config import Config
from jarvis.errors import ModelMissingError
from jarvis.interfaces.audio import AudioClip, Samples
from jarvis.interfaces.stt import Transcript
from jarvis.loop import Responder, TurnResult, VoiceLoop, echo_responder
from jarvis.state import State

from .conftest import (
    FakeAudioInput,
    FakeAudioOutput,
    FakeSynthesizer,
    FakeTranscriber,
    FakeWakeWord,
    silence_block,
    speech_block,
)


def build_loop(
    cfg: Config,
    blocks: list[Samples],
    *,
    wake_on: set[int] | None = None,
    transcripts: list[str] | None = None,
    responder: Responder = echo_responder,
    transcriber: FakeTranscriber | None = None,
    synthesizer: FakeSynthesizer | None = None,
) -> tuple[VoiceLoop, dict[str, object]]:
    parts: dict[str, object] = {
        "audio_in": FakeAudioInput(blocks),
        "audio_out": FakeAudioOutput(),
        "wake_word": FakeWakeWord(wake_on if wake_on is not None else {0}),
        "transcriber": transcriber or FakeTranscriber(transcripts),
        "synthesizer": synthesizer or FakeSynthesizer(),
    }
    states: list[State] = []
    turns: list[TurnResult] = []
    loop = VoiceLoop(
        cfg,
        audio_in=parts["audio_in"],  # type: ignore[arg-type]
        audio_out=parts["audio_out"],  # type: ignore[arg-type]
        wake_word=parts["wake_word"],  # type: ignore[arg-type]
        transcriber=parts["transcriber"],  # type: ignore[arg-type]
        synthesizer=parts["synthesizer"],  # type: ignore[arg-type]
        audit=NullAuditLog(),
        responder=responder,
        on_state=states.append,
        on_turn=turns.append,
    )
    parts["states"] = states
    parts["turns"] = turns
    return loop, parts


@pytest.fixture
def command_audio(utterance: list[Samples]) -> list[Samples]:
    """A wake block, then a spoken command, then quiet."""
    return [silence_block(), *utterance]


class TestHappyPath:
    def test_wake_record_transcribe_speak(self, cfg: Config, command_audio: list[Samples]) -> None:
        loop, parts = build_loop(cfg, command_audio, transcripts=["what time is it"])
        loop.run()

        synth = parts["synthesizer"]
        assert isinstance(synth, FakeSynthesizer)
        assert synth.spoken == ["what time is it"]

    def test_it_visits_every_state_in_order(
        self, cfg: Config, command_audio: list[Samples]
    ) -> None:
        loop, parts = build_loop(cfg, command_audio)
        loop.run()
        assert parts["states"] == [
            State.LISTENING,
            State.THINKING,
            State.SPEAKING,
            State.IDLE,
            State.STOPPED,
        ]

    def test_the_reply_is_played(self, cfg: Config, command_audio: list[Samples]) -> None:
        loop, parts = build_loop(cfg, command_audio, transcripts=["hello"])
        loop.run()
        out = parts["audio_out"]
        assert isinstance(out, FakeAudioOutput)
        assert len(out.played) == 1
        assert out.played[0].samples.size > 0

    def test_the_transcriber_receives_the_recorded_audio(
        self, cfg: Config, command_audio: list[Samples]
    ) -> None:
        transcriber = FakeTranscriber(["hello"])
        loop, _ = build_loop(cfg, command_audio, transcriber=transcriber)
        loop.run()
        assert len(transcriber.calls) == 1
        assert transcriber.calls[0].duration_seconds > 0.5

    def test_it_reports_the_turn(self, cfg: Config, command_audio: list[Samples]) -> None:
        loop, parts = build_loop(cfg, command_audio, transcripts=["good evening"])
        loop.run()
        turns = parts["turns"]
        assert isinstance(turns, list)
        assert len(turns) == 1
        assert turns[0].transcript is not None
        assert turns[0].transcript.text == "good evening"
        assert turns[0].reply == "good evening"
        assert turns[0].outcome is RecordingOutcome.COMPLETE


class TestMilestoneTwoBehaviour:
    def test_the_echo_responder_speaks_the_transcript_verbatim(self) -> None:
        """Echoing is what proves the audio path independently of the brain."""
        assert echo_responder("set a timer for ten minutes") == "set a timer for ten minutes"

    def test_a_custom_responder_is_all_milestone_3_needs(
        self, cfg: Config, command_audio: list[Samples]
    ) -> None:
        loop, parts = build_loop(
            cfg,
            command_audio,
            transcripts=["hello"],
            responder=lambda text: f"Good evening. You said: {text}",
        )
        loop.run()
        synth = parts["synthesizer"]
        assert isinstance(synth, FakeSynthesizer)
        assert synth.spoken == ["Good evening. You said: hello"]


class TestIdleBehaviour:
    def test_nothing_happens_without_the_wake_word(self, cfg: Config) -> None:
        blocks = [speech_block(seed=i) for i in range(20)]
        loop, parts = build_loop(cfg, blocks, wake_on=set())
        loop.run()
        synth = parts["synthesizer"]
        assert isinstance(synth, FakeSynthesizer)
        assert synth.spoken == []
        assert parts["turns"] == []

    def test_audio_before_the_wake_word_is_not_recorded(self, cfg: Config) -> None:
        """Only the pre-roll window survives; the rest is discarded."""
        blocks = [speech_block(seed=i) for i in range(30)]
        blocks += [speech_block(seed=100 + i) for i in range(10)] + [silence_block()] * 15
        transcriber = FakeTranscriber(["hello"])
        loop, _ = build_loop(cfg, blocks, wake_on={29}, transcriber=transcriber)
        loop.run()
        # 0.25 s pre-roll + ~0.8 s speech + the 0.8 s pause, not 30 blocks of history.
        assert transcriber.calls[0].duration_seconds < 2.5


class TestFalseWake:
    def test_a_wake_with_no_speech_does_not_reach_the_transcriber(self, cfg: Config) -> None:
        transcriber = FakeTranscriber(["should not happen"])
        blocks = [silence_block()] * 60
        loop, parts = build_loop(cfg, blocks, wake_on={0}, transcriber=transcriber)
        loop.run()
        assert transcriber.calls == []
        synth = parts["synthesizer"]
        assert isinstance(synth, FakeSynthesizer)
        assert synth.spoken == []

    def test_it_returns_to_idle_after_a_false_wake(self, cfg: Config) -> None:
        blocks = [silence_block()] * 60
        loop, _ = build_loop(cfg, blocks, wake_on={0})
        loop.run()
        assert loop.state is State.STOPPED  # via IDLE

    def test_an_empty_transcript_is_not_spoken(
        self, cfg: Config, command_audio: list[Samples]
    ) -> None:
        """Whisper hallucinating nothing out of noise must not become speech."""
        loop, parts = build_loop(cfg, command_audio, transcripts=[""])
        loop.run()
        synth = parts["synthesizer"]
        assert isinstance(synth, FakeSynthesizer)
        assert synth.spoken == []


class TestRecovery:
    def test_stale_audio_is_flushed_before_listening_again(
        self, cfg: Config, command_audio: list[Samples]
    ) -> None:
        """Otherwise the wake detector hears JARVIS's own reply and re-triggers."""
        loop, parts = build_loop(cfg, command_audio)
        loop.run()
        audio_in = parts["audio_in"]
        assert isinstance(audio_in, FakeAudioInput)
        assert audio_in.flush_calls >= 1

    def test_the_wake_detector_is_reset_after_each_turn(
        self, cfg: Config, command_audio: list[Samples]
    ) -> None:
        loop, parts = build_loop(cfg, command_audio)
        loop.run()
        wake = parts["wake_word"]
        assert isinstance(wake, FakeWakeWord)
        assert wake.resets >= 1

    def test_a_transcription_failure_is_spoken_not_raised(self, cfg: Config) -> None:
        class BrokenTranscriber(FakeTranscriber):
            def transcribe(self, clip: AudioClip) -> Transcript:
                raise ModelMissingError("model is gone", remedy="download it")

        loop, parts = build_loop(
            cfg,
            [silence_block(), *_utterance()],
            transcriber=BrokenTranscriber(),
        )
        loop.run()  # must not raise
        synth = parts["synthesizer"]
        assert isinstance(synth, FakeSynthesizer)
        assert synth.spoken == ["I could not transcribe that."]

    def test_a_broken_state_listener_does_not_stop_the_assistant(
        self, cfg: Config, command_audio: list[Samples]
    ) -> None:
        """A crashing status window is a cosmetic problem, not a fatal one."""
        loop, parts = build_loop(cfg, command_audio)
        loop._on_state = lambda state: (_ for _ in ()).throw(RuntimeError("ui exploded"))
        loop.run()
        synth = parts["synthesizer"]
        assert isinstance(synth, FakeSynthesizer)
        assert synth.spoken


class TestMultipleTurns:
    def test_two_commands_in_one_session(self, cfg: Config) -> None:
        blocks = [silence_block(), *_utterance(), silence_block(), *_utterance()]
        # FakeWakeWord counts only the blocks it is actually shown, and it is
        # bypassed entirely while recording. So index 1 is the first block after
        # the first turn finishes, not the second block of the stream.
        loop, parts = build_loop(
            cfg,
            blocks,
            wake_on={0, 1},
            transcripts=["first question", "second question"],
        )
        loop.run()
        synth = parts["synthesizer"]
        assert isinstance(synth, FakeSynthesizer)
        assert synth.spoken == ["first question", "second question"]


class TestStopping:
    def test_stop_cancels_playback(self, cfg: Config) -> None:
        loop, parts = build_loop(cfg, [silence_block()] * 5, wake_on=set())
        loop.stop()
        out = parts["audio_out"]
        assert isinstance(out, FakeAudioOutput)
        assert out.cancels == 1

    def test_the_loop_ends_in_the_stopped_state(self, cfg: Config) -> None:
        loop, _ = build_loop(cfg, [silence_block()] * 5, wake_on=set())
        loop.run()
        assert loop.state is State.STOPPED

    def test_speak_ignores_blank_text(self, cfg: Config) -> None:
        loop, parts = build_loop(cfg, [], wake_on=set())
        loop.speak("   ")
        out = parts["audio_out"]
        assert isinstance(out, FakeAudioOutput)
        assert out.played == []


def _utterance() -> list[Samples]:
    return [speech_block(seed=i) for i in range(10)] + [silence_block()] * 15
