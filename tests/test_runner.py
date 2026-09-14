"""Starting the assistant: the voice loop, the text REPL, and `say`.

The factory is patched so no model is ever loaded, which lets these tests cover
the parts users actually touch: what happens when something is missing, and
whether the failure reads like an instruction or like a stack trace.
"""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from jarvis import factory, runner
from jarvis.config import Config
from jarvis.errors import AudioDeviceError, ModelMissingError
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


@pytest.fixture
def console() -> Console:
    # force_terminal=False keeps the captured output free of escape codes.
    return Console(force_terminal=False, width=100)


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the factory so nothing loads a model or opens a device."""
    parts: dict[str, Any] = {
        "audio_in": FakeAudioInput([silence_block(), *_utterance()]),
        "audio_out": FakeAudioOutput(),
        "wake_word": FakeWakeWord({0}),
        "transcriber": FakeTranscriber(["what time is it"]),
        "synthesizer": FakeSynthesizer(),
    }
    monkeypatch.setattr(factory, "build_audio_input", lambda cfg: parts["audio_in"])
    monkeypatch.setattr(factory, "build_audio_output", lambda cfg, sample_rate: parts["audio_out"])
    monkeypatch.setattr(factory, "build_wake_word", lambda cfg: parts["wake_word"])
    monkeypatch.setattr(factory, "build_transcriber", lambda cfg: parts["transcriber"])
    monkeypatch.setattr(factory, "build_synthesizer", lambda cfg: parts["synthesizer"])
    return parts


def _utterance() -> list[Any]:
    return [speech_block(seed=i) for i in range(10)] + [silence_block()] * 15


def scripted_input(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    """Feed the REPL a fixed script, then EOF."""
    queue = list(lines)

    def fake_input(self: Console, prompt: str = "") -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr(Console, "input", fake_input)


class TestVoiceMode:
    def test_a_full_turn_runs_and_exits_cleanly(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert runner.run_voice(cfg, console) == 0
        assert wired["synthesizer"].spoken == ["what time is it"]

    def test_the_prompt_spells_the_wake_word_readably(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`hey_jarvis` is a filename; "hey jarvis" is what you say."""
        runner.run_voice(cfg, console)
        assert 'Say "hey jarvis"' in capsys.readouterr().out

    def test_the_transcript_and_reply_are_printed(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner.run_voice(cfg, console)
        out = capsys.readouterr().out
        assert "what time is it" in out

    def test_devices_are_opened_and_closed(
        self, cfg: Config, console: Console, wired: dict[str, Any]
    ) -> None:
        runner.run_voice(cfg, console)
        assert wired["audio_in"].started and wired["audio_in"].stopped
        assert wired["audio_out"].started and wired["audio_out"].stopped

    def test_a_missing_model_is_reported_not_raised(
        self,
        cfg: Config,
        console: Console,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def explode(config: Config) -> Any:
            raise ModelMissingError("The voice is missing.", remedy="download it")

        monkeypatch.setattr(factory, "build_synthesizer", explode)
        assert runner.run_voice(cfg, console) == 1
        out = capsys.readouterr().out
        assert "The voice is missing." in out
        assert "download it" in out
        assert "Traceback" not in out

    def test_a_dead_microphone_is_reported_not_raised(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def explode(config: Config) -> Any:
            raise AudioDeviceError("No microphone.", remedy="plug one in")

        monkeypatch.setattr(factory, "build_audio_input", explode)
        assert runner.run_voice(cfg, console) == 1
        assert "No microphone." in capsys.readouterr().out

    def test_a_low_confidence_transcript_is_flagged_to_the_user(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Better to admit uncertainty than to present a guess as a quotation."""
        wired["transcriber"] = FakeTranscriber(["mumble"], confidence=-1.5)
        runner.run_voice(cfg, console)
        assert "low confidence" in capsys.readouterr().out

    def test_a_confident_transcript_is_not_flagged(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        wired["transcriber"] = FakeTranscriber(["clear as day"], confidence=-0.1)
        runner.run_voice(cfg, console)
        assert "low confidence" not in capsys.readouterr().out


class TestTextMode:
    def test_a_typed_line_is_answered(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        scripted_input(monkeypatch, ["hello there"])
        assert runner.run_text(cfg, console) == 0
        assert "hello there" in capsys.readouterr().out

    def test_replies_are_also_spoken(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Text mode doubles as the way to audition the voice."""
        scripted_input(monkeypatch, ["good evening"])
        runner.run_text(cfg, console)
        assert wired["synthesizer"].spoken == ["good evening"]

    def test_quit_leaves(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scripted_input(monkeypatch, ["/quit", "never reached"])
        runner.run_text(cfg, console)
        assert wired["synthesizer"].spoken == []

    def test_blank_lines_are_ignored(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scripted_input(monkeypatch, ["", "   ", "real input"])
        runner.run_text(cfg, console)
        assert wired["synthesizer"].spoken == ["real input"]

    def test_a_custom_responder_is_used(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scripted_input(monkeypatch, ["ping"])
        runner.run_text(cfg, console, responder=lambda text: f"pong: {text}")
        assert wired["synthesizer"].spoken == ["pong: ping"]

    def test_no_speaker_degrades_to_text_only(
        self,
        cfg: Config,
        console: Console,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Text mode must remain useful on a machine with no working audio at all."""
        monkeypatch.setattr(factory, "build_synthesizer", lambda cfg: FakeSynthesizer())

        def no_speaker(config: Config, sample_rate: int) -> Any:
            raise AudioDeviceError("No speaker.", remedy="connect one")

        monkeypatch.setattr(factory, "build_audio_output", no_speaker)
        scripted_input(monkeypatch, ["still works"])

        assert runner.run_text(cfg, console) == 0
        out = capsys.readouterr().out
        assert "text-only mode" in out
        assert "still works" in out

    def test_it_records_the_session_in_the_audit_log(
        self,
        cfg: Config,
        console: Console,
        wired: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scripted_input(monkeypatch, ["remember this"])
        runner.run_text(cfg, console)
        contents = cfg.logging.audit_file.read_text()
        assert "session.start" in contents
        assert "remember this" in contents
        assert "session.end" in contents


class TestSay:
    def test_it_speaks_and_exits(
        self, cfg: Config, console: Console, wired: dict[str, Any]
    ) -> None:
        assert runner.say(cfg, console, "Good evening, sir.") == 0
        assert wired["synthesizer"].spoken == ["Good evening, sir."]
        assert wired["audio_out"].played

    def test_a_missing_voice_is_reported(
        self,
        cfg: Config,
        console: Console,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def explode(config: Config) -> Any:
            raise ModelMissingError("No voice.", remedy="./scripts/download_models.sh piper")

        monkeypatch.setattr(factory, "build_synthesizer", explode)
        assert runner.say(cfg, console, "hello") == 1
        assert "download_models.sh piper" in capsys.readouterr().out


class TestStatePresentation:
    @pytest.mark.parametrize("state", list(State))
    def test_every_state_has_a_label(self, state: State) -> None:
        """A new state must not crash the display by being unlabelled."""
        assert state in runner._STATE_STYLE
