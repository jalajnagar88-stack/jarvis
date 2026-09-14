"""The status display, and the wiring that connects it to the loop."""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from jarvis import factory
from jarvis.config import Config
from jarvis.state import State
from jarvis.ui.base import NullDisplay, StatusDisplay
from jarvis.ui.transcript import Transcript
from jarvis.ui.tui import TerminalDisplay


class TestTranscript:
    def test_lines_are_kept_in_order(self) -> None:
        transcript = Transcript()
        transcript.add("you", "what time is it")
        transcript.add("jarvis", "twenty past three")
        assert [line.text for line in transcript.lines()] == [
            "what time is it",
            "twenty past three",
        ]

    def test_it_is_bounded(self) -> None:
        """A display that grows forever eventually costs more than the assistant."""
        transcript = Transcript(max_lines=3)
        for index in range(10):
            transcript.add("you", f"line {index}")
        assert len(transcript.lines()) == 3
        assert transcript.lines()[-1].text == "line 9"

    def test_whitespace_is_collapsed(self) -> None:
        transcript = Transcript()
        transcript.add("you", "  lots   of\n  space ")
        assert transcript.lines()[0].text == "lots of space"

    def test_blank_lines_are_dropped(self) -> None:
        transcript = Transcript()
        transcript.add("you", "   ")
        assert transcript.lines() == []

    def test_a_pending_line_shows_last(self) -> None:
        transcript = Transcript()
        transcript.add("you", "hello")
        transcript.set_pending("jarvis", "good eve")
        assert [line.text for line in transcript.lines()] == ["hello", "good eve"]

    def test_a_pending_line_is_replaced_not_appended(self) -> None:
        transcript = Transcript()
        transcript.set_pending("jarvis", "good")
        transcript.set_pending("jarvis", "good evening")
        assert [line.text for line in transcript.lines()] == ["good evening"]

    def test_adding_a_line_clears_the_pending_one(self) -> None:
        transcript = Transcript()
        transcript.set_pending("jarvis", "good eve")
        transcript.add("jarvis", "good evening, sir")
        assert [line.text for line in transcript.lines()] == ["good evening, sir"]

    def test_asides_have_no_speaker(self) -> None:
        transcript = Transcript()
        transcript.add("", "running get_time")
        assert transcript.lines()[0].is_aside

    def test_clear_empties_everything(self) -> None:
        transcript = Transcript()
        transcript.add("you", "hello")
        transcript.set_pending("jarvis", "hi")
        transcript.clear()
        assert transcript.lines() == []

    def test_it_is_thread_safe(self) -> None:
        """The loop, the timers, and the extractor all write to this."""
        import threading

        transcript = Transcript(max_lines=1000)

        def worker(index: int) -> None:
            for step in range(50):
                transcript.add("you", f"{index}-{step}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(transcript.lines()) == 400


class TestNullDisplay:
    def test_every_method_is_safe(self) -> None:
        display = NullDisplay()
        with display:
            display.set_state(State.LISTENING)
            display.add_line("you", "hello")
            display.set_pending("jarvis", "hi")
            display.note("running a tool")


class TestTerminalDisplay:
    @pytest.fixture
    def console(self) -> Console:
        return Console(force_terminal=False, width=70)

    def test_it_starts_and_stops(self, console: Console) -> None:
        display = TerminalDisplay(console=console)
        display.start()
        display.stop()

    def test_starting_twice_is_harmless(self, console: Console) -> None:
        display = TerminalDisplay(console=console)
        display.start()
        display.start()
        display.stop()

    def test_stopping_twice_is_harmless(self, console: Console) -> None:
        display = TerminalDisplay(console=console)
        display.start()
        display.stop()
        display.stop()

    def test_updates_before_start_do_not_raise(self, console: Console) -> None:
        """The loop may report a state before the display is up."""
        display = TerminalDisplay(console=console)
        display.set_state(State.THINKING)
        display.add_line("you", "hello")

    def test_it_renders_every_state(self, console: Console) -> None:
        display = TerminalDisplay(console=console)
        for state in State:
            display.set_state(state)
            assert display._render() is not None

    def test_the_transcript_is_rendered(self, console: Console) -> None:
        display = TerminalDisplay(console=console)
        display.add_line("you", "what time is it")
        with console.capture() as captured:
            console.print(display._render())
        assert "what time is it" in captured.get()

    def test_it_works_as_a_context_manager(self, console: Console) -> None:
        with TerminalDisplay(console=console) as display:
            display.set_state(State.SPEAKING)


class TestWindowDisplay:
    def test_it_never_blocks_the_assistant_when_it_cannot_open(self) -> None:
        """Headless, over SSH, or no tkinter: carry on listening regardless."""
        from jarvis.ui.window import WindowDisplay

        display = WindowDisplay()
        display.start()  # no display attached in CI; must return promptly
        display.set_state(State.LISTENING)
        display.add_line("you", "hello")
        display.stop()

    def test_updates_are_queued_not_applied_directly(self) -> None:
        """Every tkinter call must happen on the window's own thread."""
        from jarvis.ui.window import WindowDisplay

        display = WindowDisplay()
        display.set_state(State.THINKING)
        display.add_line("you", "hello")
        assert not display._updates.empty()


class TestFactorySelection:
    def test_none_gives_a_null_display(self, cfg: Config) -> None:
        cfg.ui.mode = "none"
        assert isinstance(factory.build_display(cfg), NullDisplay)

    def test_tui_gives_a_terminal_display(self, cfg: Config) -> None:
        cfg.ui.mode = "tui"
        assert isinstance(factory.build_display(cfg), TerminalDisplay)

    def test_window_gives_a_window_display(self, cfg: Config) -> None:
        from jarvis.ui.window import WindowDisplay

        cfg.ui.mode = "window"
        assert isinstance(factory.build_display(cfg), WindowDisplay)

    def test_every_display_honours_the_interface(self, cfg: Config) -> None:
        for mode in ("none", "tui", "window"):
            cfg.ui.mode = mode
            assert isinstance(factory.build_display(cfg), StatusDisplay)


class TestRunnerWiring:
    """Regression tests for a whole class of bug.

    Every subsystem below was built, tested, and then reached the running
    assistant only through one line in the runner. Memory once silently did
    not, and the unit tests could not tell -- they inject their own. These
    assert the wiring itself.
    """

    def test_voice_mode_builds_every_subsystem(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[str] = []

        def record(name: str, result: Any = None) -> Any:
            def builder(*args: Any, **kwargs: Any) -> Any:
                built.append(name)
                return result

            return builder

        from .conftest import (
            FakeAudioInput,
            FakeAudioOutput,
            FakeBrain,
            FakeSynthesizer,
            FakeTranscriber,
            FakeWakeWord,
        )

        monkeypatch.setattr(factory, "build_memory", record("memory", object()))
        monkeypatch.setattr(factory, "build_display", record("display", NullDisplay()))

        class StubTimers:
            def announce_with(self, speak: Any) -> None:
                pass

        monkeypatch.setattr(factory, "build_timers", record("timers", StubTimers()))
        monkeypatch.setattr(factory, "build_brain", record("brain", FakeBrain(["ok"])))
        monkeypatch.setattr(factory, "build_synthesizer", lambda cfg: FakeSynthesizer())
        monkeypatch.setattr(factory, "build_display", record("display", NullDisplay()))
        monkeypatch.setattr(factory, "build_wake_word", lambda cfg: FakeWakeWord(set()))
        monkeypatch.setattr(factory, "build_transcriber", lambda cfg: FakeTranscriber())
        monkeypatch.setattr(factory, "build_audio_input", lambda cfg: FakeAudioInput([]))
        monkeypatch.setattr(
            factory, "build_audio_output", lambda cfg, sample_rate: FakeAudioOutput()
        )

        runner_console = Console(force_terminal=False, width=70)
        from jarvis import runner

        runner.run_voice(cfg, runner_console)

        assert "memory" in built, "memory is not wired into voice mode"
        assert "display" in built, "the status display is not wired into voice mode"
        assert "timers" in built, "timers are not wired into voice mode"

    def test_the_brain_receives_memory(self, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
        """The specific bug: build_memory ran, but its result never arrived."""
        seen: dict[str, Any] = {}
        sentinel = object()

        from .conftest import (
            FakeAudioInput,
            FakeAudioOutput,
            FakeBrain,
            FakeSynthesizer,
            FakeTranscriber,
            FakeWakeWord,
        )

        def fake_build_brain(config: Config, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return FakeBrain(["ok"])

        monkeypatch.setattr(factory, "build_memory", lambda cfg: sentinel)
        monkeypatch.setattr(factory, "build_brain", fake_build_brain)
        monkeypatch.setattr(factory, "build_synthesizer", lambda cfg: FakeSynthesizer())
        monkeypatch.setattr(factory, "build_wake_word", lambda cfg: FakeWakeWord(set()))
        monkeypatch.setattr(factory, "build_transcriber", lambda cfg: FakeTranscriber())
        monkeypatch.setattr(factory, "build_audio_input", lambda cfg: FakeAudioInput([]))
        monkeypatch.setattr(
            factory, "build_audio_output", lambda cfg, sample_rate: FakeAudioOutput()
        )

        from jarvis import runner

        runner.run_voice(cfg, Console(force_terminal=False, width=70))
        assert seen.get("memory") is sentinel

    def test_text_mode_also_receives_memory(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        sentinel = object()

        from .conftest import FakeBrain, FakeSynthesizer

        def fake_build_brain(config: Config, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return FakeBrain(["ok"])

        monkeypatch.setattr(factory, "build_memory", lambda cfg: sentinel)
        monkeypatch.setattr(factory, "build_brain", fake_build_brain)
        monkeypatch.setattr(factory, "build_synthesizer", lambda cfg: FakeSynthesizer())
        from .conftest import FakeAudioOutput

        monkeypatch.setattr(
            factory, "build_audio_output", lambda cfg, sample_rate: FakeAudioOutput()
        )
        monkeypatch.setattr(Console, "input", lambda self, prompt="": "/quit")

        from jarvis import runner

        runner.run_text(cfg, Console(force_terminal=False, width=70))
        assert seen.get("memory") is sentinel
