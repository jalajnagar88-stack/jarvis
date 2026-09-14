"""Starting the assistant: the voice loop and the text REPL.

Kept apart from ``cli.py`` so that argument parsing stays argument parsing.
Everything here is about turning a validated config into a running session, and
about failing in plain language when that is not possible.
"""

from __future__ import annotations

import contextlib
import signal
from types import FrameType

from rich.console import Console
from rich.text import Text

from jarvis import factory
from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.errors import JarvisError
from jarvis.interfaces.agent import (
    AgentFailed,
    Brain,
    SentenceComplete,
    TextDelta,
    TurnFinished,
)
from jarvis.interfaces.audio import AudioOutput
from jarvis.interfaces.stt import Transcript
from jarvis.interfaces.tts import SpeechSynthesizer
from jarvis.logging_setup import get_logger
from jarvis.loop import TurnResult, VoiceLoop
from jarvis.state import State

log = get_logger("runner")

_STATE_STYLE = {
    State.IDLE: ("idle", "dim"),
    State.LISTENING: ("listening", "cyan"),
    State.THINKING: ("thinking", "yellow"),
    State.SPEAKING: ("speaking", "green"),
    State.STOPPED: ("stopped", "dim"),
}


def run_voice(cfg: Config, console: Console, *, brain: Brain | None = None) -> int:
    """Run the wake-word voice loop until interrupted."""
    audit = AuditLog(cfg.logging.audit_file)

    try:
        thinker = brain if brain is not None else factory.build_brain(cfg)
        synthesizer = factory.build_synthesizer(cfg)
        synthesizer.load()

        wake_word = factory.build_wake_word(cfg)
        transcriber = factory.build_transcriber(cfg)

        console.print(Text("  Loading models…", style="dim"))
        transcriber.load()

        audio_in = factory.build_audio_input(cfg)
        audio_out = factory.build_audio_output(cfg, sample_rate=synthesizer.sample_rate)
    except JarvisError as exc:
        return _report(console, exc)

    loop = VoiceLoop(
        cfg,
        audio_in=audio_in,
        audio_out=audio_out,
        wake_word=wake_word,
        transcriber=transcriber,
        synthesizer=synthesizer,
        brain=thinker,
        audit=audit,
        on_state=lambda state: _print_state(console, state),
        on_turn=lambda turn: _print_turn(console, turn),
    )

    _install_interrupt_handler(loop, console)

    console.print()
    console.print(
        Text(f'  Say "{_spoken_wake_word(cfg)}", then your message. Ctrl-C to stop.', style="bold")
    )
    console.print()

    try:
        with audio_in, audio_out:
            loop.run()
    except JarvisError as exc:
        return _report(console, exc)
    except KeyboardInterrupt:
        pass

    console.print()
    console.print(Text("  Stopped.", style="dim"))
    return 0


def run_text(cfg: Config, console: Console, *, brain: Brain | None = None) -> int:
    """Run the text REPL: type instead of talking, but still hear the reply.

    This exists so the rest of the pipeline can be developed without a
    microphone -- and without talking to your laptop in an open-plan office.
    Speech output is still real, so it doubles as a way to audition the voice.
    """
    audit = AuditLog(cfg.logging.audit_file)
    audio_out: AudioOutput | None = None

    try:
        thinker = brain if brain is not None else factory.build_brain(cfg)
    except JarvisError as exc:
        return _report(console, exc)

    try:
        synthesizer = factory.build_synthesizer(cfg)
        synthesizer.load()
        audio_out = factory.build_audio_output(cfg, sample_rate=synthesizer.sample_rate)
        audio_out.start()
    except JarvisError as exc:
        # Text mode is still useful with no speaker at all, so this is a
        # downgrade rather than a failure.
        console.print()
        console.print(Text(f"  Speech output unavailable: {exc.message}", style="yellow"))
        if exc.remedy:
            console.print(Text(f"  {exc.remedy}", style="dim"))
        console.print(Text("  Continuing in text-only mode.", style="dim"))
        audio_out = None

    audit.record("session.start", detail="text REPL started")
    console.print()
    console.print(Text("  Text mode. Type a message, or /quit to leave.", style="bold"))
    if audio_out is not None:
        console.print(Text("  Replies are spoken aloud as well as printed.", style="dim"))
    console.print()

    try:
        while True:
            try:
                line = console.input("[bold cyan]you [/bold cyan]> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            except OSError:
                # stdin is closed or not a terminal -- piped from /dev/null, run
                # under nohup, or launched by a scheduler. Leave quietly.
                log.debug("Standard input is unavailable; leaving the REPL.")
                break

            if not line:
                continue
            if line.lower() in {"/quit", "/exit", "/q"}:
                break

            audit.record("turn.transcribed", outcome="ok", detail=line, source="text")
            _run_text_turn(cfg, console, thinker, synthesizer, audio_out, audit, line)
            console.print()
    finally:
        if audio_out is not None:
            audio_out.stop()
        audit.record("session.end", detail="text REPL ended")

    console.print(Text("  Goodbye.", style="dim"))
    return 0


def say(cfg: Config, console: Console, text: str) -> int:
    """Speak one phrase and exit. For auditioning a voice without a conversation."""
    try:
        synthesizer = factory.build_synthesizer(cfg)
        synthesizer.load()
        clip = synthesizer.synthesize(text)
        audio_out = factory.build_audio_output(cfg, sample_rate=synthesizer.sample_rate)
        with audio_out:
            audio_out.play(clip)
            audio_out.wait_until_done(timeout=clip.duration_seconds + 5.0)
    except JarvisError as exc:
        return _report(console, exc)

    console.print(Text(f"  Spoke {clip.duration_seconds:.1f}s of audio.", style="dim"))
    return 0


def _run_text_turn(
    cfg: Config,
    console: Console,
    brain: Brain,
    synthesizer: SpeechSynthesizer,
    audio_out: AudioOutput | None,
    audit: AuditLog,
    line: str,
) -> None:
    """One typed exchange, streamed to the terminal and spoken sentence by sentence.

    Text mode runs the same brain and the same event stream as voice mode, so
    what you hear here is what you would hear standing in the room. Only the
    input differs.
    """
    console.print(Text(f"{cfg.general.name.lower()} > ", style="bold green"), end="")
    wrote_text = False

    for event in brain.respond(line):
        if isinstance(event, TextDelta):
            # Print as it streams, so text mode shows the same incremental
            # arrival that voice mode turns into early speech.
            console.print(event.text, end="", markup=False, highlight=False)
            wrote_text = True

        elif isinstance(event, SentenceComplete):
            if audio_out is not None:
                try:
                    audio_out.play(synthesizer.synthesize(event.text))
                except JarvisError as exc:
                    console.print(Text(f"\n  (could not speak: {exc.message})", style="yellow"))

        elif isinstance(event, AgentFailed):
            if wrote_text:
                console.print()
            console.print(Text(event.spoken, style="yellow"))
            _print_remedy(console, event.detail)
            wrote_text = True
            if audio_out is not None:
                with contextlib.suppress(JarvisError):
                    audio_out.play(synthesizer.synthesize(event.spoken))

        elif isinstance(event, TurnFinished):
            audit.record(
                "turn.replied", outcome="ok", detail=event.text, source="text", **event.usage
            )

    console.print()
    if audio_out is not None:
        audio_out.wait_until_done(timeout=120.0)


def _print_remedy(console: Console, detail: str) -> None:
    """Show the `Try: ...` line from a JarvisError under a spoken apology.

    What is said aloud has to stay short. The instruction that actually fixes it
    belongs on screen, where it can be read and copied.
    """
    for line in detail.splitlines():
        stripped = line.strip()
        if stripped.startswith("Try:"):
            console.print(Text(f"  {stripped}", style="dim"))


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _spoken_wake_word(cfg: Config) -> str:
    """Turn a model name like `hey_jarvis` into something you can read aloud."""
    return cfg.wake_word.model.replace("_", " ")


def _print_state(console: Console, state: State) -> None:
    label, style = _STATE_STYLE[state]
    if state in (State.LISTENING, State.THINKING):
        console.print(Text(f"  [{label}]", style=style))


def _print_turn(console: Console, turn: TurnResult) -> None:
    if turn.transcript is not None and not turn.transcript.is_empty:
        console.print(Text("  you > ", style="bold cyan"), end="")
        console.print(turn.transcript.text)
        console.print(_confidence_note(turn.transcript))
    if turn.reply:
        console.print(Text("  jarvis > ", style="bold green"), end="")
        console.print(turn.reply)
    console.print()


def _confidence_note(transcript: Transcript) -> Text:
    """Surface a low-confidence transcription rather than letting it look certain."""
    if transcript.confidence is None or transcript.confidence > -0.6:
        return Text("")
    return Text(
        "  (low confidence — try speaking a little closer to the microphone)", style="yellow"
    )


def _report(console: Console, exc: JarvisError) -> int:
    console.print()
    console.print(Text(f"  {exc.message}", style="bold red"))
    if exc.remedy:
        console.print(Text(f"  {exc.remedy}", style="dim"))
    console.print()
    console.print(Text("  Run `python -m jarvis health` for the full picture.", style="dim"))
    console.print()
    return 1


def _install_interrupt_handler(loop: VoiceLoop, console: Console) -> None:
    """Make Ctrl-C stop the loop cleanly rather than unwinding mid-callback."""

    def handler(signum: int, frame: FrameType | None) -> None:
        console.print()
        console.print(Text("  Stopping…", style="dim"))
        loop.stop()

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        # Not on the main thread (a test, or an embedded run). Ctrl-C will still
        # raise KeyboardInterrupt, which run_voice catches.
        log.debug("Could not install a SIGINT handler off the main thread.")
