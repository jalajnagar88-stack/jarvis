"""The voice loop.

    wake word -> record until silence -> transcribe -> reply -> speak

At milestone 2 the reply is the transcript itself, spoken back. That is not a
placeholder for its own sake: echoing proves every stage of the audio pipeline
independently of whether the brain is any good, and it is the only way to tell a
transcription problem from a reasoning one later.

The loop owns no audio backend of its own. It is handed the interfaces, which is
what lets the tests drive the whole thing with synthetic audio and no hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from jarvis.audio.recorder import PreRollBuffer, RecordingOutcome, UtteranceRecorder
from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.errors import JarvisError
from jarvis.interfaces.audio import AudioInput, AudioOutput, Samples
from jarvis.interfaces.stt import Transcriber, Transcript
from jarvis.interfaces.tts import SpeechSynthesizer
from jarvis.interfaces.wake_word import WakeWordDetector
from jarvis.logging_setup import get_logger
from jarvis.state import State, StateListener

log = get_logger("loop")

Responder = Callable[[str], str]
"""Turns the user's words into a reply. Milestone 3 swaps the echo for the brain."""


def echo_responder(text: str) -> str:
    """The milestone 2 'brain': say back exactly what was heard."""
    return text


@dataclass(slots=True)
class TurnResult:
    """What happened in one wake-to-reply cycle. Returned for the tests and the UI."""

    transcript: Transcript | None
    reply: str | None
    outcome: RecordingOutcome
    heard_wake_word: bool = True


class VoiceLoop:
    """Runs the listen-think-speak cycle until stopped."""

    def __init__(
        self,
        cfg: Config,
        *,
        audio_in: AudioInput,
        audio_out: AudioOutput,
        wake_word: WakeWordDetector,
        transcriber: Transcriber,
        synthesizer: SpeechSynthesizer,
        audit: AuditLog,
        responder: Responder = echo_responder,
        on_state: StateListener | None = None,
        on_turn: Callable[[TurnResult], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._audio_in = audio_in
        self._audio_out = audio_out
        self._wake_word = wake_word
        self._transcriber = transcriber
        self._synthesizer = synthesizer
        self._audit = audit
        self._responder = responder
        self._on_state = on_state
        self._on_turn = on_turn

        self._state = State.IDLE
        self._running = False
        self._recorder: UtteranceRecorder | None = None
        self._preroll = PreRollBuffer(
            sample_rate=cfg.audio.sample_rate,
            seconds=cfg.audio.silence.preroll_seconds,
            block_size=cfg.audio.block_size,
        )

    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, state: State) -> None:
        if state is self._state:
            return
        self._state = state
        log.debug("State -> %s", state.value)
        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception as exc:  # a broken UI must not stop the assistant
                log.warning("State listener raised %r; continuing.", exc)

    # -- running ------------------------------------------------------------ #

    def run(self) -> None:
        """Listen until :meth:`stop` is called or the input ends."""
        self._running = True
        self._audit.record("session.start", detail="voice loop started")
        self._set_state(State.IDLE)
        log.info(
            "Listening for '%s'. Say it, then speak. Ctrl-C to stop.",
            self._cfg.wake_word.model,
        )

        try:
            for block in self._audio_in.blocks():
                if not self._running:
                    break
                self._consume(block)
        finally:
            self._set_state(State.STOPPED)
            self._audit.record("session.end", detail="voice loop stopped")

    def stop(self) -> None:
        self._running = False
        self._audio_out.cancel()

    def _consume(self, block: Samples) -> None:
        """Route one audio block according to the current state."""
        if self._state is State.IDLE:
            self._preroll.push(block)
            if self._wake_word.process(block) is not None:
                self._begin_listening()
            return

        if self._state is State.LISTENING:
            assert self._recorder is not None
            if self._recorder.feed(block) is not RecordingOutcome.RECORDING:
                self._finish_listening()

    def _begin_listening(self) -> None:
        self._audit.record("wake.detected", detail=self._cfg.wake_word.model)
        self._recorder = UtteranceRecorder(
            sample_rate=self._cfg.audio.sample_rate,
            silence=self._cfg.audio.silence,
            preroll=self._preroll.drain(),
        )
        self._set_state(State.LISTENING)

    def _finish_listening(self) -> None:
        assert self._recorder is not None
        recording = self._recorder.result()
        self._recorder = None

        if not recording.usable:
            log.info("Nothing was said after the wake word; going back to sleep.")
            self._audit.record("turn.abandoned", outcome="info", detail=recording.outcome.value)
            self._notify_turn(TurnResult(None, None, recording.outcome))
            self._return_to_idle()
            return

        self._set_state(State.THINKING)
        try:
            started = time.monotonic()
            transcript = self._transcriber.transcribe(recording.clip)
            elapsed = time.monotonic() - started
        except JarvisError as exc:
            self._fail("I could not transcribe that.", exc)
            return

        if transcript.is_empty:
            log.info("Heard %.1fs of audio but no words in it.", recording.clip.duration_seconds)
            self._audit.record(
                "turn.empty",
                outcome="info",
                detail=f"{recording.clip.duration_seconds:.1f}s of audio",
            )
            self._notify_turn(TurnResult(transcript, None, recording.outcome))
            self._return_to_idle()
            return

        log.info("Heard: %s", transcript.text)
        log.debug("Transcribed %.1fs of audio in %.2fs", recording.clip.duration_seconds, elapsed)
        self._audit.record(
            "turn.transcribed",
            outcome="ok",
            detail=transcript.text,
            audio_seconds=round(recording.clip.duration_seconds, 2),
        )

        reply = self._responder(transcript.text)
        self.speak(reply)
        self._notify_turn(TurnResult(transcript, reply, recording.outcome))
        self._return_to_idle()

    # -- speaking ----------------------------------------------------------- #

    def speak(self, text: str) -> None:
        """Synthesise and play ``text``, blocking until it has finished."""
        if not text.strip():
            return
        self._set_state(State.SPEAKING)
        try:
            clip = self._synthesizer.synthesize(text)
            self._audio_out.play(clip)
            # Wait a little beyond the clip's own length; if the stream dies we
            # would otherwise block here forever.
            self._audio_out.wait_until_done(timeout=clip.duration_seconds + 5.0)
        except JarvisError as exc:
            log.error("Could not speak: %s", exc)
            self._audit.record("tts.failed", outcome="error", detail=str(exc))

    def _fail(self, spoken: str, exc: Exception) -> None:
        log.error("%s (%s)", spoken, exc)
        self._audit.record("turn.failed", outcome="error", detail=str(exc))
        self.speak(spoken)
        self._return_to_idle()

    def _return_to_idle(self) -> None:
        # Everything the microphone captured while we were transcribing and
        # talking is stale, and includes JARVIS's own voice. Throw it away
        # before listening again, or the wake detector hears the echo.
        discarded = self._audio_in.flush()
        if discarded:
            log.debug("Dropped %d block(s) captured while busy", discarded)
        self._wake_word.reset()
        self._preroll.drain()
        self._set_state(State.IDLE)

    def _notify_turn(self, result: TurnResult) -> None:
        if self._on_turn is not None:
            try:
                self._on_turn(result)
            except Exception as exc:
                log.warning("Turn listener raised %r; continuing.", exc)
