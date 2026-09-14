"""The voice loop.

    wake word -> record until silence -> transcribe -> think -> speak

The brain streams its reply, and the loop speaks each sentence the moment it is
complete rather than waiting for the whole answer. Because playback queues
rather than blocks, sentence two is being synthesised while sentence one is
still being heard.

The loop owns no audio backend and no API client of its own. It is handed the
interfaces, which is what lets the tests drive the whole thing with synthetic
audio, no hardware and no API key.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from jarvis.audio.recorder import PreRollBuffer, RecordingOutcome, UtteranceRecorder
from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.errors import JarvisError
from jarvis.interfaces.agent import (
    AgentFailed,
    Brain,
    ConfirmationRequired,
    SentenceComplete,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
)
from jarvis.interfaces.audio import AudioInput, AudioOutput, Samples
from jarvis.interfaces.stt import Transcriber, Transcript
from jarvis.interfaces.tts import SpeechSynthesizer
from jarvis.interfaces.wake_word import WakeWordDetector
from jarvis.logging_setup import get_logger
from jarvis.state import State, StateListener
from jarvis.tools.confirm import interpret_confirmation

log = get_logger("loop")


@dataclass(slots=True)
class TurnResult:
    """What happened in one wake-to-reply cycle. Returned for the tests and the UI."""

    transcript: Transcript | None
    reply: str | None
    outcome: RecordingOutcome
    failed: bool = False


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
        brain: Brain,
        audit: AuditLog,
        on_state: StateListener | None = None,
        on_turn: Callable[[TurnResult], None] | None = None,
        on_reply_delta: Callable[[str], None] | None = None,
        on_confirmation: Callable[[ConfirmationRequired], None] | None = None,
        on_tool: Callable[[str, bool | None], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._audio_in = audio_in
        self._audio_out = audio_out
        self._wake_word = wake_word
        self._transcriber = transcriber
        self._synthesizer = synthesizer
        self._brain = brain
        self._audit = audit
        self._on_state = on_state
        self._on_turn = on_turn
        self._on_reply_delta = on_reply_delta
        self._on_confirmation = on_confirmation
        self._on_tool = on_tool

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

        reply, failed = self.think_and_speak(transcript.text)
        self._notify_turn(TurnResult(transcript, reply, recording.outcome, failed=failed))
        self._return_to_idle()

    # -- thinking ----------------------------------------------------------- #

    def think_and_speak(self, user_text: str) -> tuple[str, bool]:
        """Run one brain turn, speaking each sentence the moment it completes.

        This is where the reply stops feeling slow. The brain yields a
        SentenceComplete as soon as one is finished, and playback queues rather
        than blocks, so sentence two is being synthesised while sentence one is
        still being heard.

        Returns:
            The full reply, and whether the turn failed.
        """
        self._set_state(State.THINKING)
        spoken_anything = False
        reply = ""
        failed = False
        started = time.monotonic()
        first_sentence_at: float | None = None

        for event in self._brain.respond(user_text):
            if isinstance(event, TextDelta) and self._on_reply_delta is not None:
                self._on_reply_delta(event.text)

            elif isinstance(event, SentenceComplete):
                if first_sentence_at is None:
                    first_sentence_at = time.monotonic() - started
                if not spoken_anything:
                    self._set_state(State.SPEAKING)
                    spoken_anything = True
                self._enqueue(event.text)

            elif isinstance(event, ConfirmationRequired):
                # Drain anything already queued so the question is not spoken
                # over the tail of the previous sentence.
                self._drain_playback()
                approved = self._ask_for_confirmation(event)
                self._brain.confirm(event.call_id, approved)
                self._set_state(State.THINKING)

            elif isinstance(event, ToolCallStarted):
                log.info("Tool: %s", event.name)
                if self._on_tool is not None:
                    self._on_tool(event.name, None)

            elif isinstance(event, ToolCallFinished):
                log.debug("Tool %s finished: %s", event.name, event.summary)
                self._audit.record(
                    "tool.call",
                    outcome="ok" if event.ok else "error",
                    detail=event.name,
                    summary=event.summary,
                )
                if self._on_tool is not None:
                    self._on_tool(event.name, event.ok)

            elif isinstance(event, AgentFailed):
                failed = True
                log.error("Brain failed: %s", event.detail)
                self._audit.record("turn.failed", outcome="error", detail=event.detail)
                self._set_state(State.SPEAKING)
                self._enqueue(event.spoken)
                reply = event.spoken
                spoken_anything = True

            elif isinstance(event, TurnFinished) and not failed:
                reply = event.text
                self._audit.record(
                    "turn.replied",
                    outcome="ok",
                    detail=event.text,
                    stop_reason=event.stop_reason,
                    **event.usage,
                )

        if first_sentence_at is not None:
            log.debug("First sentence ready after %.2fs", first_sentence_at)

        if spoken_anything:
            self._drain_playback()
        return reply, failed

    def _enqueue(self, sentence: str) -> None:
        """Synthesise one sentence and hand it to the speaker.

        Synthesis happens inline rather than on a worker thread. Piper runs
        several times faster than real time, so by the time a sentence finishes
        playing the next is long since ready -- and a queue of audio clips is
        far easier to cancel cleanly than a queue of threads, which is what
        milestone 6 needs.
        """
        if not sentence.strip():
            return
        try:
            clip = self._synthesizer.synthesize(sentence)
        except JarvisError as exc:
            log.error("Could not synthesise speech: %s", exc)
            self._audit.record("tts.failed", outcome="error", detail=str(exc))
            return
        self._audio_out.play(clip)

    def _drain_playback(self) -> None:
        """Wait for everything queued to finish playing.

        The timeout is generous because the queue may hold several sentences by
        now; it exists only so a dead output stream cannot park the loop.
        """
        self._audio_out.wait_until_done(timeout=120.0)

    # -- confirmation ------------------------------------------------------- #

    def _ask_for_confirmation(self, event: ConfirmationRequired) -> bool:
        """Put a pending tool call to the user and wait for an answer.

        Speaks the question, then records and transcribes the reply through the
        same path as any other utterance. Anything that is not a clear yes is a
        refusal, and so is silence: the user walking away must not be read as
        consent.
        """
        self._audit.record(
            "tool.confirmation_asked", outcome="info", detail=event.name, prompt=event.prompt
        )
        log.info("Confirmation needed: %s", event.prompt)
        if self._on_confirmation is not None:
            self._on_confirmation(event)

        for attempt in range(2):
            self.speak(event.prompt if attempt == 0 else "Sorry -- yes or no?")
            answer = self._listen_for_answer()
            if answer is None:
                continue
            decision = interpret_confirmation(answer)
            if decision is not None:
                log.info("The user said %r -> %s", answer, "yes" if decision else "no")
                self._audit.record(
                    "tool.confirmation_answered",
                    outcome="confirmed" if decision else "declined",
                    detail=event.name,
                    heard=answer,
                )
                return decision

        self._audit.record(
            "tool.confirmation_answered",
            outcome="declined",
            detail=event.name,
            heard="(no clear answer)",
        )
        self.speak("I'll take that as a no.")
        return False

    def _listen_for_answer(self) -> str | None:
        """Record and transcribe one short reply."""
        self._audio_in.flush()
        self._set_state(State.LISTENING)
        recorder = UtteranceRecorder(
            sample_rate=self._cfg.audio.sample_rate, silence=self._cfg.audio.silence
        )
        for block in self._audio_in.blocks():
            if not self._running:
                return None
            if recorder.feed(block) is not RecordingOutcome.RECORDING:
                break

        recording = recorder.result()
        if not recording.usable:
            return None
        try:
            transcript = self._transcriber.transcribe(recording.clip)
        except JarvisError as exc:
            log.error("Could not transcribe the confirmation: %s", exc)
            return None
        return None if transcript.is_empty else transcript.text

    # -- speaking ----------------------------------------------------------- #

    def speak(self, text: str) -> None:
        """Synthesise and play a fixed phrase, blocking until it has finished.

        Used for the loop's own apologies. Replies from the brain go through
        :meth:`think_and_speak`, which starts speaking sooner.
        """
        if not text.strip():
            return
        self._set_state(State.SPEAKING)
        self._enqueue(text)
        self._drain_playback()

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
