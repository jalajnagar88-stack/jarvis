"""Local transcription with faster-whisper.

Runs on the CPU. On Apple Silicon that is not a compromise: CTranslate2 has no
Metal backend, so ``cpu`` with ``int8`` quantisation is both the only option and
a genuinely fast one for ``base.en``.

The model is loaded on first use rather than in the constructor, so building
the object stays cheap and the health check can ask about it without spending
a second and a few hundred megabytes of RAM.
"""

from __future__ import annotations

from typing import Any

from jarvis.config import SttConfig
from jarvis.errors import DependencyMissingError, ModelMissingError
from jarvis.interfaces.audio import AudioClip
from jarvis.interfaces.stt import Transcriber, Transcript
from jarvis.logging_setup import get_logger

log = get_logger("stt")

WHISPER_SAMPLE_RATE = 16_000
"""Whisper is trained at 16 kHz and resamples anything else internally."""

# Whisper reliably hallucinates these on silence or noise, because its training
# data was full of subtitled videos. Dropping them is cheaper and more reliable
# than tuning no_speech_threshold to catch every case.
_HALLUCINATIONS = frozenset(
    {
        "thank you.",
        "thanks for watching!",
        "thank you for watching.",
        "thank you for watching!",
        "you",
        "bye.",
        ".",
        "so",
        "[blank_audio]",
        "subtitles by the amara.org community",
    }
)


class FasterWhisperTranscriber(Transcriber):
    """Whisper transcription, locally."""

    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyMissingError(
                "faster-whisper is not installed, so JARVIS cannot transcribe speech.",
                remedy="uv sync --extra voice",
            ) from exc

        log.debug(
            "Loading Whisper '%s' (%s/%s)",
            self._cfg.model,
            self._cfg.device,
            self._cfg.compute_type,
        )
        try:
            self._model = WhisperModel(
                self._cfg.model,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
                cpu_threads=self._cfg.cpu_threads or 0,
                download_root=str(self._cfg.model_dir),
                # Never reach for the network mid-conversation. The model is a
                # setup step; a missing one is a setup error, not a stall.
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelMissingError(
                f"Could not load the Whisper model '{self._cfg.model}' from "
                f"{self._cfg.model_dir} ({exc}).",
                remedy="./scripts/download_models.sh whisper",
            ) from exc

    def transcribe(self, clip: AudioClip) -> Transcript:
        self.load()
        assert self._model is not None

        samples = clip.samples
        if clip.sample_rate != WHISPER_SAMPLE_RATE:
            from jarvis.audio.resample import resample

            samples = resample(samples, clip.sample_rate, WHISPER_SAMPLE_RATE)

        if samples.size == 0:
            return Transcript(text="", duration_seconds=0.0)

        segments, info = self._model.transcribe(
            samples,
            language=self._cfg.language,
            beam_size=self._cfg.beam_size,
            without_timestamps=True,
            # Each utterance is independent: carrying context across turns makes
            # Whisper repeat the previous command when this one is unclear.
            condition_on_previous_text=False,
        )

        collected = list(segments)
        text = " ".join(segment.text.strip() for segment in collected).strip()
        confidence = (
            sum(segment.avg_logprob for segment in collected) / len(collected)
            if collected
            else None
        )

        if self._is_hallucination(text, confidence):
            log.debug("Discarding likely hallucination %r (logprob %s)", text, confidence)
            return Transcript(
                text="", confidence=confidence, duration_seconds=clip.duration_seconds
            )

        return Transcript(
            text=text,
            language=getattr(info, "language", self._cfg.language),
            confidence=confidence,
            duration_seconds=clip.duration_seconds,
        )

    def _is_hallucination(self, text: str, confidence: float | None) -> bool:
        """Filter output that Whisper invented out of silence."""
        if not text:
            return False
        if text.strip().lower().strip("!.,") in {phrase.strip("!.,") for phrase in _HALLUCINATIONS}:
            return True
        return confidence is not None and confidence < self._cfg.min_logprob
