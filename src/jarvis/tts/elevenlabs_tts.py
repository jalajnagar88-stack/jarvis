"""Cloud speech synthesis with ElevenLabs.

An adapter behind the same interface as Piper, for when you want a better voice
than a local model can give and accept the trade.

That trade is real and is stated wherever the engine is selected: using this
sends every reply JARVIS speaks to a third party. Your audio still never leaves
the machine -- ElevenLabs receives text, not recordings -- but the text of every
answer does. Piper sends nothing.
"""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np

from jarvis.audio.resample import from_int16
from jarvis.config import ElevenLabsConfig
from jarvis.errors import AuthError, DependencyMissingError, JarvisError
from jarvis.interfaces.audio import AudioClip
from jarvis.interfaces.tts import SpeechSynthesizer
from jarvis.logging_setup import get_logger

log = get_logger("tts.elevenlabs")

OUTPUT_FORMAT = "pcm_22050"
"""Raw PCM rather than MP3, so no audio decoder dependency is needed."""

OUTPUT_SAMPLE_RATE = 22_050


class ElevenLabsSynthesizer(SpeechSynthesizer):
    """Cloud TTS. Requires a network round trip per sentence."""

    def __init__(self, cfg: ElevenLabsConfig, api_key: str | None) -> None:
        self._cfg = cfg
        self._api_key = api_key
        self._client: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    @property
    def sample_rate(self) -> int:
        return OUTPUT_SAMPLE_RATE

    @property
    def sends_text_off_machine(self) -> bool:
        return True

    def load(self) -> None:
        if self._client is not None:
            return

        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise DependencyMissingError(
                "tts.engine is 'elevenlabs' but the elevenlabs package is not installed.",
                remedy="uv sync --extra elevenlabs  # or set tts.engine: piper in config.yaml",
            ) from exc

        if not self._api_key:
            raise AuthError(
                "ELEVENLABS_API_KEY is not set, so the ElevenLabs voice cannot be used.",
                remedy="Add it to .env, or set tts.engine: piper in config.yaml.",
            )
        if not self._cfg.voice_id:
            raise JarvisError(
                "tts.elevenlabs.voice_id is not set.",
                remedy="Pick a voice in the ElevenLabs dashboard and paste its ID into "
                "config.yaml.",
            )

        log.warning(
            "Using the ElevenLabs voice: reply text is sent to ElevenLabs. "
            "Set tts.engine: piper for fully offline speech."
        )
        self._client = ElevenLabs(api_key=self._api_key)

    def synthesize(self, text: str) -> AudioClip:
        self.load()
        assert self._client is not None

        stripped = text.strip()
        if not stripped:
            return AudioClip(np.zeros(0, dtype=np.float32), OUTPUT_SAMPLE_RATE)

        try:
            stream = self._client.text_to_speech.convert(
                text=stripped,
                voice_id=self._cfg.voice_id,
                model_id=self._cfg.model_id,
                output_format=OUTPUT_FORMAT,
            )
            payload = b"".join(stream)
        except Exception as exc:
            raise JarvisError(
                f"ElevenLabs could not synthesise speech ({exc}).",
                remedy="Check the API key and your quota, or set tts.engine: piper to "
                "fall back to the offline voice.",
            ) from exc

        return AudioClip(_decode_pcm(payload), OUTPUT_SAMPLE_RATE)


def _decode_pcm(payload: bytes) -> np.ndarray:
    """Decode the response body to float32 samples.

    Asking for ``pcm_*`` normally yields headerless little-endian int16, but a
    WAV header turns up often enough to be worth handling rather than playing
    44 bytes of header as a click.
    """
    if payload[:4] == b"RIFF":
        with wave.open(io.BytesIO(payload), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
        return from_int16(np.frombuffer(frames, dtype="<i2"))
    if len(payload) % 2:
        payload = payload[:-1]
    return from_int16(np.frombuffer(payload, dtype="<i2"))
