"""Offline speech synthesis with Piper.

Piper runs a small ONNX voice model locally. Nothing is sent anywhere, which is
why it is the default: the assistant can hold a conversation with the network
unplugged apart from the one API call the brain makes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jarvis.config import PiperConfig
from jarvis.errors import DependencyMissingError, ModelMissingError
from jarvis.interfaces.audio import AudioClip
from jarvis.interfaces.tts import SpeechSynthesizer
from jarvis.logging_setup import get_logger

log = get_logger("tts.piper")

DEFAULT_SAMPLE_RATE = 22_050
"""Reported before the voice is loaded; replaced by the voice's real rate."""


class PiperSynthesizer(SpeechSynthesizer):
    """Local neural TTS."""

    def __init__(self, cfg: PiperConfig) -> None:
        self._cfg = cfg
        self._voice: Any = None
        self._sample_rate = DEFAULT_SAMPLE_RATE

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def sends_text_off_machine(self) -> bool:
        return False

    def load(self) -> None:
        if self._voice is not None:
            return

        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise DependencyMissingError(
                "piper-tts is not installed, so JARVIS has no voice.",
                remedy="uv sync --extra voice",
            ) from exc

        model, sidecar = self._cfg.model_path, self._cfg.config_path
        missing = [p.name for p in (model, sidecar) if not p.is_file()]
        if missing:
            raise ModelMissingError(
                f"The Piper voice '{self._cfg.voice}' is incomplete: "
                f"{' and '.join(missing)} missing from {self._cfg.model_dir}.",
                remedy="./scripts/download_models.sh piper",
            )

        log.debug("Loading Piper voice %s", self._cfg.voice)
        try:
            self._voice = PiperVoice.load(model, config_path=sidecar)
        except Exception as exc:
            raise ModelMissingError(
                f"Could not load the Piper voice '{self._cfg.voice}' ({exc}).",
                remedy=(
                    "The model file may be truncated. Delete it and re-run "
                    "./scripts/download_models.sh piper"
                ),
            ) from exc

        rate = getattr(getattr(self._voice, "config", None), "sample_rate", None)
        if isinstance(rate, int) and rate > 0:
            self._sample_rate = rate
        log.debug("Piper voice ready at %d Hz", self._sample_rate)

    def _synthesis_config(self) -> Any:
        from piper import SynthesisConfig

        return SynthesisConfig(length_scale=self._cfg.length_scale)

    def synthesize(self, text: str) -> AudioClip:
        self.load()
        assert self._voice is not None

        stripped = text.strip()
        if not stripped:
            return AudioClip(np.zeros(0, dtype=np.float32), self._sample_rate)

        chunks = list(self._voice.synthesize(stripped, syn_config=self._synthesis_config()))
        if not chunks:
            return AudioClip(np.zeros(0, dtype=np.float32), self._sample_rate)

        # Piper emits one chunk per sentence; insert the configured gap between
        # them so a multi-sentence reply does not run together.
        rate = int(getattr(chunks[0], "sample_rate", self._sample_rate))
        self._sample_rate = rate
        gap = np.zeros(int(self._cfg.sentence_silence * rate), dtype=np.float32)

        pieces: list[np.ndarray] = []
        for index, chunk in enumerate(chunks):
            if index and gap.size:
                pieces.append(gap)
            pieces.append(np.asarray(chunk.audio_float_array, dtype=np.float32))

        return AudioClip(np.concatenate(pieces).astype(np.float32), rate)
