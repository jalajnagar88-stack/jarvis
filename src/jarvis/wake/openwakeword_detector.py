"""Wake word detection with openWakeWord.

Runs entirely offline on the CPU. The model takes 80 ms frames of 16 kHz int16
audio and returns a score per wake word per frame.

Two details that are easy to get wrong and unpleasant to debug:

* openWakeWord needs two *shared* models -- a melspectrogram front end and an
  embedding model -- alongside the wake word itself. Loading a wake model
  without them succeeds and then fails at the first prediction, so the paths are
  resolved and checked up front.
* The library ships ``.tflite`` feature models by default even when you ask for
  the ONNX framework, so the paths are passed explicitly rather than left to
  the library's defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.audio.resample import to_int16
from jarvis.config import WakeWordConfig
from jarvis.errors import DependencyMissingError, ModelMissingError
from jarvis.interfaces.audio import Samples
from jarvis.interfaces.wake_word import WakeEvent, WakeWordDetector
from jarvis.logging_setup import get_logger

log = get_logger("wake")

EXPECTED_BLOCK_SIZE = 1280
"""80 ms at 16 kHz, the frame length openWakeWord is trained on."""


class OpenWakeWordDetector(WakeWordDetector):
    """Streaming detector for a single wake phrase."""

    def __init__(self, cfg: WakeWordConfig) -> None:
        self._cfg = cfg
        self._model: Any = None
        self._cooldown_blocks_remaining = 0
        self._blocks_per_second = 16_000 / EXPECTED_BLOCK_SIZE

    @property
    def expected_block_size(self) -> int:
        return EXPECTED_BLOCK_SIZE

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- loading ------------------------------------------------------------ #

    def load(self) -> None:
        """Load the wake model and its feature extractors. Idempotent."""
        if self._model is not None:
            return

        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise DependencyMissingError(
                "openwakeword is not installed, so JARVIS cannot listen for a wake word.",
                remedy="uv sync --extra voice",
            ) from exc

        wake_path = self._resolve_wake_model()
        melspec, embedding = self._resolve_feature_models()

        log.debug("Loading wake word model %s", wake_path.name)
        self._model = Model(
            wakeword_models=[str(wake_path)],
            inference_framework=self._cfg.framework,
            melspec_model_path=str(melspec),
            embedding_model_path=str(embedding),
        )

    def _suffix(self) -> str:
        return ".onnx" if self._cfg.framework == "onnx" else ".tflite"

    def _resolve_wake_model(self) -> Path:
        directory = self._cfg.model_dir
        suffix = self._suffix()
        matches = sorted(p for p in directory.glob(f"{self._cfg.model}*{suffix}") if p.is_file())
        if not matches:
            raise ModelMissingError(
                f"No '{self._cfg.model}' wake word model ({suffix}) in {directory}.",
                remedy="./scripts/download_models.sh wakeword",
            )
        if len(matches) > 1:
            log.debug("Several matching wake models; using %s", matches[-1].name)
        # Sorted order puts the highest version last, e.g. v0.1 before v0.2.
        return matches[-1]

    def _resolve_feature_models(self) -> tuple[Path, Path]:
        suffix = self._suffix()
        melspec = self._cfg.model_dir / f"melspectrogram{suffix}"
        embedding = self._cfg.model_dir / f"embedding_model{suffix}"
        missing = [p.name for p in (melspec, embedding) if not p.is_file()]
        if missing:
            raise ModelMissingError(
                f"The wake word model needs its shared feature extractors, but "
                f"{' and '.join(missing)} are missing from {self._cfg.model_dir}.",
                remedy="./scripts/download_models.sh wakeword",
            )
        return melspec, embedding

    # -- detection ---------------------------------------------------------- #

    def process(self, block: Samples) -> WakeEvent | None:
        if self._model is None:
            self.load()
        assert self._model is not None

        if self._cooldown_blocks_remaining > 0:
            # Still feed the model during the cooldown so its internal buffers
            # stay continuous; just refuse to report anything.
            self._model.predict(to_int16(block))
            self._cooldown_blocks_remaining -= 1
            return None

        scores = self._model.predict(to_int16(block))
        if not scores:
            return None

        name, score = max(scores.items(), key=lambda item: item[1])
        if score < self._cfg.threshold:
            return None

        log.info("Wake word '%s' detected (score %.2f)", name, score)
        self._begin_cooldown()
        return WakeEvent(model=str(name), score=float(score))

    def _begin_cooldown(self) -> None:
        self._cooldown_blocks_remaining = round(
            self._cfg.cooldown_seconds * self._blocks_per_second
        )

    def reset(self) -> None:
        """Clear model state and start the cooldown.

        Called after a wake has been handled. Without the reset, the elevated
        scores from the phrase just spoken linger in the model's buffers and
        immediately re-trigger.
        """
        if self._model is not None:
            reset = getattr(self._model, "reset", None)
            if callable(reset):
                reset()
            else:
                log.debug("This openWakeWord build has no reset(); relying on cooldown alone.")
        self._begin_cooldown()
