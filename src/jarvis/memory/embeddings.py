"""Local sentence embeddings.

Optional by design. Without sentence-transformers, memory still stores and
retrieves facts -- just by keyword rather than by meaning -- and that is a
deliberate trade: the alternative is making a 2 GB torch download mandatory for
a feature that degrades perfectly well.

Loading is deferred to first use so that constructing one is free and the health
check can ask about it without paying for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jarvis.logging_setup import get_logger

log = get_logger("memory.embeddings")


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers behind a two-method interface."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None
        self._unavailable = False

    @property
    def is_available(self) -> bool:
        """False once loading has been tried and failed.

        Sticky: a missing model should be reported once, not on every turn.
        """
        return not self._unavailable

    def load(self) -> bool:
        """Load the model. Returns whether it is usable."""
        if self._model is not None:
            return True
        if self._unavailable:
            return False

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.info(
                "sentence-transformers is not installed; memory will recall by keyword. "
                "Install it with: uv sync --extra memory"
            )
            self._unavailable = True
            return False

        try:
            self._model = SentenceTransformer(self._model_name, cache_folder=str(self._cache_dir))
        except Exception as exc:
            log.warning(
                "Could not load the embedding model %s (%s); recalling by keyword instead.",
                self._model_name,
                exc,
            )
            self._unavailable = True
            return False

        log.debug("Embedding model %s ready", self._model_name)
        return True

    def encode(self, text: str) -> np.ndarray | None:
        """Embed one string, or None if embeddings are unavailable."""
        if not self.load():
            return None
        vector = self._model.encode(text, show_progress_bar=False)
        return np.asarray(vector, dtype=np.float32).reshape(-1)
