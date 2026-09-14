"""Sample-rate conversion.

Deliberately minimal: linear interpolation, numpy only. Adding scipy or
soxr for a better filter would be a ~40 MB dependency to fix aliasing nobody
can hear in a 22 kHz butler voice played through a laptop speaker.

Resampling is the exception rather than the rule — the output stream is opened
at the synthesiser's own rate — so this runs only when a clip arrives at an
unexpected rate.
"""

from __future__ import annotations

import numpy as np

from jarvis.interfaces.audio import Samples


def resample(samples: Samples, from_rate: int, to_rate: int) -> Samples:
    """Convert ``samples`` from one sample rate to another.

    Returns the input unchanged when the rates already match, which is the
    common case.
    """
    if from_rate == to_rate or samples.size == 0:
        return samples
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {from_rate} -> {to_rate}")

    duration = samples.size / from_rate
    target_count = max(1, round(duration * to_rate))

    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.linspace(0, samples.size - 1, target_count, dtype=np.float64)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def to_int16(samples: Samples) -> np.ndarray:
    """Convert float32 in [-1, 1] to int16, clipping rather than wrapping.

    Wrapping turns a slightly-too-loud sample into a full-scale sample of the
    opposite sign, which sounds like a gunshot. Clipping just sounds loud.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def from_int16(samples: np.ndarray) -> Samples:
    """Convert int16 PCM to float32 in [-1, 1]."""
    return (samples.astype(np.float32) / 32768.0).astype(np.float32)
