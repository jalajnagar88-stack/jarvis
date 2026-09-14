"""The interface layer.

These tests defend the architecture rather than any behaviour: the contracts
must stay abstract, must be importable without touching hardware, and the audio
dataclasses must do their arithmetic correctly, because silence detection
depends on it.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from jarvis import interfaces
from jarvis.interfaces import (
    AudioClip,
    AudioInput,
    AudioOutput,
    Brain,
    MemoryStore,
    SpeechSynthesizer,
    Transcriber,
    Transcript,
    WakeWordDetector,
)
from jarvis.interfaces.wake_word import WakeEvent

ABSTRACT = [
    AudioInput,
    AudioOutput,
    WakeWordDetector,
    Transcriber,
    SpeechSynthesizer,
    Brain,
    MemoryStore,
]


class TestContracts:
    @pytest.mark.parametrize("cls", ABSTRACT, ids=lambda c: c.__name__)
    def test_cannot_be_instantiated_directly(self, cls: type) -> None:
        with pytest.raises(TypeError):
            cls()

    @pytest.mark.parametrize("cls", ABSTRACT, ids=lambda c: c.__name__)
    def test_has_at_least_one_abstract_method(self, cls: type) -> None:
        assert getattr(cls, "__abstractmethods__", frozenset())

    @pytest.mark.parametrize("cls", ABSTRACT, ids=lambda c: c.__name__)
    def test_every_public_member_is_documented(self, cls: type) -> None:
        """These docstrings are the spec each later milestone implements against."""
        assert cls.__doc__, f"{cls.__name__} has no docstring"
        undocumented = [
            name
            for name in getattr(cls, "__abstractmethods__", frozenset())
            if not (inspect.getattr_static(cls, name).__doc__ or "").strip()
            and not isinstance(inspect.getattr_static(cls, name), property)
        ]
        assert not undocumented, f"{cls.__name__}: {undocumented}"

    def test_importing_interfaces_pulls_in_no_heavy_dependency(self) -> None:
        """The health check imports this; it must not cost a torch load.

        Checked in a fresh interpreter: asserting against this process's
        sys.modules would only prove that no earlier test happened to import
        them, which several legitimately do.
        """
        import subprocess
        import sys

        probe = (
            "import sys, jarvis.interfaces, jarvis.config, jarvis.health; "
            "heavy = [m for m in ('torch', 'faster_whisper', 'openwakeword', "
            "'sounddevice', 'sentence_transformers', 'piper') if m in sys.modules]; "
            "print(','.join(heavy))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "", f"eagerly imported: {result.stdout.strip()}"

    def test_public_api_is_exported(self) -> None:
        for name in interfaces.__all__:
            assert hasattr(interfaces, name), name


class TestAudioClip:
    def test_duration_is_samples_over_rate(self) -> None:
        clip = AudioClip(np.zeros(16_000, dtype=np.float32), 16_000)
        assert clip.duration_seconds == pytest.approx(1.0)

    def test_duration_of_an_empty_clip_is_zero(self) -> None:
        assert AudioClip(np.zeros(0, dtype=np.float32), 16_000).duration_seconds == 0.0

    def test_a_zero_sample_rate_does_not_divide_by_zero(self) -> None:
        assert AudioClip(np.zeros(10, dtype=np.float32), 0).duration_seconds == 0.0

    def test_rms_of_silence_is_zero(self) -> None:
        assert AudioClip(np.zeros(1000, dtype=np.float32), 16_000).rms() == 0.0

    def test_rms_of_an_empty_clip_is_zero(self) -> None:
        assert AudioClip(np.zeros(0, dtype=np.float32), 16_000).rms() == 0.0

    def test_rms_of_a_constant_signal_is_its_magnitude(self) -> None:
        clip = AudioClip(np.full(1000, 0.5, dtype=np.float32), 16_000)
        assert clip.rms() == pytest.approx(0.5)

    def test_rms_of_a_sine_is_the_expected_ratio(self) -> None:
        """A full-scale sine reads 1/sqrt(2); silence thresholds assume this."""
        t = np.linspace(0, 1, 16_000, endpoint=False, dtype=np.float32)
        clip = AudioClip(np.sin(2 * np.pi * 440 * t).astype(np.float32), 16_000)
        assert clip.rms() == pytest.approx(1 / np.sqrt(2), abs=1e-3)

    def test_rms_is_sign_agnostic(self) -> None:
        negative = AudioClip(np.full(100, -0.3, dtype=np.float32), 16_000)
        assert negative.rms() == pytest.approx(0.3)

    def test_clips_are_immutable(self) -> None:
        clip = AudioClip(np.zeros(10, dtype=np.float32), 16_000)
        with pytest.raises(AttributeError):
            clip.sample_rate = 44_100  # type: ignore[misc]


class TestValueTypes:
    def test_empty_transcript_is_detected(self) -> None:
        assert Transcript(text="   ").is_empty
        assert not Transcript(text="what time is it").is_empty

    def test_wake_event_carries_its_score(self) -> None:
        event = WakeEvent(model="hey_jarvis", score=0.81)
        assert event.model == "hey_jarvis"
        assert event.score == pytest.approx(0.81)


class TestDefaultImplementations:
    def test_synthesizer_stream_skips_blank_sentences(self) -> None:
        """The sentence splitter in milestone 3 will emit empty strings."""

        class Stub(SpeechSynthesizer):
            def load(self) -> None: ...
            def synthesize(self, text: str) -> AudioClip:
                return AudioClip(np.full(len(text), 0.1, dtype=np.float32), 22_050)

            @property
            def sample_rate(self) -> int:
                return 22_050

            @property
            def is_loaded(self) -> bool:
                return True

            @property
            def sends_text_off_machine(self) -> bool:
                return False

        clips = list(Stub().stream(iter(["Good evening.", "  ", "", "Shall I?"])))
        assert [len(c.samples) for c in clips] == [len("Good evening."), len("Shall I?")]
