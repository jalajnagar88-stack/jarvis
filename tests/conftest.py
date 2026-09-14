"""Shared fixtures.

The whole suite runs with no microphone, no speakers, no downloaded models, and
no API key — that is the point. Anything that would touch hardware or the
network is mocked here.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from jarvis.config import Config
from jarvis.interfaces.audio import (
    AudioClip,
    AudioInput,
    AudioOutput,
    DeviceInfo,
    Samples,
)
from jarvis.interfaces.stt import Transcriber, Transcript
from jarvis.interfaces.tts import SpeechSynthesizer
from jarvis.interfaces.wake_word import WakeEvent, WakeWordDetector

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = REPO_ROOT / "config.yaml"

# Environment variables that would otherwise leak the developer's real setup
# into the tests. Cleared for every test; opt back in with monkeypatch.
_LEAKY_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ELEVENLABS_API_KEY",
    "JARVIS_CONFIG",
    "JARVIS_LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LEAKY_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stop pydantic-settings finding a developer's real .env.

    ``Secrets`` resolves ``.env`` relative to the working directory, so point
    the working directory at an empty temp dir for the duration of each test
    unless the test deliberately changes it.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """A copy of the repo's real config.yaml.

    Using the shipped file rather than a minimal stub means these tests also
    catch config.yaml drifting away from the pydantic schema.
    """
    destination = tmp_path / "config.yaml"
    shutil.copy(SHIPPED_CONFIG, destination)
    return destination


@pytest.fixture
def cfg(config_path: Path) -> Config:
    return Config.load(config_path)


@pytest.fixture
def fake_devices() -> list[DeviceInfo]:
    """A plausible macOS device list: built-in mic, built-in speakers, a headset."""
    return [
        DeviceInfo(0, "MacBook Pro Microphone", 1, 0, 48000.0, is_default_input=True),
        DeviceInfo(1, "MacBook Pro Speakers", 0, 2, 48000.0, is_default_output=True),
        DeviceInfo(2, "Jabra Evolve 65", 1, 2, 16000.0),
    ]


@pytest.fixture
def no_devices() -> list[DeviceInfo]:
    return []


@pytest.fixture
def stub_models(cfg: Config) -> Iterator[Config]:
    """Create zero-byte stand-ins for every model file the health check looks for.

    Lets us exercise the "models are present" path without downloading 1.4 GB.
    """
    cfg.wake_word.model_dir.mkdir(parents=True, exist_ok=True)
    (cfg.wake_word.model_dir / f"{cfg.wake_word.model}_v0.1.onnx").write_bytes(b"")
    (cfg.wake_word.model_dir / "melspectrogram.onnx").write_bytes(b"")
    (cfg.wake_word.model_dir / "embedding_model.onnx").write_bytes(b"")

    whisper_dir = cfg.stt.model_dir / "models--Systran--faster-whisper-base.en"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    (whisper_dir / "model.bin").write_bytes(b"")

    cfg.tts.piper.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.tts.piper.model_path.write_bytes(b"")
    cfg.tts.piper.config_path.write_text("{}")

    yield cfg


# --------------------------------------------------------------------------- #
# Fake pipeline components
#
# Each one implements a real interface, so a test that passes against these is
# testing the loop's logic rather than the mock's. Hardware-free by
# construction.
# --------------------------------------------------------------------------- #


class FakeAudioInput(AudioInput):
    """Replays a scripted list of blocks, then ends the stream.

    ``flush()`` only counts calls. The real draining semantics belong to
    SoundDeviceInput's own queue and are tested there; here we want a script
    that survives the loop flushing between turns.
    """

    def __init__(
        self,
        blocks: list[Samples],
        *,
        sample_rate: int = 16_000,
        block_size: int = 1280,
    ) -> None:
        self._blocks = list(blocks)
        self._sample_rate = sample_rate
        self._block_size = block_size
        self.started = False
        self.stopped = False
        self.flush_calls = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def block_size(self) -> int:
        return self._block_size

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def blocks(self) -> Iterator[Samples]:
        yield from self._blocks

    def flush(self) -> int:
        self.flush_calls += 1
        return 0


class FakeAudioOutput(AudioOutput):
    """Records what was played instead of making noise."""

    def __init__(self, sample_rate: int = 22_050) -> None:
        self._sample_rate = sample_rate
        self.played: list[AudioClip] = []
        self.cancels = 0
        self.started = False
        self.stopped = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_playing(self) -> bool:
        return False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def play(self, clip: AudioClip) -> None:
        self.played.append(clip)

    def cancel(self) -> None:
        self.cancels += 1

    def wait_until_done(self, timeout: float | None = None) -> bool:
        return True


class FakeWakeWord(WakeWordDetector):
    """Fires on whichever block indices the test names."""

    def __init__(self, fire_on: set[int] | None = None) -> None:
        self.fire_on = fire_on or set()
        self.index = -1
        self.resets = 0

    @property
    def expected_block_size(self) -> int:
        return 1280

    def process(self, block: Samples) -> WakeEvent | None:
        self.index += 1
        if self.index in self.fire_on:
            return WakeEvent(model="hey_jarvis", score=0.9)
        return None

    def reset(self) -> None:
        self.resets += 1


class FakeTranscriber(Transcriber):
    """Returns scripted transcripts, one per call."""

    def __init__(self, texts: list[str] | None = None, *, confidence: float | None = None) -> None:
        self._texts = list(texts or ["what time is it"])
        self._confidence = confidence
        self.calls: list[AudioClip] = []
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, clip: AudioClip) -> Transcript:
        self.calls.append(clip)
        text = self._texts.pop(0) if self._texts else ""
        return Transcript(
            text=text, confidence=self._confidence, duration_seconds=clip.duration_seconds
        )


class FakeSynthesizer(SpeechSynthesizer):
    """Produces one sample per character, so length is assertable."""

    def __init__(self, *, sample_rate: int = 22_050, offline: bool = True) -> None:
        self._sample_rate = sample_rate
        self._offline = offline
        self.spoken: list[str] = []
        self.loaded = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def sends_text_off_machine(self) -> bool:
        return not self._offline

    def load(self) -> None:
        self.loaded = True

    def synthesize(self, text: str) -> AudioClip:
        self.spoken.append(text)
        return AudioClip(np.full(max(1, len(text)), 0.1, dtype=np.float32), self._sample_rate)


# --------------------------------------------------------------------------- #
# Synthetic audio
# --------------------------------------------------------------------------- #


def speech_block(size: int = 1280, level: float = 0.2, seed: int = 0) -> Samples:
    """A block loud enough to count as speech."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(size) * level).astype(np.float32)


def silence_block(size: int = 1280) -> Samples:
    return np.zeros(size, dtype=np.float32)


@pytest.fixture
def utterance() -> list[Samples]:
    """A plausible captured command: a beat of quiet, speech, then a clear pause.

    At 1280-frame blocks and 16 kHz each block is 80 ms, so this is 80 ms of
    lead-in, 800 ms of speech, and 1.2 s of trailing silence -- comfortably past
    the 0.8 s default.
    """
    return [silence_block()] + [speech_block(seed=i) for i in range(10)] + [silence_block()] * 15
