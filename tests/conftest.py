"""Shared fixtures.

The whole suite runs with no microphone, no speakers, no downloaded models, and
no API key — that is the point. Anything that would touch hardware or the
network is mocked here.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from jarvis.config import Config
from jarvis.interfaces.audio import DeviceInfo

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
