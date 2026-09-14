"""Subsystem health checks.

``python -m jarvis`` runs these and prints the result. The contract is that a
health check never throws and never loads a model into memory: it asks cheap
questions (is the package importable, does the file exist, is the directory
writable) and reports what it finds.

Every failing check carries a remedy the user can act on. "faster-whisper is
not installed" is useless on its own; "run `uv sync --extra voice`" is not.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.interfaces.audio import DeviceInfo

MIN_PYTHON = (3, 11)


class Status(StrEnum):
    """Outcome of a single check, ordered by severity."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"

    @property
    def severity(self) -> int:
        return {Status.OK: 0, Status.SKIP: 0, Status.WARN: 1, Status.FAIL: 2}[self]


@dataclass(frozen=True, slots=True)
class Check:
    """One health check result."""

    id: str
    subsystem: str
    name: str
    status: Status
    detail: str
    remedy: str | None = None


@dataclass(slots=True)
class HealthReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def overall(self) -> Status:
        if not self.checks:
            return Status.OK
        worst = max(self.checks, key=lambda c: c.status.severity)
        return worst.status if worst.status.severity else Status.OK

    def by_id(self, check_id: str) -> Check | None:
        return next((c for c in self.checks if c.id == check_id), None)

    def passed(self, *check_ids: str) -> bool:
        """True when every named check is OK (WARN does not count as passing)."""
        return all((c := self.by_id(i)) is not None and c.status is Status.OK for i in check_ids)

    @property
    def capabilities(self) -> dict[str, tuple[bool, str]]:
        """What JARVIS can actually do right now, and why not if it can't.

        Keyed by capability; the value is ``(available, explanation)``. This is
        the part of the report worth reading first — it answers "can I test
        anything yet?" without decoding twenty individual checks.
        """
        caps: dict[str, tuple[bool, str]] = {}

        text_ok = self.passed("brain.sdk", "brain.key")
        caps["Text conversation (--text)"] = (
            text_ok,
            "ready" if text_ok else "needs the anthropic SDK and ANTHROPIC_API_KEY",
        )

        listen_ok = self.passed(
            "audio.backend", "audio.input", "wake.package", "wake.model", "stt.package", "stt.model"
        )
        caps["Wake word and transcription"] = (
            listen_ok,
            "ready" if listen_ok else "needs the voice extra, a microphone, and downloaded models",
        )

        speak_ok = self.passed("audio.backend", "audio.output", "tts.engine", "tts.voice")
        caps["Speech output"] = (
            speak_ok,
            "ready" if speak_ok else "needs a speaker and a downloaded voice",
        )

        caps["Full voice loop"] = (
            text_ok and listen_ok and speak_ok,
            "ready" if (text_ok and listen_ok and speak_ok) else "see the checks above",
        )

        recall_ok = self.passed("memory.database", "memory.embeddings")
        caps["Semantic memory"] = (
            recall_ok,
            "ready" if recall_ok else "works without embeddings, but recall falls back to keywords",
        )
        return caps


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _importable(module: str) -> bool:
    """True if ``module`` can be imported, without importing it.

    ``find_spec`` can raise for packages with a broken ``__init__``; a package
    we cannot even inspect is not usable, so treat that as absent.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _version_of(module: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(module)
    except (PackageNotFoundError, ImportError, ValueError):
        return None


def _dir_writable(path: Path) -> tuple[bool, str | None]:
    """Can we create ``path`` and write inside it?"""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"cannot create the directory: {exc}"
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".jarvis-write-test-"):
            pass
    except OSError as exc:
        return False, f"directory exists but is not writable: {exc}"
    return True, None


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _dir_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


# --------------------------------------------------------------------------- #
# Audio device discovery
# --------------------------------------------------------------------------- #


def probe_audio_backend() -> tuple[list[DeviceInfo], str | None]:
    """Enumerate audio devices, distinguishing the ways this can go wrong.

    Returns:
        ``(devices, error)``. ``error`` is None on success. The three failure
        modes are genuinely different problems with different fixes, so they get
        different messages: the package is absent, the package is present but
        the native PortAudio library is not, or PortAudio works and there is
        simply no hardware.
    """
    if not _importable("sounddevice"):
        return [], "sounddevice is not installed"

    try:
        import sounddevice as sd
    except OSError as exc:
        # The classic macOS case: `pip install sounddevice` succeeds, but the
        # PortAudio dylib it binds to was never installed.
        return [], f"sounddevice is installed but its native library will not load ({exc})"
    except Exception as exc:  # a broken backend must not crash the report
        return [], f"sounddevice could not be imported ({exc!r})"

    try:
        raw = sd.query_devices()
        try:
            default_in, default_out = sd.default.device
        except (AttributeError, TypeError):
            default_in, default_out = -1, -1
    except Exception as exc:  # PortAudio raises a variety of host errors
        return [], f"PortAudio could not list devices ({exc})"

    devices: list[DeviceInfo] = []
    for index, dev in enumerate(raw):
        devices.append(
            DeviceInfo(
                index=index,
                name=str(dev.get("name", f"device {index}")),
                max_input_channels=int(dev.get("max_input_channels", 0)),
                max_output_channels=int(dev.get("max_output_channels", 0)),
                default_sample_rate=float(dev.get("default_samplerate", 0.0)),
                is_default_input=(index == default_in),
                is_default_output=(index == default_out),
            )
        )
    return devices, None


def list_audio_devices() -> list[DeviceInfo]:
    """Enumerate audio devices, or an empty list if that is not possible."""
    devices, _ = probe_audio_backend()
    return devices


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _check_runtime(cfg: Config) -> Iterator[Check]:
    version = ".".join(str(p) for p in sys.version_info[:3])
    if sys.version_info[:2] >= MIN_PYTHON:
        yield Check(
            "runtime.python",
            "Runtime",
            "Python version",
            Status.OK,
            f"{version} on {platform.system()} {platform.machine()}",
        )
    else:
        yield Check(
            "runtime.python",
            "Runtime",
            "Python version",
            Status.FAIL,
            f"{version} is too old; JARVIS needs {'.'.join(map(str, MIN_PYTHON))} or newer.",
            remedy="uv python install 3.11 && uv sync",
        )

    source = cfg.source_path or Path("<defaults>")
    yield Check(
        "runtime.config",
        "Runtime",
        "Configuration",
        Status.OK,
        str(source),
    )


def _check_audio(cfg: Config) -> Iterator[Check]:
    devices, error = probe_audio_backend()

    if error is not None:
        yield Check(
            "audio.backend",
            "Audio",
            "Backend",
            Status.FAIL,
            f"{error}.",
            remedy=_audio_remedy(error),
        )
        for check_id, label in (("audio.input", "Microphone"), ("audio.output", "Speaker")):
            yield Check(
                check_id, "Audio", label, Status.SKIP, "not checked — no working audio backend"
            )
        return

    if not devices:
        yield Check(
            "audio.backend",
            "Audio",
            "Backend",
            Status.FAIL,
            "PortAudio is working but reports no audio devices at all.",
            remedy="Connect a microphone or speakers, then re-run this check.",
        )
        for check_id, label in (("audio.input", "Microphone"), ("audio.output", "Speaker")):
            yield Check(check_id, "Audio", label, Status.SKIP, "not checked — no devices")
        return

    yield Check(
        "audio.backend",
        "Audio",
        "Backend",
        Status.OK,
        f"sounddevice {_version_of('sounddevice') or '?'}, {len(devices)} device(s) visible",
    )

    yield _check_one_direction(cfg, devices, want_input=True)
    yield _check_one_direction(cfg, devices, want_input=False)


def _audio_remedy(error: str) -> str:
    """Map a backend failure to the command that actually fixes it."""
    if "not installed" in error:
        return "uv sync --extra voice"
    if "native library" in error:
        return (
            "The PortAudio library is missing. On macOS: brew install portaudio "
            "(then: uv sync --extra voice --reinstall-package sounddevice). "
            "On Debian/Ubuntu: sudo apt install libportaudio2"
        )
    return "uv sync --extra voice --reinstall-package sounddevice"


def _check_one_direction(cfg: Config, devices: list[DeviceInfo], *, want_input: bool) -> Check:
    """Resolve and report either the microphone or the speaker."""
    if want_input:
        check_id, label, selector = "audio.input", "Microphone", cfg.audio.input_device
        usable = [d for d in devices if d.max_input_channels > 0]
        setting, absent_remedy = (
            "audio.input_device",
            (
                "Plug in or enable a microphone. On macOS also grant microphone access: "
                "System Settings > Privacy & Security > Microphone."
            ),
        )
    else:
        check_id, label, selector = "audio.output", "Speaker", cfg.audio.output_device
        usable = [d for d in devices if d.max_output_channels > 0]
        setting, absent_remedy = "audio.output_device", "Connect speakers or headphones."

    if not usable:
        return Check(
            check_id,
            "Audio",
            label,
            Status.FAIL,
            f"no {'input' if want_input else 'output'} devices — JARVIS has nothing to "
            f"{'listen with' if want_input else 'speak through'}.",
            remedy=absent_remedy,
        )

    chosen = _pick_device(usable, selector, want_input=want_input)
    if chosen is None:
        return Check(
            check_id,
            "Audio",
            label,
            Status.FAIL,
            f"{setting} is set to {selector!r}, which matches no device.",
            remedy=f"python -m jarvis devices  # then set {setting} in config.yaml",
        )

    is_default = chosen.is_default_input if want_input else chosen.is_default_output
    channels = chosen.max_input_channels if want_input else chosen.max_output_channels
    return Check(
        check_id,
        "Audio",
        label,
        Status.OK,
        f"{chosen.name} ({'system default' if is_default else 'selected'}, "
        f"{channels}ch @ {chosen.default_sample_rate:.0f} Hz)",
    )


def _pick_device(
    devices: list[DeviceInfo], selector: int | str | None, *, want_input: bool
) -> DeviceInfo | None:
    """Resolve a config selector to a device, mirroring what the audio layer does."""
    if selector is None:
        default = next(
            (d for d in devices if (d.is_default_input if want_input else d.is_default_output)),
            None,
        )
        return default or devices[0]
    if isinstance(selector, int):
        return next((d for d in devices if d.index == selector), None)
    needle = selector.lower()
    return next((d for d in devices if needle in d.name.lower()), None)


def _check_wake_word(cfg: Config) -> Iterator[Check]:
    if not cfg.wake_word.enabled:
        yield Check(
            "wake.package",
            "Wake word",
            "openWakeWord",
            Status.SKIP,
            "disabled in config (wake_word.enabled: false)",
        )
        yield Check("wake.model", "Wake word", "Model", Status.SKIP, "disabled in config")
        return

    if not _importable("openwakeword"):
        yield Check(
            "wake.package",
            "Wake word",
            "openWakeWord",
            Status.FAIL,
            "openwakeword is not installed.",
            remedy="uv sync --extra voice",
        )
    else:
        yield Check(
            "wake.package",
            "Wake word",
            "openWakeWord",
            Status.OK,
            f"openwakeword {_version_of('openwakeword') or '?'} ({cfg.wake_word.framework})",
        )

    found = _find_wake_models(cfg.wake_word.model_dir, cfg.wake_word.model, cfg.wake_word.framework)
    missing_support = _missing_wake_support(cfg.wake_word.model_dir, cfg.wake_word.framework)

    if not found:
        yield Check(
            "wake.model",
            "Wake word",
            "Model",
            Status.FAIL,
            f"no '{cfg.wake_word.model}' model in {cfg.wake_word.model_dir}.",
            remedy="./scripts/download_models.sh wakeword",
        )
    elif missing_support:
        yield Check(
            "wake.model",
            "Wake word",
            "Model",
            Status.FAIL,
            f"found {found[0].name} but the shared feature extractors are missing "
            f"({', '.join(missing_support)}).",
            remedy="./scripts/download_models.sh wakeword",
        )
    else:
        yield Check(
            "wake.model",
            "Wake word",
            "Model",
            Status.OK,
            f"{found[0].name} ({_human_size(found[0].stat().st_size)}), "
            f"threshold {cfg.wake_word.threshold}",
        )


def _find_wake_models(model_dir: Path, name: str, framework: str) -> list[Path]:
    if not model_dir.is_dir():
        return []
    suffix = ".onnx" if framework == "onnx" else ".tflite"
    return sorted(p for p in model_dir.glob(f"{name}*{suffix}") if p.is_file())


def _missing_wake_support(model_dir: Path, framework: str) -> list[str]:
    """openWakeWord needs a melspectrogram and an embedding model alongside the
    wake word itself. A wake model without them loads and then fails at runtime."""
    suffix = ".onnx" if framework == "onnx" else ".tflite"
    required = [f"melspectrogram{suffix}", f"embedding_model{suffix}"]
    return [name for name in required if not (model_dir / name).is_file()]


def _check_stt(cfg: Config) -> Iterator[Check]:
    if not _importable("faster_whisper"):
        yield Check(
            "stt.package",
            "Speech to text",
            "faster-whisper",
            Status.FAIL,
            "faster-whisper is not installed.",
            remedy="uv sync --extra voice",
        )
    else:
        yield Check(
            "stt.package",
            "Speech to text",
            "faster-whisper",
            Status.OK,
            f"faster-whisper {_version_of('faster-whisper') or '?'}, "
            f"{cfg.stt.device}/{cfg.stt.compute_type}",
        )

    model_path = _find_whisper_model(cfg.stt.model_dir)
    if model_path is None:
        yield Check(
            "stt.model",
            "Speech to text",
            f"Model '{cfg.stt.model}'",
            Status.FAIL,
            f"not found under {cfg.stt.model_dir}.",
            remedy="./scripts/download_models.sh whisper",
        )
    else:
        size = _human_size(_dir_size(model_path.parent))
        yield Check(
            "stt.model",
            "Speech to text",
            f"Model '{cfg.stt.model}'",
            Status.OK,
            f"{model_path.parent} ({size})",
        )

    if cfg.stt.device == "cuda" and platform.system() == "Darwin":
        yield Check(
            "stt.device",
            "Speech to text",
            "Device",
            Status.FAIL,
            "stt.device is 'cuda', but CTranslate2 has no GPU backend on macOS.",
            remedy="Set stt.device: cpu in config.yaml.",
        )


def _find_whisper_model(model_dir: Path) -> Path | None:
    """Locate a converted CTranslate2 model.

    faster-whisper writes either a flat directory containing ``model.bin`` or a
    HuggingFace cache layout with the file several levels down, so search for
    the file itself rather than guessing the layout.
    """
    if not model_dir.is_dir():
        return None
    direct = model_dir / "model.bin"
    if direct.is_file():
        return direct
    return next(iter(sorted(model_dir.rglob("model.bin"))), None)


def _check_brain(cfg: Config) -> Iterator[Check]:
    if not _importable("anthropic"):
        yield Check(
            "brain.sdk",
            "Brain",
            "Anthropic SDK",
            Status.FAIL,
            "the anthropic package is not installed.",
            remedy="uv sync",
        )
    else:
        yield Check(
            "brain.sdk",
            "Brain",
            "Anthropic SDK",
            Status.OK,
            f"anthropic {_version_of('anthropic') or '?'}, model {cfg.llm.model} "
            f"(effort {cfg.llm.effort}, thinking {cfg.llm.thinking})",
        )

    if cfg.secrets.anthropic_api_key:
        key = cfg.secrets.anthropic_api_key
        yield Check(
            "brain.key",
            "Brain",
            "API key",
            Status.OK,
            f"ANTHROPIC_API_KEY is set (…{key[-4:]})",
        )
    else:
        yield Check(
            "brain.key",
            "Brain",
            "API key",
            Status.FAIL,
            "ANTHROPIC_API_KEY is not set, so JARVIS can listen and speak but cannot think.",
            remedy="cp .env.example .env  # then paste your key from console.anthropic.com",
        )

    if cfg.secrets.anthropic_base_url:
        yield Check(
            "brain.endpoint",
            "Brain",
            "Endpoint",
            Status.WARN,
            f"ANTHROPIC_BASE_URL overrides the default API endpoint "
            f"({cfg.secrets.anthropic_base_url}).",
            remedy="Unset ANTHROPIC_BASE_URL in .env if that was not deliberate.",
        )


def _check_tts(cfg: Config) -> Iterator[Check]:
    if cfg.tts.engine == "piper":
        yield from _check_piper(cfg)
    else:
        yield from _check_elevenlabs(cfg)


def _check_piper(cfg: Config) -> Iterator[Check]:
    piper = cfg.tts.piper
    binary = str(piper.binary) if piper.binary else shutil.which("piper")
    has_package = _importable("piper")

    if has_package:
        yield Check(
            "tts.engine",
            "Text to speech",
            "Piper",
            Status.OK,
            f"piper-tts {_version_of('piper-tts') or '?'} (Python package), fully offline",
        )
    elif binary and Path(binary).exists():
        yield Check(
            "tts.engine",
            "Text to speech",
            "Piper",
            Status.OK,
            f"binary at {binary}, fully offline",
        )
    else:
        yield Check(
            "tts.engine",
            "Text to speech",
            "Piper",
            Status.FAIL,
            "neither the piper-tts package nor a piper binary was found.",
            remedy="uv sync --extra voice",
        )

    model, sidecar = piper.model_path, piper.config_path
    if model.is_file() and sidecar.is_file():
        yield Check(
            "tts.voice",
            "Text to speech",
            f"Voice '{piper.voice}'",
            Status.OK,
            f"{model} ({_human_size(model.stat().st_size)})",
        )
    elif model.is_file():
        yield Check(
            "tts.voice",
            "Text to speech",
            f"Voice '{piper.voice}'",
            Status.FAIL,
            f"{model.name} is present but its {sidecar.name} sidecar is missing. Piper needs both.",
            remedy="./scripts/download_models.sh piper",
        )
    else:
        yield Check(
            "tts.voice",
            "Text to speech",
            f"Voice '{piper.voice}'",
            Status.FAIL,
            f"not found in {piper.model_dir}.",
            remedy="./scripts/download_models.sh piper",
        )


def _check_elevenlabs(cfg: Config) -> Iterator[Check]:
    if not _importable("elevenlabs"):
        yield Check(
            "tts.engine",
            "Text to speech",
            "ElevenLabs",
            Status.FAIL,
            "tts.engine is 'elevenlabs' but the elevenlabs package is not installed.",
            remedy="uv sync --extra elevenlabs  # or set tts.engine: piper",
        )
    else:
        yield Check(
            "tts.engine",
            "Text to speech",
            "ElevenLabs",
            Status.WARN,
            "cloud engine selected — reply text is sent to ElevenLabs. "
            "Switch to Piper for a fully offline setup.",
        )

    if not cfg.secrets.elevenlabs_api_key:
        yield Check(
            "tts.voice",
            "Text to speech",
            "ElevenLabs key",
            Status.FAIL,
            "ELEVENLABS_API_KEY is not set.",
            remedy="Add it to .env, or set tts.engine: piper in config.yaml.",
        )
    elif not cfg.tts.elevenlabs.voice_id:
        yield Check(
            "tts.voice",
            "Text to speech",
            "ElevenLabs voice",
            Status.FAIL,
            "tts.elevenlabs.voice_id is not set.",
            remedy="Pick a voice in the ElevenLabs dashboard and paste its ID into config.yaml.",
        )
    else:
        yield Check(
            "tts.voice",
            "Text to speech",
            "ElevenLabs voice",
            Status.OK,
            f"voice {cfg.tts.elevenlabs.voice_id}, model {cfg.tts.elevenlabs.model_id}",
        )


def _check_memory(cfg: Config) -> Iterator[Check]:
    db_dir = cfg.memory.db_path.parent
    writable, reason = _dir_writable(db_dir)
    if not writable:
        yield Check(
            "memory.database",
            "Memory",
            "Database",
            Status.FAIL,
            f"cannot use {cfg.memory.db_path}: {reason}",
            remedy=f"Fix permissions on {db_dir}, or point memory.db_path somewhere writable.",
        )
    else:
        try:
            with sqlite3.connect(cfg.memory.db_path) as conn:
                conn.execute("SELECT 1")
            exists = cfg.memory.db_path.is_file()
            size = _human_size(cfg.memory.db_path.stat().st_size) if exists else "new"
            yield Check(
                "memory.database",
                "Memory",
                "Database",
                Status.OK,
                f"SQLite {sqlite3.sqlite_version} at {cfg.memory.db_path} ({size})",
            )
        except sqlite3.Error as exc:
            yield Check(
                "memory.database",
                "Memory",
                "Database",
                Status.FAIL,
                f"SQLite could not open {cfg.memory.db_path}: {exc}",
                remedy="Delete the file to start fresh, or point memory.db_path elsewhere.",
            )

    if not _importable("sentence_transformers"):
        yield Check(
            "memory.embeddings",
            "Memory",
            "Embeddings",
            Status.WARN,
            "sentence-transformers is not installed. Facts will still be stored and "
            "recalled, but by keyword rather than by meaning.",
            remedy="uv sync --extra memory",
        )
    else:
        cached = cfg.memory.embedding_model_dir.is_dir() and any(
            cfg.memory.embedding_model_dir.iterdir()
        )
        detail = (
            f"sentence-transformers {_version_of('sentence-transformers') or '?'}, "
            f"{cfg.memory.embedding_model}"
        )
        if cached:
            yield Check(
                "memory.embeddings", "Memory", "Embeddings", Status.OK, f"{detail} (cached)"
            )
        else:
            yield Check(
                "memory.embeddings",
                "Memory",
                "Embeddings",
                Status.WARN,
                f"{detail} — not downloaded yet; it will be fetched on first use.",
                remedy="./scripts/download_models.sh embeddings",
            )


def _check_tools(cfg: Config) -> Iterator[Check]:
    workspace = cfg.tools.filesystem.workspace
    writable, reason = _dir_writable(workspace)
    if writable:
        yield Check(
            "tools.workspace",
            "Tools",
            "File sandbox",
            Status.OK,
            f"{workspace} — read_file and write_file cannot escape this directory",
        )
    else:
        yield Check(
            "tools.workspace",
            "Tools",
            "File sandbox",
            Status.FAIL,
            f"cannot use {workspace}: {reason}",
            remedy=f"mkdir -p {workspace}  # or change tools.filesystem.workspace",
        )

    unknown = sorted(set(cfg.tools.enabled) - _KNOWN_TOOLS)
    if unknown:
        yield Check(
            "tools.enabled",
            "Tools",
            "Enabled tools",
            Status.WARN,
            f"{len(cfg.tools.enabled)} enabled, but these are not recognised: {', '.join(unknown)}",
            remedy="Remove them from tools.enabled, or check the spelling.",
        )
    else:
        yield Check(
            "tools.enabled",
            "Tools",
            "Enabled tools",
            Status.OK,
            f"{len(cfg.tools.enabled)} enabled: {', '.join(cfg.tools.enabled) or 'none'}",
        )

    yield Check(
        "tools.shell",
        "Tools",
        "Shell safety",
        Status.OK,
        "run_shell and write_file always require confirmation; destructive patterns "
        "are denied outright",
    )


# Tools that will exist by the end of milestone 4 and 5. Listed here so a typo
# in config.yaml is caught at startup rather than silently dropping a capability.
_KNOWN_TOOLS = {
    "get_time",
    "get_weather",
    "web_search",
    "run_shell",
    "read_file",
    "write_file",
    "set_timer",
    "add_note",
    "open_app",
    "control_volume",
    "remember_this",
    "forget_this",
}


def _check_logging(cfg: Config) -> Iterator[Check]:
    if cfg.logging.file is not None:
        writable, reason = _dir_writable(cfg.logging.file.parent)
        if writable:
            yield Check(
                "logging.file",
                "Logging",
                "Application log",
                Status.OK,
                f"{cfg.logging.file} (level {cfg.logging.level})",
            )
        else:
            yield Check(
                "logging.file",
                "Logging",
                "Application log",
                Status.WARN,
                f"cannot write {cfg.logging.file}: {reason}. Logging to console only.",
                remedy="Set logging.file to null to silence this, or fix the directory.",
            )
    else:
        yield Check(
            "logging.file",
            "Logging",
            "Application log",
            Status.SKIP,
            "file logging disabled (logging.file: null)",
        )

    audit = AuditLog(cfg.logging.audit_file)
    ok, reason = audit.is_writable()
    if ok:
        yield Check(
            "logging.audit",
            "Logging",
            "Audit trail",
            Status.OK,
            f"{cfg.logging.audit_file} — every tool call and confirmation is recorded here",
        )
    else:
        yield Check(
            "logging.audit",
            "Logging",
            "Audit trail",
            Status.FAIL,
            f"cannot write {cfg.logging.audit_file}: {reason}. JARVIS refuses to run "
            "tools it cannot account for.",
            remedy=f"mkdir -p {cfg.logging.audit_file.parent} and check permissions.",
        )


def _check_ui(cfg: Config) -> Iterator[Check]:
    if cfg.ui.mode == "none":
        yield Check(
            "ui.mode",
            "Interface",
            "Status display",
            Status.SKIP,
            "disabled (ui.mode: none) -- try `--ui tui`",
        )
        return
    if cfg.ui.mode == "window" and not _importable("tkinter"):
        yield Check(
            "ui.mode",
            "Interface",
            "Status display",
            Status.WARN,
            "ui.mode is 'window' but tkinter is not available; JARVIS will run without a display.",
            remedy="Set ui.mode: tui for a terminal panel instead.",
        )
        return
    yield Check(
        "ui.mode",
        "Interface",
        "Status display",
        Status.OK,
        f"{cfg.ui.mode}, {cfg.ui.transcript_lines} transcript lines",
    )


def _check_privacy(cfg: Config) -> Iterator[Check]:
    """State the privacy posture as a check so it is impossible to miss."""
    cloud_tts = cfg.tts.engine != "piper"
    if cloud_tts:
        yield Check(
            "privacy.audio",
            "Privacy",
            "Data leaving machine",
            Status.WARN,
            "Transcribed text goes to the Anthropic API, and reply text goes to "
            "ElevenLabs. Audio itself never leaves the machine.",
        )
    else:
        yield Check(
            "privacy.audio",
            "Privacy",
            "Data leaving machine",
            Status.OK,
            "Only transcribed text goes to the Anthropic API. Audio never leaves the "
            "machine; wake word, transcription, and speech all run locally.",
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

_SECTIONS = (
    _check_runtime,
    _check_audio,
    _check_wake_word,
    _check_stt,
    _check_brain,
    _check_tts,
    _check_memory,
    _check_tools,
    _check_logging,
    _check_ui,
    _check_privacy,
)


def run_health_checks(cfg: Config) -> HealthReport:
    """Run every check. Never raises — a check that blows up becomes a FAIL."""
    report = HealthReport()
    for section in _SECTIONS:
        try:
            report.checks.extend(section(cfg))
        except Exception as exc:  # a broken check must not kill the report
            report.checks.append(
                Check(
                    f"{section.__name__}.error",
                    "Internal",
                    section.__name__.removeprefix("_check_").title(),
                    Status.FAIL,
                    f"this check itself failed: {exc!r}",
                    remedy="Please report this — a health check should never throw.",
                )
            )
    return report


def environment_summary() -> dict[str, str]:
    """Context worth printing alongside the report."""
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cwd": str(Path.cwd()),
        "virtualenv": os.environ.get("VIRTUAL_ENV", "none"),
    }
