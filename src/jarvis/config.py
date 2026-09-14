"""Configuration: ``config.yaml`` for behaviour, ``.env`` for secrets.

The split is strict. Anything that would be embarrassing in a screenshot lives
in ``.env`` and is loaded into :class:`Secrets`; everything else lives in
``config.yaml`` and is validated into :class:`Config`. Nothing reads
``os.environ`` outside this module.

Relative paths in ``config.yaml`` are resolved against the directory containing
that file, so the repo is portable and ``~`` works everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jarvis.errors import ConfigError

DEFAULT_CONFIG_FILENAME = "config.yaml"


class _Base(BaseModel):
    """Reject unknown keys so a typo in config.yaml is an error, not a silent no-op."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


class GeneralConfig(_Base):
    name: str = "JARVIS"
    persona: str = "You are JARVIS, a calm, precise British butler."
    timezone: str | None = None
    units: Literal["metric", "imperial"] = "metric"


class SilenceConfig(_Base):
    threshold: float = Field(0.015, ge=0.0, le=1.0)
    duration_seconds: float = Field(0.8, gt=0.0)
    max_utterance_seconds: float = Field(20.0, gt=0.0)
    min_utterance_seconds: float = Field(0.4, ge=0.0)


class AudioConfig(_Base):
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = Field(16_000, gt=0)
    channels: int = Field(1, ge=1, le=2)
    block_size: int = Field(1280, gt=0)
    silence: SilenceConfig = Field(default_factory=SilenceConfig)


class WakeWordConfig(_Base):
    enabled: bool = True
    model: str = "hey_jarvis"
    model_dir: Path = Path("./models/openwakeword")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    cooldown_seconds: float = Field(2.0, ge=0.0)
    framework: Literal["onnx", "tflite"] = "onnx"


class SttConfig(_Base):
    engine: Literal["faster_whisper"] = "faster_whisper"
    model: str = "base.en"
    model_dir: Path = Path("./models/whisper")
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    compute_type: str = "int8"
    cpu_threads: int | None = None
    language: str | None = "en"
    beam_size: int = Field(1, ge=1)
    min_logprob: float = -1.0


class LlmConfig(_Base):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-5"
    max_tokens: int = Field(2048, gt=0)
    streaming: bool = True
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    thinking: Literal["adaptive", "disabled"] = "adaptive"
    timeout_seconds: float = Field(60.0, gt=0.0)
    max_retries: int = Field(2, ge=0)
    history_turns: int = Field(20, ge=0)


class PiperConfig(_Base):
    voice: str = "en_GB-alan-medium"
    model_dir: Path = Path("./models/piper")
    binary: Path | None = None
    length_scale: float = Field(1.05, gt=0.0)
    sentence_silence: float = Field(0.2, ge=0.0)

    @property
    def model_path(self) -> Path:
        """Absolute path to the voice's ``.onnx`` file."""
        return self.model_dir / f"{self.voice}.onnx"

    @property
    def config_path(self) -> Path:
        """Absolute path to the voice's ``.onnx.json`` sidecar."""
        return self.model_dir / f"{self.voice}.onnx.json"


class ElevenLabsConfig(_Base):
    voice_id: str | None = None
    model_id: str = "eleven_turbo_v2_5"


class TtsConfig(_Base):
    engine: Literal["piper", "elevenlabs"] = "piper"
    piper: PiperConfig = Field(default_factory=PiperConfig)
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)


class ExtractionConfig(_Base):
    enabled: bool = True
    model: str = "claude-haiku-4-5"
    max_tokens: int = Field(512, gt=0)


class MemoryConfig(_Base):
    db_path: Path = Path("./var/jarvis.db")
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_model_dir: Path = Path("./models/embeddings")
    top_k: int = Field(5, ge=0)
    min_similarity: float = Field(0.35, ge=-1.0, le=1.0)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)


class FilesystemToolConfig(_Base):
    workspace: Path = Path("./workspace")
    max_file_mb: float = Field(10.0, gt=0.0)


class ShellToolConfig(_Base):
    # Declared here so it shows up in the schema and the audit log, but the
    # value is not trusted: jarvis.tools.safety always requires confirmation.
    require_confirmation: bool = True
    timeout_seconds: float = Field(30.0, gt=0.0)
    cwd: Path = Path("./workspace")
    extra_denylist: list[str] = Field(default_factory=list)


class WeatherToolConfig(_Base):
    default_latitude: float | None = Field(None, ge=-90.0, le=90.0)
    default_longitude: float | None = Field(None, ge=-180.0, le=180.0)
    default_location_name: str | None = None


class WebSearchToolConfig(_Base):
    max_uses: int = Field(3, ge=1)
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)

    @field_validator("blocked_domains")
    @classmethod
    def _not_both(cls, v: list[str], info: Any) -> list[str]:
        if v and info.data.get("allowed_domains"):
            raise ValueError(
                "set at most one of tools.web_search.allowed_domains or blocked_domains, "
                "never both — the API rejects requests that carry both"
            )
        return v


class NotesToolConfig(_Base):
    # Relative to the filesystem workspace, never resolved against the repo.
    path: str = "notes.md"


class ToolsConfig(_Base):
    enabled: list[str] = Field(default_factory=list)
    max_iterations: int = Field(8, ge=1)
    filesystem: FilesystemToolConfig = Field(default_factory=FilesystemToolConfig)
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    weather: WeatherToolConfig = Field(default_factory=WeatherToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    notes: NotesToolConfig = Field(default_factory=NotesToolConfig)


class InterruptConfig(_Base):
    enabled: bool = True
    threshold: float = Field(0.05, ge=0.0, le=1.0)
    min_duration_seconds: float = Field(0.25, ge=0.0)


class UiConfig(_Base):
    mode: Literal["none", "tui", "window"] = "none"
    always_on_top: bool = True
    transcript_lines: int = Field(12, ge=1)


class LoggingConfig(_Base):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path | None = Path("./logs/jarvis.log")
    max_bytes: int = Field(5_242_880, gt=0)
    backup_count: int = Field(3, ge=0)
    audit_file: Path = Path("./logs/audit.log")


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


class Secrets(BaseSettings):
    """API keys, read from the environment and ``.env``.

    Never logged, never written to the audit file, never included in
    ``Config.model_dump()``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    anthropic_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    anthropic_base_url: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_none(cls, v: Any) -> Any:
        """Treat ``KEY=`` in .env as unset rather than as an empty-string key."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def redacted(self) -> dict[str, str]:
        """A dict safe to print: presence only, never the value."""
        return {
            name: ("set" if getattr(self, name) else "not set")
            for name in ("anthropic_api_key", "elevenlabs_api_key", "anthropic_base_url")
        }


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #


class Config(_Base):
    """The whole of ``config.yaml``, validated."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    interrupt: InterruptConfig = Field(default_factory=InterruptConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Populated by `load()`; not part of the YAML schema.
    source_path: Path | None = Field(default=None, exclude=True)
    root_dir: Path = Field(default_factory=Path.cwd, exclude=True)
    secrets: Secrets = Field(default_factory=Secrets, exclude=True)

    # -- path handling ------------------------------------------------------ #

    def resolve_paths(self, root: Path) -> None:
        """Rewrite every path field to an absolute path.

        Relative paths are taken as relative to ``root`` (the directory holding
        config.yaml), not the process's working directory, so JARVIS behaves the
        same no matter where you launch it from.
        """
        self.root_dir = root
        self.wake_word.model_dir = _abspath(self.wake_word.model_dir, root)
        self.stt.model_dir = _abspath(self.stt.model_dir, root)
        self.tts.piper.model_dir = _abspath(self.tts.piper.model_dir, root)
        if self.tts.piper.binary is not None:
            self.tts.piper.binary = _abspath(self.tts.piper.binary, root)
        self.memory.db_path = _abspath(self.memory.db_path, root)
        self.memory.embedding_model_dir = _abspath(self.memory.embedding_model_dir, root)
        self.tools.filesystem.workspace = _abspath(self.tools.filesystem.workspace, root)
        self.tools.shell.cwd = _abspath(self.tools.shell.cwd, root)
        if self.logging.file is not None:
            self.logging.file = _abspath(self.logging.file, root)
        self.logging.audit_file = _abspath(self.logging.audit_file, root)

    # -- loading ------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load and validate configuration.

        Resolution order for the config file: the ``path`` argument, then
        ``$JARVIS_CONFIG``, then ``./config.yaml``.

        Raises:
            ConfigError: the file is missing, is not valid YAML, or fails
                validation. The message names the offending key.
        """
        config_path = _locate(path)
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"{config_path} is not valid YAML: {exc}",
                remedy="Check indentation — YAML is whitespace-sensitive.",
            ) from exc
        except OSError as exc:
            raise ConfigError(f"Could not read {config_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError(
                f"{config_path} should contain a mapping of sections, got {type(raw).__name__}.",
                remedy="Compare it against the config.yaml shipped with the repo.",
            )

        try:
            cfg = cls.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(
                f"{config_path} has invalid settings:\n{_format_validation_error(exc)}",
                remedy="Fix the keys listed above, or delete them to fall back to defaults.",
            ) from exc

        cfg.source_path = config_path
        cfg.resolve_paths(config_path.parent.resolve())
        cfg.secrets = Secrets()
        _apply_env_overrides(cfg)
        return cfg


def _locate(path: str | Path | None) -> Path:
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise ConfigError(
                f"No config file at {candidate}.",
                remedy=f"Copy the example: cp config.yaml {candidate}",
            )
        return candidate.resolve()

    env_path = os.environ.get("JARVIS_CONFIG")
    if env_path:
        return _locate(env_path)

    candidate = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if not candidate.is_file():
        raise ConfigError(
            f"No {DEFAULT_CONFIG_FILENAME} found in {Path.cwd()}.",
            remedy=(
                "Run JARVIS from the repository root, or point at the file with "
                "JARVIS_CONFIG=/path/to/config.yaml"
            ),
        )
    return candidate.resolve()


def _apply_env_overrides(cfg: Config) -> None:
    """Environment overrides for the handful of settings worth changing per-run."""
    level = os.environ.get("JARVIS_LOG_LEVEL")
    if level:
        normalised = level.strip().upper()
        if normalised not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ConfigError(
                f"JARVIS_LOG_LEVEL={level!r} is not a valid level.",
                remedy="Use one of DEBUG, INFO, WARNING, ERROR.",
            )
        cfg.logging.level = normalised  # type: ignore[assignment]


def _abspath(p: Path, root: Path) -> Path:
    expanded = p.expanduser()
    return expanded if expanded.is_absolute() else (root / expanded).resolve()


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)
