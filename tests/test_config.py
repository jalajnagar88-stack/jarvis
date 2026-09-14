"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import Config, Secrets
from jarvis.errors import ConfigError


class TestLoading:
    def test_loads_the_shipped_config(self, config_path: Path) -> None:
        cfg = Config.load(config_path)
        assert cfg.general.name == "JARVIS"
        assert cfg.llm.model == "claude-sonnet-5"
        assert cfg.wake_word.model == "hey_jarvis"
        assert cfg.tts.piper.voice == "en_GB-alan-medium"
        assert cfg.source_path == config_path

    def test_missing_file_names_the_path_and_a_remedy(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as excinfo:
            Config.load(tmp_path / "nope.yaml")
        assert "nope.yaml" in str(excinfo.value)
        assert excinfo.value.remedy

    def test_no_config_in_cwd_explains_where_it_looked(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as excinfo:
            Config.load()
        assert "config.yaml" in str(excinfo.value)

    def test_env_var_selects_the_file(
        self, config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JARVIS_CONFIG", str(config_path))
        assert Config.load().source_path == config_path

    def test_malformed_yaml_is_reported_as_such(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("general:\n  name: [unclosed\n")
        with pytest.raises(ConfigError, match="not valid YAML"):
            Config.load(bad)

    def test_non_mapping_yaml_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("- just\n- a\n- list\n")
        with pytest.raises(ConfigError, match="mapping of sections"):
            Config.load(bad)

    def test_empty_file_falls_back_to_defaults(self, tmp_path: Path) -> None:
        empty = tmp_path / "config.yaml"
        empty.write_text("")
        cfg = Config.load(empty)
        assert cfg.general.name == "JARVIS"


class TestValidation:
    def test_unknown_key_is_an_error_not_a_silent_noop(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("general:\n  nmae: JARVIS\n")
        with pytest.raises(ConfigError) as excinfo:
            Config.load(bad)
        assert "nmae" in str(excinfo.value)

    def test_out_of_range_value_names_the_key(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("wake_word:\n  threshold: 5.0\n")
        with pytest.raises(ConfigError) as excinfo:
            Config.load(bad)
        assert "wake_word.threshold" in str(excinfo.value)

    def test_bad_enum_value_names_the_key(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("tts:\n  engine: festival\n")
        with pytest.raises(ConfigError) as excinfo:
            Config.load(bad)
        assert "tts.engine" in str(excinfo.value)

    def test_allowed_and_blocked_domains_are_mutually_exclusive(self, tmp_path: Path) -> None:
        """The API rejects a request carrying both, so catch it at startup."""
        bad = tmp_path / "config.yaml"
        bad.write_text(
            "tools:\n"
            "  web_search:\n"
            "    allowed_domains: ['example.com']\n"
            "    blocked_domains: ['spam.example']\n"
        )
        with pytest.raises(ConfigError, match="at most one"):
            Config.load(bad)

    def test_either_domain_list_alone_is_fine(self, tmp_path: Path) -> None:
        good = tmp_path / "config.yaml"
        good.write_text("tools:\n  web_search:\n    allowed_domains: ['example.com']\n")
        assert Config.load(good).tools.web_search.allowed_domains == ["example.com"]


class TestPathResolution:
    def test_relative_paths_resolve_against_the_config_file(self, cfg: Config) -> None:
        root = cfg.source_path.parent  # type: ignore[union-attr]
        assert cfg.memory.db_path == root / "var" / "jarvis.db"
        assert cfg.tools.filesystem.workspace == root / "workspace"
        assert cfg.logging.audit_file == root / "logs" / "audit.log"

    def test_every_path_field_ends_up_absolute(self, cfg: Config) -> None:
        paths = [
            cfg.wake_word.model_dir,
            cfg.stt.model_dir,
            cfg.tts.piper.model_dir,
            cfg.memory.db_path,
            cfg.memory.embedding_model_dir,
            cfg.tools.filesystem.workspace,
            cfg.tools.shell.cwd,
            cfg.logging.audit_file,
        ]
        assert cfg.logging.file is not None
        paths.append(cfg.logging.file)
        assert all(p.is_absolute() for p in paths), [p for p in paths if not p.is_absolute()]

    def test_absolute_paths_are_left_alone(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere" / "jarvis.db"
        cfgfile = tmp_path / "config.yaml"
        cfgfile.write_text(f"memory:\n  db_path: {elsewhere}\n")
        assert Config.load(cfgfile).memory.db_path == elsewhere

    def test_tilde_is_expanded(self, tmp_path: Path) -> None:
        cfgfile = tmp_path / "config.yaml"
        cfgfile.write_text("memory:\n  db_path: ~/jarvis-test.db\n")
        assert Config.load(cfgfile).memory.db_path == Path.home() / "jarvis-test.db"

    def test_piper_derives_both_model_files_from_the_voice_name(self, cfg: Config) -> None:
        assert cfg.tts.piper.model_path.name == "en_GB-alan-medium.onnx"
        assert cfg.tts.piper.config_path.name == "en_GB-alan-medium.onnx.json"


class TestEnvOverrides:
    def test_log_level_override(self, config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JARVIS_LOG_LEVEL", "debug")
        assert Config.load(config_path).logging.level == "DEBUG"

    def test_invalid_log_level_is_rejected(
        self, config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JARVIS_LOG_LEVEL", "CHATTY")
        with pytest.raises(ConfigError, match="not a valid level"):
            Config.load(config_path)


class TestSecrets:
    def test_absent_keys_are_none(self) -> None:
        assert Secrets().anthropic_api_key is None

    def test_blank_env_value_counts_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`.env.example` ships `ANTHROPIC_API_KEY=`; that must not become a key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        assert Secrets().anthropic_api_key is None

    def test_key_is_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert Secrets().anthropic_api_key == "sk-ant-test"

    def test_redacted_never_exposes_the_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
        redacted = Secrets().redacted()
        assert redacted["anthropic_api_key"] == "set"
        assert "supersecret" not in repr(redacted)

    def test_secrets_are_excluded_from_config_dump(
        self, config_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A config dump ends up in logs and bug reports; it must not carry keys."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
        dumped = repr(Config.load(config_path).model_dump())
        assert "supersecret" not in dumped
