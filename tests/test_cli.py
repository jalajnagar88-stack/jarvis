"""Command line behaviour.

Exit codes matter here: they are what a launch agent or a CI job will read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis import health
from jarvis.cli import build_parser, main
from jarvis.config import Config
from jarvis.interfaces.audio import DeviceInfo


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI reconfigures the root logger; don't let that leak between tests."""
    monkeypatch.setattr("jarvis.cli.setup_logging", lambda *a, **k: None)


class TestParser:
    def test_health_is_the_default_command(self) -> None:
        assert build_parser().parse_args([]).command is None

    def test_accepts_a_config_path(self) -> None:
        args = build_parser().parse_args(["--config", "/tmp/x.yaml"])
        assert args.config == "/tmp/x.yaml"

    def test_text_flag_lives_on_run(self) -> None:
        assert build_parser().parse_args(["run", "--text"]).text is True

    def test_rejects_an_unknown_log_level(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--log-level", "SHOUTING"])


class TestHealthCommand:
    def test_missing_config_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--config", str(tmp_path / "absent.yaml")]) == 2
        assert "Configuration error" in capsys.readouterr().out

    def test_invalid_config_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("llm:\n  max_tokens: -5\n")
        assert main(["--config", str(bad)]) == 2

    def test_incomplete_install_exits_1(self, config_path: Path) -> None:
        """A fresh clone has no models, so the health check must report failure."""
        assert main(["--config", str(config_path)]) == 1

    def test_fully_provisioned_exits_0(
        self,
        stub_models: Config,
        fake_devices: list[DeviceInfo],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(health, "_importable", lambda module: True)
        monkeypatch.setattr(health, "probe_audio_backend", lambda: (fake_devices, None))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert main(["--config", str(stub_models.source_path)]) == 0

    def test_warnings_alone_do_not_fail(
        self,
        stub_models: Config,
        fake_devices: list[DeviceInfo],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing embeddings is a warning; it must not break the exit code."""
        monkeypatch.setattr(health, "_importable", lambda module: module != "sentence_transformers")
        monkeypatch.setattr(health, "probe_audio_backend", lambda: (fake_devices, None))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert main(["--config", str(stub_models.source_path)]) == 0

    def test_report_names_every_subsystem(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--config", str(config_path)])
        out = capsys.readouterr().out
        for subsystem in (
            "Runtime",
            "Audio",
            "Wake word",
            "Speech to text",
            "Brain",
            "Text to speech",
            "Memory",
            "Tools",
            "Logging",
            "Privacy",
        ):
            assert subsystem in out, f"{subsystem} missing from the report"

    def test_report_states_the_privacy_posture(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--config", str(config_path)])
        assert "Audio never leaves the machine" in capsys.readouterr().out

    def test_report_never_prints_the_api_key(
        self, config_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-hunter2secretvalue")
        main(["--config", str(config_path)])
        assert "hunter2secretvalue" not in capsys.readouterr().out

    def test_health_check_is_audited(self, config_path: Path) -> None:
        main(["--config", str(config_path)])
        audit_file = config_path.parent / "logs" / "audit.log"
        assert "health.check" in audit_file.read_text()


class TestJsonOutput:
    def test_emits_valid_json(self, config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--config", str(config_path), "health", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["overall"] in {"ok", "warn", "fail"}
        assert payload["checks"]
        assert payload["capabilities"]

    def test_every_check_has_the_expected_shape(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--config", str(config_path), "health", "--json"])
        payload = json.loads(capsys.readouterr().out)
        for check in payload["checks"]:
            assert set(check) == {"id", "subsystem", "name", "status", "detail", "remedy"}

    def test_json_mode_uses_the_same_exit_code(self, config_path: Path) -> None:
        assert main(["--config", str(config_path), "health", "--json"]) == 1


class TestDevicesCommand:
    def test_lists_devices(
        self,
        config_path: Path,
        fake_devices: list[DeviceInfo],
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("jarvis.cli.list_audio_devices", lambda: fake_devices)
        assert main(["--config", str(config_path), "devices"]) == 0
        out = capsys.readouterr().out
        assert "Jabra Evolve 65" in out
        assert "MacBook Pro Microphone" in out

    def test_no_devices_explains_the_two_likely_causes(
        self, config_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("jarvis.cli.list_audio_devices", list)
        assert main(["--config", str(config_path), "devices"]) == 1
        out = capsys.readouterr().out
        assert "--extra voice" in out
        assert "Privacy & Security" in out


class TestRunCommand:
    def test_refuses_to_listen_when_models_are_missing(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Naming the blockers up front beats a ten-second model load then a crash."""
        assert main(["--config", str(config_path), "run"]) == 1
        out = capsys.readouterr().out
        assert "Not ready to listen yet" in out
        assert "Traceback" not in out

    def test_refusal_points_at_text_mode(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--config", str(config_path), "run"])
        assert "--text" in capsys.readouterr().out

    def test_text_mode_survives_a_closed_stdin(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Piped from /dev/null or run under a scheduler: leave quietly."""
        assert main(["--config", str(config_path), "run", "--text"]) == 0
        out = capsys.readouterr().out
        assert "Traceback" not in out
        assert "Goodbye" in out

    def test_text_mode_degrades_when_there_is_no_voice(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No Piper voice downloaded is a downgrade to text-only, not a failure."""
        assert main(["--config", str(config_path), "run", "--text"]) == 0
        assert "text-only mode" in capsys.readouterr().out


class TestGracefulDegradation:
    def test_no_mic_no_key_no_models_still_produces_a_report(
        self, config_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worst case must be a readable report, never a stack trace."""
        monkeypatch.setattr(health, "_importable", lambda module: False)
        monkeypatch.setattr(health, "probe_audio_backend", lambda: ([], "no backend"))
        assert main(["--config", str(config_path)]) == 1
        out = capsys.readouterr().out
        assert "Traceback" not in out
        assert "To fix" in out


class TestFactsCommand:
    def test_an_empty_memory_says_so(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--config", str(config_path), "facts"]) == 0
        assert "Nothing remembered yet" in capsys.readouterr().out

    def test_stored_facts_are_listed(self, cfg: Config, capsys: pytest.CaptureFixture[str]) -> None:
        from jarvis.memory.store import SqliteMemory

        memory = SqliteMemory(cfg.memory.db_path)
        memory.initialise()
        memory.add_fact("the user's sister is called Priya", source="explicit")
        memory.close()

        assert main(["--config", str(cfg.source_path), "facts"]) == 0
        out = capsys.readouterr().out
        assert "Priya" in out
        assert "explicit" in out

    def test_forget_all_requires_typing_yes(
        self,
        cfg: Config,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Deleting everything is irreversible; a stray keypress must not do it."""
        from rich.console import Console

        from jarvis.memory.store import SqliteMemory

        memory = SqliteMemory(cfg.memory.db_path)
        memory.initialise()
        memory.add_fact("the user prefers tea")
        memory.close()

        monkeypatch.setattr(Console, "input", lambda self, prompt="": "y")
        assert main(["--config", str(cfg.source_path), "facts", "--forget-all"]) == 0
        assert "Left alone" in capsys.readouterr().out

        reopened = SqliteMemory(cfg.memory.db_path)
        reopened.initialise()
        assert reopened.count_facts() == 1
