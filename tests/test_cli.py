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
    def test_reports_honestly_that_it_is_not_built_yet(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Better an honest 'not yet' than a half-wired loop that fails obscurely."""
        assert main(["--config", str(config_path), "run"]) == 0
        assert "not built yet" in capsys.readouterr().out

    def test_text_mode_names_itself(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--config", str(config_path), "run", "--text"])
        assert "text REPL" in capsys.readouterr().out


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
