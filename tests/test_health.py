"""Health checks.

The contract under test: a health check never raises, never loads a model, and
always attaches an actionable remedy to anything it marks as broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis import health
from jarvis.config import Config
from jarvis.health import Status, run_health_checks
from jarvis.interfaces.audio import DeviceInfo


@pytest.fixture
def all_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend every optional dependency is present."""
    monkeypatch.setattr(health, "_importable", lambda module: True)
    monkeypatch.setattr(health, "_version_of", lambda module: "1.2.3")


@pytest.fixture
def none_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend no optional dependency is present (the fresh-clone case)."""
    core = {"anthropic", "yaml", "pydantic", "rich", "numpy"}
    monkeypatch.setattr(health, "_importable", lambda module: module in core)


class TestOverallBehaviour:
    def test_never_raises_even_with_a_hostile_config(self, cfg: Config) -> None:
        cfg.wake_word.model_dir = Path("/definitely/not/here")
        cfg.stt.model_dir = Path("/definitely/not/here")
        cfg.memory.db_path = Path("/definitely/not/here/jarvis.db")
        report = run_health_checks(cfg)
        assert report.checks

    def test_a_check_that_explodes_becomes_a_failure_not_a_crash(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_: Config) -> None:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(health, "_SECTIONS", (boom,))
        report = run_health_checks(cfg)
        assert report.overall is Status.FAIL
        assert "kaboom" in report.checks[0].detail

    def test_every_failure_carries_a_remedy(self, cfg: Config, none_installed: None) -> None:
        """A failure without a fix is just bad news. Every one must be actionable."""
        report = run_health_checks(cfg)
        failures = [c for c in report.checks if c.status is Status.FAIL]
        assert failures
        missing = [c.id for c in failures if not c.remedy]
        assert not missing, f"failing checks with no remedy: {missing}"

    def test_check_ids_are_unique(self, cfg: Config) -> None:
        ids = [c.id for c in run_health_checks(cfg).checks]
        assert len(ids) == len(set(ids))

    def test_overall_is_the_worst_status(self, cfg: Config) -> None:
        report = run_health_checks(cfg)
        has_fail = any(c.status is Status.FAIL for c in report.checks)
        assert (report.overall is Status.FAIL) == has_fail

    def test_skip_does_not_make_the_report_unhealthy(self, cfg: Config) -> None:
        report = health.HealthReport(checks=[health.Check("a", "S", "n", Status.SKIP, "skipped")])
        assert report.overall is Status.OK


class TestAudio:
    def test_reports_the_backend_as_missing(self, cfg: Config, none_installed: None) -> None:
        report = run_health_checks(cfg)
        backend = report.by_id("audio.backend")
        assert backend is not None and backend.status is Status.FAIL
        assert "sounddevice is not installed" in backend.detail
        assert backend.remedy == "uv sync --extra voice"
        # Device checks are skipped rather than failed: we genuinely don't know.
        for check_id in ("audio.input", "audio.output"):
            check = report.by_id(check_id)
            assert check is not None and check.status is Status.SKIP

    def test_finds_the_default_microphone_and_speaker(
        self,
        cfg: Config,
        all_installed: None,
        fake_devices: list[DeviceInfo],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(health, "probe_audio_backend", lambda: (fake_devices, None))
        report = run_health_checks(cfg)
        mic = report.by_id("audio.input")
        speaker = report.by_id("audio.output")
        assert mic is not None and mic.status is Status.OK
        assert "MacBook Pro Microphone" in mic.detail
        assert speaker is not None and speaker.status is Status.OK
        assert "MacBook Pro Speakers" in speaker.detail

    def test_missing_native_portaudio_is_distinguished_from_missing_hardware(
        self, cfg: Config, all_installed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`pip install sounddevice` succeeding does not mean PortAudio is present.

        This is the single most common macOS setup failure, and "no devices
        found" would send the user looking at their microphone instead of brew.
        """
        monkeypatch.setattr(
            health,
            "probe_audio_backend",
            lambda: (
                [],
                "sounddevice is installed but its native library will not load "
                "(PortAudio library not found)",
            ),
        )
        backend = run_health_checks(cfg).by_id("audio.backend")
        assert backend is not None and backend.status is Status.FAIL
        assert backend.remedy is not None and "brew install portaudio" in backend.remedy

    def test_working_backend_with_no_hardware_says_so(
        self, cfg: Config, all_installed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(health, "probe_audio_backend", lambda: ([], None))
        backend = run_health_checks(cfg).by_id("audio.backend")
        assert backend is not None and backend.status is Status.FAIL
        assert "no audio devices" in backend.detail
        assert backend.remedy is not None and "Connect a microphone" in backend.remedy

    def test_a_device_selector_that_matches_nothing_is_a_failure(
        self,
        cfg: Config,
        all_installed: None,
        fake_devices: list[DeviceInfo],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(health, "probe_audio_backend", lambda: (fake_devices, None))
        cfg.audio.input_device = "Nonexistent Microphone"
        mic = run_health_checks(cfg).by_id("audio.input")
        assert mic is not None and mic.status is Status.FAIL

    def test_list_audio_devices_is_empty_without_the_backend(self, none_installed: None) -> None:
        assert health.list_audio_devices() == []

    def test_probe_names_the_reason_when_the_package_is_absent(self, none_installed: None) -> None:
        devices, error = health.probe_audio_backend()
        assert devices == []
        assert error is not None and "not installed" in error


class TestDeviceSelection:
    def test_none_prefers_the_system_default(self, fake_devices: list[DeviceInfo]) -> None:
        inputs = [d for d in fake_devices if d.max_input_channels]
        chosen = health._pick_device(inputs, None, want_input=True)
        assert chosen is not None and chosen.name == "MacBook Pro Microphone"

    def test_integer_selects_by_index(self, fake_devices: list[DeviceInfo]) -> None:
        chosen = health._pick_device(fake_devices, 2, want_input=True)
        assert chosen is not None and chosen.name == "Jabra Evolve 65"

    def test_string_selects_by_case_insensitive_substring(
        self, fake_devices: list[DeviceInfo]
    ) -> None:
        chosen = health._pick_device(fake_devices, "jabra", want_input=True)
        assert chosen is not None and chosen.index == 2

    def test_unmatched_selector_returns_none(self, fake_devices: list[DeviceInfo]) -> None:
        assert health._pick_device(fake_devices, "aardvark", want_input=True) is None
        assert health._pick_device(fake_devices, 99, want_input=True) is None


class TestModelDiscovery:
    def test_wake_word_model_missing(self, cfg: Config, all_installed: None) -> None:
        check = run_health_checks(cfg).by_id("wake.model")
        assert check is not None and check.status is Status.FAIL

    def test_wake_word_model_present(self, stub_models: Config, all_installed: None) -> None:
        check = run_health_checks(stub_models).by_id("wake.model")
        assert check is not None and check.status is Status.OK

    def test_wake_word_without_feature_extractors_fails(
        self, stub_models: Config, all_installed: None
    ) -> None:
        """A wake model alone loads and then fails at runtime. Catch it here."""
        (stub_models.wake_word.model_dir / "melspectrogram.onnx").unlink()
        check = run_health_checks(stub_models).by_id("wake.model")
        assert check is not None and check.status is Status.FAIL
        assert "melspectrogram" in check.detail

    def test_wake_word_disabled_skips_both_checks(self, cfg: Config, all_installed: None) -> None:
        cfg.wake_word.enabled = False
        report = run_health_checks(cfg)
        for check_id in ("wake.package", "wake.model"):
            check = report.by_id(check_id)
            assert check is not None and check.status is Status.SKIP

    def test_whisper_model_found_in_a_huggingface_cache_layout(
        self, stub_models: Config, all_installed: None
    ) -> None:
        """faster-whisper nests model.bin several levels down; find it anyway."""
        check = run_health_checks(stub_models).by_id("stt.model")
        assert check is not None and check.status is Status.OK

    def test_whisper_model_found_in_a_flat_layout(self, cfg: Config, all_installed: None) -> None:
        cfg.stt.model_dir.mkdir(parents=True, exist_ok=True)
        (cfg.stt.model_dir / "model.bin").write_bytes(b"")
        check = run_health_checks(cfg).by_id("stt.model")
        assert check is not None and check.status is Status.OK

    def test_piper_voice_present(self, stub_models: Config, all_installed: None) -> None:
        check = run_health_checks(stub_models).by_id("tts.voice")
        assert check is not None and check.status is Status.OK

    def test_piper_voice_without_its_sidecar_fails(
        self, stub_models: Config, all_installed: None
    ) -> None:
        stub_models.tts.piper.config_path.unlink()
        check = run_health_checks(stub_models).by_id("tts.voice")
        assert check is not None and check.status is Status.FAIL
        assert "sidecar" in check.detail


class TestBrain:
    def test_missing_key_is_a_failure_that_explains_the_consequence(
        self, cfg: Config, all_installed: None
    ) -> None:
        check = run_health_checks(cfg).by_id("brain.key")
        assert check is not None and check.status is Status.FAIL
        assert "cannot think" in check.detail

    def test_present_key_is_never_printed_in_full(
        self, config_path: Path, all_installed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefghijklmnop")
        check = run_health_checks(Config.load(config_path)).by_id("brain.key")
        assert check is not None and check.status is Status.OK
        assert "abcdefghij" not in check.detail
        assert check.detail.endswith("mnop)")

    def test_base_url_override_is_surfaced_as_a_warning(
        self, config_path: Path, all_installed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.internal")
        check = run_health_checks(Config.load(config_path)).by_id("brain.endpoint")
        assert check is not None and check.status is Status.WARN


class TestTtsEngineSelection:
    def test_elevenlabs_without_a_key_fails(self, cfg: Config, all_installed: None) -> None:
        cfg.tts.engine = "elevenlabs"
        cfg.tts.elevenlabs.voice_id = "abc123"
        check = run_health_checks(cfg).by_id("tts.voice")
        assert check is not None and check.status is Status.FAIL
        assert "ELEVENLABS_API_KEY" in check.detail

    def test_elevenlabs_is_flagged_as_sending_text_off_machine(
        self, config_path: Path, all_installed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
        cfg = Config.load(config_path)
        cfg.tts.engine = "elevenlabs"
        cfg.tts.elevenlabs.voice_id = "abc123"
        report = run_health_checks(cfg)
        engine = report.by_id("tts.engine")
        privacy = report.by_id("privacy.audio")
        assert engine is not None and engine.status is Status.WARN
        assert privacy is not None and privacy.status is Status.WARN
        assert "ElevenLabs" in privacy.detail

    def test_piper_is_reported_as_fully_offline(
        self, stub_models: Config, all_installed: None
    ) -> None:
        privacy = run_health_checks(stub_models).by_id("privacy.audio")
        assert privacy is not None and privacy.status is Status.OK
        assert "never leaves the machine" in privacy.detail


class TestMemoryAndTools:
    def test_missing_embeddings_is_a_warning_not_a_failure(
        self, cfg: Config, none_installed: None
    ) -> None:
        """Memory degrades to keyword matching; that is not a broken install."""
        check = run_health_checks(cfg).by_id("memory.embeddings")
        assert check is not None and check.status is Status.WARN

    def test_database_directory_is_created(self, cfg: Config) -> None:
        check = run_health_checks(cfg).by_id("memory.database")
        assert check is not None and check.status is Status.OK
        assert cfg.memory.db_path.parent.is_dir()

    def test_unwritable_database_directory_fails(self, cfg: Config) -> None:
        blocker = cfg.memory.db_path.parent
        blocker.parent.mkdir(parents=True, exist_ok=True)
        blocker.write_text("I am a file, not a directory")
        check = run_health_checks(cfg).by_id("memory.database")
        assert check is not None and check.status is Status.FAIL

    def test_workspace_is_created_and_reported(self, cfg: Config) -> None:
        check = run_health_checks(cfg).by_id("tools.workspace")
        assert check is not None and check.status is Status.OK
        assert cfg.tools.filesystem.workspace.is_dir()

    def test_a_misspelled_tool_name_is_flagged(self, cfg: Config) -> None:
        cfg.tools.enabled = ["get_time", "get_wether"]
        check = run_health_checks(cfg).by_id("tools.enabled")
        assert check is not None and check.status is Status.WARN
        assert "get_wether" in check.detail

    def test_shipped_config_enables_only_known_tools(self, cfg: Config) -> None:
        check = run_health_checks(cfg).by_id("tools.enabled")
        assert check is not None and check.status is Status.OK


class TestCapabilities:
    def test_nothing_is_available_on_a_fresh_clone(self, cfg: Config, none_installed: None) -> None:
        caps = run_health_checks(cfg).capabilities
        assert not any(available for available, _ in caps.values())

    def test_everything_is_available_when_fully_provisioned(
        self,
        stub_models: Config,
        all_installed: None,
        fake_devices: list[DeviceInfo],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(health, "probe_audio_backend", lambda: (fake_devices, None))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        stub_models.secrets.anthropic_api_key = "sk-ant-test"
        stub_models.memory.embedding_model_dir.mkdir(parents=True, exist_ok=True)
        (stub_models.memory.embedding_model_dir / "placeholder").write_text("")

        caps = run_health_checks(stub_models).capabilities
        unavailable = [name for name, (ok, _) in caps.items() if not ok]
        assert not unavailable, unavailable

    def test_text_mode_is_available_without_any_audio(
        self, cfg: Config, none_installed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of --text: no microphone required."""
        monkeypatch.setattr(health, "_importable", lambda module: module == "anthropic")
        cfg.secrets.anthropic_api_key = "sk-ant-test"
        caps = run_health_checks(cfg).capabilities
        assert caps["Text conversation (--text)"][0]
        assert not caps["Full voice loop"][0]


class TestHelpers:
    def test_importable_is_true_for_the_standard_library(self) -> None:
        assert health._importable("json")

    def test_importable_is_false_for_nonsense(self) -> None:
        assert not health._importable("definitely_not_a_real_module_xyz")

    def test_importable_does_not_import_the_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A health check must stay cheap; importing torch would defeat that."""
        import sys

        monkeypatch.delitem(sys.modules, "wave", raising=False)
        health._importable("wave")
        assert "wave" not in sys.modules

    @pytest.mark.parametrize(
        ("size", "expected"),
        [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")],
    )
    def test_human_size(self, size: int, expected: str) -> None:
        assert health._human_size(size) == expected

    def test_dir_writable_creates_missing_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        ok, reason = health._dir_writable(target)
        assert ok and reason is None and target.is_dir()

    def test_dir_writable_reports_a_file_in_the_way(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        ok, reason = health._dir_writable(blocker)
        assert not ok and reason
