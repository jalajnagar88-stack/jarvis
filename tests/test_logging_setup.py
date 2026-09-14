"""Logging setup."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from jarvis.config import LoggingConfig
from jarvis.logging_setup import describe_log_targets, get_logger, setup_logging


@pytest.fixture
def log_config(tmp_path: Path) -> LoggingConfig:
    return LoggingConfig(
        level="DEBUG",
        file=tmp_path / "logs" / "jarvis.log",
        audit_file=tmp_path / "logs" / "audit.log",
    )


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Leave the root logger as we found it, or later tests see stray handlers."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


class TestSetup:
    def test_installs_console_and_file_handlers(self, log_config: LoggingConfig) -> None:
        setup_logging(log_config)
        assert len(logging.getLogger().handlers) == 2

    def test_creates_the_log_directory(self, log_config: LoggingConfig) -> None:
        setup_logging(log_config)
        assert log_config.file is not None
        assert log_config.file.parent.is_dir()

    def test_messages_reach_the_file(self, log_config: LoggingConfig) -> None:
        setup_logging(log_config)
        get_logger("test").info("hello from the test")
        logging.shutdown()
        assert log_config.file is not None
        assert "hello from the test" in log_config.file.read_text()

    def test_respects_the_configured_level(self, log_config: LoggingConfig) -> None:
        log_config.level = "WARNING"
        setup_logging(log_config)
        get_logger("test").debug("should not appear")
        get_logger("test").warning("should appear")
        logging.shutdown()
        assert log_config.file is not None
        contents = log_config.file.read_text()
        assert "should not appear" not in contents
        assert "should appear" in contents

    def test_calling_twice_does_not_duplicate_handlers(self, log_config: LoggingConfig) -> None:
        setup_logging(log_config)
        setup_logging(log_config)
        assert len(logging.getLogger().handlers) == 2

    def test_file_logging_can_be_disabled(self, log_config: LoggingConfig) -> None:
        log_config.file = None
        setup_logging(log_config)
        assert len(logging.getLogger().handlers) == 1

    def test_an_unwritable_log_file_falls_back_to_console(
        self, tmp_path: Path, log_config: LoggingConfig
    ) -> None:
        """No mic, no key, no writable disk — still must not throw."""
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        log_config.file = blocker / "jarvis.log"
        setup_logging(log_config)
        assert len(logging.getLogger().handlers) == 1

    def test_noisy_third_party_loggers_are_quietened(self, log_config: LoggingConfig) -> None:
        setup_logging(log_config)
        assert logging.getLogger("httpx").level >= logging.INFO


class TestHelpers:
    def test_get_logger_namespaces_under_jarvis(self) -> None:
        assert get_logger("agent").name == "jarvis.agent"

    def test_get_logger_does_not_double_prefix(self) -> None:
        assert get_logger("jarvis.agent").name == "jarvis.agent"

    def test_describe_log_targets_lists_both_sinks(self, log_config: LoggingConfig) -> None:
        assert describe_log_targets(log_config) == [log_config.file, log_config.audit_file]

    def test_describe_log_targets_omits_a_disabled_file(self, log_config: LoggingConfig) -> None:
        log_config.file = None
        assert describe_log_targets(log_config) == [log_config.audit_file]
