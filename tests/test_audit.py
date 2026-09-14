"""The audit trail.

Two properties matter: every action produces exactly one parseable line, and a
broken audit sink never propagates an exception into the caller.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from jarvis.audit import AuditLog, NullAuditLog


@pytest.fixture
def audit(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "logs" / "audit.log")


def read_entries(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestWriting:
    def test_creates_the_directory_on_first_write(self, audit: AuditLog) -> None:
        assert not audit.path.parent.exists()
        audit.record("session.start")
        assert audit.path.is_file()

    def test_one_record_is_one_json_line(self, audit: AuditLog) -> None:
        audit.record("tool.get_time", outcome="ok", detail="2026-09-14T12:00:00Z")
        entries = read_entries(audit.path)
        assert len(entries) == 1
        assert entries[0]["event"] == "tool.get_time"
        assert entries[0]["outcome"] == "ok"
        assert entries[0]["detail"] == "2026-09-14T12:00:00Z"

    def test_every_record_is_timestamped_in_utc(self, audit: AuditLog) -> None:
        audit.record("session.start")
        assert str(read_entries(audit.path)[0]["ts"]).endswith("+00:00")

    def test_appends_rather_than_truncates(self, audit: AuditLog) -> None:
        for index in range(5):
            audit.record("tool.add_note", detail=f"note {index}")
        assert len(read_entries(audit.path)) == 5

    def test_survives_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.log"
        AuditLog(path).record("session.start")
        AuditLog(path).record("session.start")
        assert len(read_entries(path)) == 2

    def test_structured_extras_are_preserved(self, audit: AuditLog) -> None:
        audit.record("tool.run_shell", outcome="denied", command="rm -rf /", exit_code=None)
        entry = read_entries(audit.path)[0]
        assert entry["command"] == "rm -rf /"
        assert entry["exit_code"] is None

    def test_unserialisable_values_do_not_lose_the_record(self, audit: AuditLog) -> None:
        """Better a stringified object in the log than a missing entry."""
        audit.record("tool.read_file", path=Path("/tmp/x"))
        assert read_entries(audit.path)[0]["path"] == "/tmp/x"

    def test_non_ascii_is_written_readably(self, audit: AuditLog) -> None:
        audit.record("memory.fact", detail="sister is called Priya — prefers °C")
        assert "°C" in audit.path.read_text(encoding="utf-8")


class TestDenialsAreRecorded:
    def test_a_refused_command_is_auditable(self, audit: AuditLog) -> None:
        """A denial is the single most important thing the audit log records."""
        audit.record(
            "tool.run_shell",
            outcome="denied",
            detail="matched denylist pattern 'rm -rf'",
            command="rm -rf ~/Documents",
        )
        entry = read_entries(audit.path)[0]
        assert entry["outcome"] == "denied"
        assert "rm -rf" in str(entry["command"])

    def test_a_declined_confirmation_is_distinguishable_from_a_denial(
        self, audit: AuditLog
    ) -> None:
        audit.record("tool.write_file", outcome="declined", detail="user said no")
        audit.record("tool.run_shell", outcome="denied", detail="denylist")
        outcomes = [e["outcome"] for e in read_entries(audit.path)]
        assert outcomes == ["declined", "denied"]


class TestResilience:
    def test_an_unwritable_path_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Losing the ability to record must not abort an approved action."""
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        audit = AuditLog(blocker / "audit.log")
        with caplog.at_level(logging.ERROR):
            audit.record("session.start")
        assert "not writable" in caplog.text

    def test_it_complains_only_once(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        audit = AuditLog(blocker / "audit.log")
        with caplog.at_level(logging.ERROR):
            for _ in range(10):
                audit.record("session.start")
        assert caplog.text.count("not writable") == 1

    def test_is_writable_reports_without_writing_an_entry(self, audit: AuditLog) -> None:
        ok, reason = audit.is_writable()
        assert ok and reason is None
        assert audit.path.read_text() == ""

    def test_is_writable_detects_a_blocked_path(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        ok, reason = AuditLog(blocker / "audit.log").is_writable()
        assert not ok and reason

    def test_concurrent_writers_do_not_interleave(self, audit: AuditLog) -> None:
        """The voice loop, timers, and the fact extractor all write at once."""

        def worker(index: int) -> None:
            for step in range(20):
                audit.record("tool.set_timer", detail=f"{index}-{step}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        entries = read_entries(audit.path)  # raises if any line is torn
        assert len(entries) == 160


class TestNullAuditLog:
    def test_discards_everything(self) -> None:
        null = NullAuditLog()
        null.record("tool.run_shell", outcome="denied")
        assert null.is_writable() == (True, None)
