"""The audit trail.

Every action JARVIS takes on the user's behalf — every tool call, every
confirmation prompt, every refusal — is appended here as one JSON object per
line, with a UTC timestamp. It is append-only, local, and never rotated
automatically: an audit log that silently discards its oldest entries is not an
audit log.

This file is the answer to "what did it actually do?". Keep it boring and
complete. Writing to it must never raise into the caller: a failed audit write
is logged and swallowed, because losing the ability to record an action is not
a reason to abort the action the user already approved.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from jarvis.logging_setup import get_logger

log = get_logger("audit")

AuditOutcome = Literal["allowed", "denied", "confirmed", "declined", "ok", "error", "info"]


class AuditLog:
    """Append-only JSONL audit sink.

    Thread-safe: the voice loop, the timer thread, and the background fact
    extractor all write to it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._warned = False

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        except OSError as exc:
            if not self._warned:
                log.error(
                    "Audit log at %s is not writable (%s). Actions continue "
                    "but are not being recorded.",
                    self.path,
                    exc,
                )
                self._warned = True

    def record(
        self,
        event: str,
        *,
        outcome: AuditOutcome = "info",
        detail: str | None = None,
        **fields: Any,
    ) -> None:
        """Append one audit entry.

        Args:
            event: Dotted event name, e.g. ``tool.run_shell`` or ``session.start``.
            outcome: What happened. ``denied`` means the safety layer refused it.
            detail: Human-readable summary, shown when reviewing the log by eye.
            **fields: Structured extras. Never pass secrets or raw audio here.
        """
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": event,
            "outcome": outcome,
        }
        if detail is not None:
            record["detail"] = detail
        record.update(fields)
        self._write(record)

    def is_writable(self) -> tuple[bool, str | None]:
        """Check writability without emitting an entry. Used by the health check."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8"):
                pass
        except OSError as exc:
            return False, str(exc)
        return True, None


class NullAuditLog(AuditLog):
    """Discards everything. For tests only — never wire this into a real run."""

    def __init__(self) -> None:
        super().__init__(Path(os.devnull))

    def _write(self, record: dict[str, Any]) -> None:
        return

    def is_writable(self) -> tuple[bool, str | None]:
        return True, None
