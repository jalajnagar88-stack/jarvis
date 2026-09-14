"""Logging setup.

Two sinks, deliberately separate:

* the **application log** (``logging.file``) — what the code did, for debugging;
* the **audit log** (:mod:`jarvis.audit`) — what JARVIS did on your behalf.

Mixing them makes the audit trail unreadable exactly when you need it. The
console handler uses rich so that a health check is scannable at a glance.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from jarvis.config import LoggingConfig

_CONFIGURED = False


def setup_logging(cfg: LoggingConfig, *, console: Console | None = None) -> None:
    """Install console and rotating-file handlers on the root logger.

    Safe to call more than once; later calls replace the handlers rather than
    stacking them, which otherwise produces duplicated lines in tests.
    """
    global _CONFIGURED

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(cfg.level)

    console_handler = RichHandler(
        console=console or Console(stderr=True),
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        omit_repeated_times=False,
        markup=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    root.addHandler(console_handler)

    if cfg.file is not None:
        try:
            cfg.file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                cfg.file,
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S%z",
                )
            )
            root.addHandler(file_handler)
        except OSError as exc:
            # A missing log file must never stop the assistant from running.
            root.warning("Could not open log file %s (%s). Logging to console only.", cfg.file, exc)

    # These libraries are chatty at DEBUG and none of it is ours.
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "numba", "filelock"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, root.level))

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Always use this rather than ``logging.getLogger``."""
    return logging.getLogger(f"jarvis.{name}" if not name.startswith("jarvis") else name)


def is_configured() -> bool:
    return _CONFIGURED


def describe_log_targets(cfg: LoggingConfig) -> list[Path]:
    """Files that will be written to, for the health check to report."""
    targets = [cfg.audit_file]
    if cfg.file is not None:
        targets.insert(0, cfg.file)
    return targets
