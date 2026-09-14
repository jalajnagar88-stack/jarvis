"""The safety layer.

Two independent protections, and the distinction between them matters:

* **The denylist** refuses outright. There is no confirmation that lets
  ``rm -rf /`` through, because a spoken "yes" is far too easy to produce by
  accident — a misheard word, a television in the background, a sentence that
  happened to contain "sure". Anything that could destroy a machine is simply
  not reachable through this program.
* **Confirmation** gates everything else that writes. The user is shown exactly
  what will happen and must agree. The tool cannot waive this; the dispatcher
  enforces it.

The filesystem sandbox resolves paths fully before comparing them. String
matching would be defeated by ``../``, by a symlink pointing outward, or by a
path that only looks like it is inside the workspace.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from jarvis.errors import SafetyViolation
from jarvis.logging_setup import get_logger

log = get_logger("tools.safety")


@dataclass(frozen=True, slots=True)
class DenyRule:
    """One forbidden pattern."""

    pattern: re.Pattern[str]
    reason: str


def _rule(pattern: str, reason: str) -> DenyRule:
    return DenyRule(re.compile(pattern, re.IGNORECASE), reason)


# Refused outright, with no confirmation path. Deliberately conservative: a
# false positive costs one inconvenienced command, a false negative can cost a
# machine. Patterns match the normalised command string.
DENYLIST: tuple[DenyRule, ...] = (
    _rule(
        r"\brm\b[^|;&]*\s-[a-z]*[rR][a-z]*f|\brm\b[^|;&]*\s-[a-z]*f[a-z]*[rR]",
        "recursive forced delete",
    ),
    _rule(r"\brm\b[^|;&]*\s+/(\s|$)", "delete of the root directory"),
    _rule(r"\bdd\b[^|;&]*\bof=/dev/", "raw write to a block device"),
    _rule(r"\bmkfs(\.\w+)?\b", "filesystem format"),
    _rule(r"\bfdisk\b|\bparted\b|\bdiskutil\s+(erase|partition|reformat)", "disk partitioning"),
    _rule(r":\(\)\s*\{.*\|.*&.*\}\s*;?\s*:", "fork bomb"),
    _rule(
        r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|k|fi)?sh\b",
        "piping a download straight into a shell",
    ),
    _rule(r"\bchmod\b\s+(-[a-zA-Z]+\s+)*777\s+/(\s|$)", "world-writable root"),
    _rule(
        r"\bchown\b[^|;&]*\s+-[a-zA-Z]*R[a-zA-Z]*\s[^|;&]*\s/(\s|$)",
        "recursive ownership change on root",
    ),
    _rule(r">\s*/dev/(sd|nvme|disk|hd)", "overwriting a disk device"),
    _rule(r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b", "shutting down the machine"),
    _rule(r"\bkillall\b\s+-9|\bkill\b\s+-9\s+1\b", "indiscriminate force kill"),
    _rule(r"\bsudo\b", "elevated privileges"),
    _rule(r"\bhistory\s+-c\b|\b>\s*~?/?\.bash_history", "erasing shell history"),
    _rule(r"\b(launchctl|systemctl)\s+(disable|unload|stop)\b", "disabling a system service"),
    _rule(r"\bdefaults\s+delete\b", "deleting system preferences"),
    _rule(r"\bcrontab\s+-r\b", "deleting all scheduled jobs"),
    _rule(r"\b(nc|netcat|ncat)\b[^|;&]*\s-[a-z]*e", "opening a remote shell"),
)


def check_command(command: str, extra_patterns: list[str] | None = None) -> None:
    """Refuse a command that matches the denylist.

    Raises:
        SafetyViolation: the command is forbidden. Never let this be caught and
            retried with confirmation -- that is the whole point of it being
            separate from the confirmation gate.
    """
    normalised = " ".join(command.split())
    if not normalised:
        raise SafetyViolation("An empty command was refused.")

    for rule in DENYLIST:
        if rule.pattern.search(normalised):
            log.warning("Refused a command matching the denylist (%s): %s", rule.reason, normalised)
            raise SafetyViolation(
                f"That command is blocked: it looks like {rule.reason}. "
                "This is refused outright, not something I can confirm."
            )

    for extra in extra_patterns or []:
        try:
            if re.search(extra, normalised, re.IGNORECASE):
                log.warning("Refused a command matching a configured pattern: %s", normalised)
                raise SafetyViolation("That command is blocked by a pattern in your configuration.")
        except re.error:
            log.error("tools.shell.extra_denylist contains an invalid regex: %r", extra)


def looks_like_shell_metacharacters(command: str) -> bool:
    """True if the command uses shell features that broaden what it can do.

    Not forbidden, but worth surfacing in the confirmation prompt: a pipeline
    or a chained command does more than it appears to at a glance.
    """
    try:
        lexed = shlex.split(command, comments=True)
    except ValueError:
        return True
    return any(token in {"|", "||", "&&", ";", ">", ">>", "<", "&"} for token in lexed) or any(
        ch in command for ch in "|;&><`$("
    )


def resolve_in_workspace(candidate: str | Path, workspace: Path) -> Path:
    """Resolve ``candidate`` and confirm it is inside ``workspace``.

    Both sides are fully resolved first, so symlinks are followed and ``..`` is
    collapsed before the comparison. Checking the string would let a symlink
    inside the workspace point anywhere on disk.

    Raises:
        SafetyViolation: the path escapes the workspace.
    """
    root = workspace.expanduser().resolve()
    raw = Path(candidate).expanduser()
    target = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()

    if target != root and root not in target.parents:
        log.warning("Refused filesystem access outside the workspace: %s", target)
        raise SafetyViolation(
            f"That path is outside the workspace. I can only read and write inside {root}.",
            remedy=f"Move the file into {root}, or change tools.filesystem.workspace.",
        )
    return target


def within_size_limit(path: Path, max_megabytes: float) -> None:
    """Refuse a file larger than the configured ceiling."""
    try:
        size = path.stat().st_size
    except OSError:
        return
    limit = int(max_megabytes * 1024 * 1024)
    if size > limit:
        raise SafetyViolation(
            f"{path.name} is {size / 1024 / 1024:.1f} MB, over the {max_megabytes:.0f} MB limit.",
            remedy="Raise tools.filesystem.max_file_mb if that is deliberate.",
        )
