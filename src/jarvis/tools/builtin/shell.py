"""Running shell commands, behind two independent gates.

The denylist refuses destructive commands outright. Everything else is shown to
the user in full and run only if they agree. Neither gate can be disabled from
configuration: ``tools.shell.require_confirmation`` is recorded in the audit log
for completeness, but the dispatcher does not consult it.
"""

from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from jarvis.errors import ToolError
from jarvis.logging_setup import get_logger
from jarvis.tools.registry import ToolContext, tool
from jarvis.tools.safety import check_command, looks_like_shell_metacharacters

log = get_logger("tools.shell")

MAX_OUTPUT_CHARS = 2000


class RunShellArgs(BaseModel):
    command: str = Field(..., description="The exact command line to run.")
    reason: str = Field("", description="One short phrase explaining why, shown to the user.")


def _shell_prompt(args: RunShellArgs) -> str:
    """The confirmation question.

    Always quotes the command verbatim. Paraphrasing it would defeat the point:
    the user is agreeing to *this* command, not to a description of it.
    """
    question = f"Shall I run: {args.command}"
    if args.reason:
        question += f"  ({args.reason})"
    if looks_like_shell_metacharacters(args.command):
        question += ". Note that it uses shell operators, so it does more than one thing"
    return question + "?"


@tool(
    name="run_shell",
    description=(
        "Run a shell command on the user's machine and return its output. The user "
        "must approve every command. Destructive commands are refused outright, so "
        "do not attempt them. Prefer read_file and the other tools where they fit."
    ),
    args=RunShellArgs,
    requires_confirmation=True,
    confirmation_prompt=_shell_prompt,
    destructive=True,
)
def run_shell(args: RunShellArgs, ctx: ToolContext) -> str:
    cfg = ctx.config.tools.shell

    # Checked again here, not only at dispatch. A tool that is safe only because
    # of its caller is not safe.
    check_command(args.command, cfg.extra_denylist)

    cwd = cfg.cwd
    cwd.mkdir(parents=True, exist_ok=True)

    log.info("Running: %s", args.command)
    try:
        # shell=True is deliberate: the user approved this exact string, and
        # pipelines and redirection are the point of a shell tool.
        completed = subprocess.run(
            args.command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        ctx.audit.record("tool.run_shell", outcome="error", detail=args.command, error="timeout")
        raise ToolError(
            f"That command was still running after {cfg.timeout_seconds:.0f} seconds, "
            "so I stopped it."
        ) from exc
    except OSError as exc:
        ctx.audit.record("tool.run_shell", outcome="error", detail=args.command, error=str(exc))
        raise ToolError(f"Could not run that command: {exc}") from exc

    ctx.audit.record(
        "tool.run_shell",
        outcome="ok" if completed.returncode == 0 else "error",
        detail=args.command,
        exit_code=completed.returncode,
        cwd=str(cwd),
    )
    return _format(completed)


def _format(completed: subprocess.CompletedProcess[str]) -> str:
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    sections = []
    if stdout:
        sections.append(_truncate(stdout))
    if stderr:
        sections.append(f"Standard error:\n{_truncate(stderr)}")
    if completed.returncode != 0:
        sections.append(f"The command exited with status {completed.returncode}.")
    elif not sections:
        sections.append("The command succeeded and printed nothing.")
    return "\n\n".join(sections)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[truncated; {len(text)} characters in total]"
