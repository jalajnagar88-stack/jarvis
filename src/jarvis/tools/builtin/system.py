"""Opening applications and changing the volume.

Platform-specific by nature. Each tool degrades to a clear explanation rather
than an error when it does not know how to do its job on the current system,
because "I can't do that on Linux" is a useful answer and a traceback is not.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

from pydantic import BaseModel, Field

from jarvis.errors import ToolError
from jarvis.logging_setup import get_logger
from jarvis.tools.registry import ToolContext, tool

log = get_logger("tools.system")

TIMEOUT = 10.0


def _system() -> str:
    return platform.system()


class OpenAppArgs(BaseModel):
    name: str = Field(..., description="The application's name, e.g. 'Safari'.")


def _open_prompt(args: OpenAppArgs) -> str:
    return f"Shall I open {args.name}?"


@tool(
    name="open_app",
    description=(
        "Open an application on the user's machine. Requires confirmation. "
        "Only works on macOS and Linux desktops."
    ),
    args=OpenAppArgs,
    requires_confirmation=True,
    confirmation_prompt=_open_prompt,
    destructive=True,
)
def open_app(args: OpenAppArgs, ctx: ToolContext) -> str:
    name = args.name.strip()
    if not name:
        raise ToolError("No application name was given.")

    system = _system()
    if system == "Darwin":
        command = ["open", "-a", name]
    elif system == "Linux":
        if shutil.which("xdg-open") is None:
            raise ToolError("I can't open applications on this machine: xdg-open is missing.")
        command = ["xdg-open", name]
    else:
        raise ToolError(f"I don't know how to open applications on {system}.")

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"Could not open {name}: {exc}") from exc

    ctx.audit.record(
        "tool.open_app",
        outcome="ok" if completed.returncode == 0 else "error",
        detail=name,
        exit_code=completed.returncode,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or "the application was not found"
        raise ToolError(f"Could not open {name}: {detail}.")
    return f"Opened {name}."


class VolumeArgs(BaseModel):
    level: int | None = Field(None, ge=0, le=100, description="Target volume as a percentage.")
    change: int | None = Field(
        None, ge=-100, le=100, description="Relative change, e.g. -10 to lower it."
    )
    mute: bool | None = Field(None, description="True to mute, false to unmute.")


@tool(
    name="control_volume",
    description=(
        "Set, adjust, or mute the system volume. Give exactly one of level, change, "
        "or mute. Only works on macOS and on Linux with pactl or amixer."
    ),
    args=VolumeArgs,
)
def control_volume(args: VolumeArgs, ctx: ToolContext) -> str:
    given = [field for field in (args.level, args.change, args.mute) if field is not None]
    if len(given) != 1:
        raise ToolError("Give exactly one of level, change, or mute.")

    system = _system()
    if system == "Darwin":
        result = _macos_volume(args)
    elif system == "Linux":
        result = _linux_volume(args)
    else:
        raise ToolError(f"I can't change the volume on {system}.")

    ctx.audit.record("tool.control_volume", outcome="ok", detail=result)
    return result


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"Could not change the volume: {exc}") from exc
    if completed.returncode != 0:
        raise ToolError(
            f"Could not change the volume: {(completed.stderr or '').strip() or 'unknown error'}"
        )
    return (completed.stdout or "").strip()


def _macos_volume(args: VolumeArgs) -> str:
    if args.mute is not None:
        _run(["osascript", "-e", f"set volume {'with' if args.mute else 'without'} output muted"])
        return "Muted." if args.mute else "Unmuted."

    if args.level is not None:
        target = args.level
    else:
        current = _run(["osascript", "-e", "output volume of (get volume settings)"])
        try:
            target = max(0, min(100, int(current) + int(args.change or 0)))
        except ValueError as exc:
            raise ToolError("Could not read the current volume.") from exc

    _run(["osascript", "-e", f"set volume output volume {target}"])
    return f"Volume set to {target} percent."


def _linux_volume(args: VolumeArgs) -> str:
    if shutil.which("pactl"):
        sink = "@DEFAULT_SINK@"
        if args.mute is not None:
            _run(["pactl", "set-sink-mute", sink, "1" if args.mute else "0"])
            return "Muted." if args.mute else "Unmuted."
        if args.level is not None:
            _run(["pactl", "set-sink-volume", sink, f"{args.level}%"])
            return f"Volume set to {args.level} percent."
        delta = args.change or 0
        _run(["pactl", "set-sink-volume", sink, f"{delta:+d}%"])
        return f"Volume {'raised' if delta > 0 else 'lowered'} by {abs(delta)} percent."

    if shutil.which("amixer"):
        if args.mute is not None:
            _run(["amixer", "set", "Master", "mute" if args.mute else "unmute"])
            return "Muted." if args.mute else "Unmuted."
        if args.level is not None:
            _run(["amixer", "set", "Master", f"{args.level}%"])
            return f"Volume set to {args.level} percent."
        delta = args.change or 0
        _run(["amixer", "set", "Master", f"{abs(delta)}%{'+' if delta > 0 else '-'}"])
        return f"Volume {'raised' if delta > 0 else 'lowered'} by {abs(delta)} percent."

    raise ToolError("I can't change the volume: neither pactl nor amixer is installed.")
