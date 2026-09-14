"""Reading and writing files, confined to the workspace.

Every path goes through :func:`~jarvis.tools.safety.resolve_in_workspace`, which
resolves symlinks and ``..`` before comparing against the real workspace path.
Writes additionally require confirmation, shown with the full destination.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.errors import ToolError
from jarvis.logging_setup import get_logger
from jarvis.tools.registry import ToolContext, tool
from jarvis.tools.safety import resolve_in_workspace, within_size_limit

log = get_logger("tools.files")

MAX_SPOKEN_CHARS = 4000
"""Truncate what goes back to the model. A whole file in the context window is
rarely what was wanted and is expensive."""


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="Path relative to the workspace directory.")


@tool(
    name="read_file",
    description=(
        "Read a text file from the user's workspace directory. Paths are relative to "
        "that directory; anything outside it is refused."
    ),
    args=ReadFileArgs,
)
def read_file(args: ReadFileArgs, ctx: ToolContext) -> str:
    cfg = ctx.config.tools.filesystem
    target = resolve_in_workspace(args.path, cfg.workspace)

    if not target.exists():
        raise ToolError(f"There is no file at {args.path} in the workspace.")
    if target.is_dir():
        entries = sorted(child.name for child in target.iterdir())[:50]
        return f"{args.path} is a directory containing: {', '.join(entries) or 'nothing'}"

    within_size_limit(target, cfg.max_file_mb)
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"Could not read {args.path}: {exc}") from exc

    ctx.audit.record(
        "tool.read_file", outcome="ok", detail=str(target), bytes=len(content.encode("utf-8"))
    )

    if len(content) > MAX_SPOKEN_CHARS:
        return content[:MAX_SPOKEN_CHARS] + f"\n\n[truncated; {len(content)} characters in total]"
    return content or "(the file is empty)"


class WriteFileArgs(BaseModel):
    path: str = Field(..., description="Path relative to the workspace directory.")
    content: str = Field(..., description="The complete new contents of the file.")
    append: bool = Field(False, description="Append instead of replacing.")


def _write_prompt(args: WriteFileArgs) -> str:
    verb = "append to" if args.append else "write"
    lines = args.content.count("\n") + 1
    return (
        f"Shall I {verb} {args.path}? "
        f"That's {len(args.content)} characters across {lines} "
        f"line{'s' if lines != 1 else ''}."
    )


@tool(
    name="write_file",
    description=(
        "Write a text file into the user's workspace directory. Requires the user's "
        "confirmation every time. Paths outside the workspace are refused."
    ),
    args=WriteFileArgs,
    requires_confirmation=True,
    confirmation_prompt=_write_prompt,
    destructive=True,
)
def write_file(args: WriteFileArgs, ctx: ToolContext) -> str:
    cfg = ctx.config.tools.filesystem
    target = resolve_in_workspace(args.path, cfg.workspace)

    size_mb = len(args.content.encode("utf-8")) / 1024 / 1024
    if size_mb > cfg.max_file_mb:
        raise ToolError(
            f"That content is {size_mb:.1f} MB, over the {cfg.max_file_mb:.0f} MB limit."
        )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        with target.open("a" if args.append else "w", encoding="utf-8") as handle:
            handle.write(args.content)
    except OSError as exc:
        raise ToolError(f"Could not write {args.path}: {exc}") from exc

    ctx.audit.record(
        "tool.write_file",
        outcome="ok",
        detail=str(target),
        mode="append" if args.append else "replace",
        replaced_existing=existed and not args.append,
        bytes=len(args.content.encode("utf-8")),
    )
    action = "Appended to" if args.append else ("Replaced" if existed else "Created")
    return f"{action} {args.path}."


class AddNoteArgs(BaseModel):
    text: str = Field(..., description="The note to append.")


@tool(
    name="add_note",
    description=(
        "Append a dated line to the user's notes file. Use for 'make a note', "
        "'jot this down', 'remind me that'. For durable facts about the user "
        "prefer remember_this."
    ),
    args=AddNoteArgs,
)
def add_note(args: AddNoteArgs, ctx: ToolContext) -> str:
    from datetime import datetime

    cfg = ctx.config.tools
    target = resolve_in_workspace(cfg.notes.path, cfg.filesystem.workspace)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(f"- {stamp}  {args.text.strip()}\n")
    except OSError as exc:
        raise ToolError(f"Could not write the note: {exc}") from exc

    ctx.audit.record("tool.add_note", outcome="ok", detail=args.text.strip())
    return "Noted."
