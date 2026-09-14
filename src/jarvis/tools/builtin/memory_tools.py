"""Remembering and forgetting, on request.

These exist so the user can correct JARVIS out loud. Automatic extraction
(milestone 5) is best-effort and will sometimes store the wrong thing; without
a way to say "no, forget that", the only remedy would be editing SQLite by hand.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.errors import ToolError
from jarvis.logging_setup import get_logger
from jarvis.tools.registry import ToolContext, tool

log = get_logger("tools.memory")


class RememberArgs(BaseModel):
    fact: str = Field(
        ...,
        description=(
            "One short, self-contained statement, written in the third person: "
            "'the user's sister is called Priya', not 'my sister is Priya'."
        ),
    )


@tool(
    name="remember_this",
    description=(
        "Store a durable fact about the user so it is available in future "
        "conversations. Use for preferences, names, and standing instructions. "
        "Do not use for things that are only true today."
    ),
    args=RememberArgs,
)
def remember_this(args: RememberArgs, ctx: ToolContext) -> str:
    if ctx.memory is None:
        raise ToolError("Memory is not available in this session.")
    fact = args.fact.strip()
    if not fact:
        raise ToolError("No fact was given.")

    stored = ctx.memory.add_fact(fact, source="explicit")
    ctx.audit.record("tool.remember_this", outcome="ok", detail=stored.text)
    return f"Remembered: {stored.text}"


class ForgetArgs(BaseModel):
    query: str = Field(..., description="What to forget, in the user's own words.")


@tool(
    name="forget_this",
    description=(
        "Delete remembered facts matching a description. Use when the user says "
        "to forget something or corrects something you remembered wrongly."
    ),
    args=ForgetArgs,
)
def forget_this(args: ForgetArgs, ctx: ToolContext) -> str:
    if ctx.memory is None:
        raise ToolError("Memory is not available in this session.")

    removed = ctx.memory.forget(args.query.strip())
    ctx.audit.record(
        "tool.forget_this",
        outcome="ok",
        detail=args.query,
        removed=[fact.text for fact in removed],
    )
    if not removed:
        return (
            "I found nothing specific enough to forget. Ask the user to say more "
            "precisely what they want removed."
        )
    if len(removed) == 1:
        return f"Forgotten: {removed[0].text}"
    return f"Forgotten {len(removed)} things, including: {removed[0].text}"
