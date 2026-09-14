"""Time and timers."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from jarvis.agent.prompt import resolve_timezone
from jarvis.errors import ToolError
from jarvis.logging_setup import get_logger
from jarvis.tools.registry import ToolContext, tool

log = get_logger("tools.clock")


class GetTimeArgs(BaseModel):
    timezone: str | None = Field(
        None,
        description="IANA timezone name such as Europe/London. Omit for the user's own.",
    )


@tool(
    name="get_time",
    description=(
        "The current date and time. Use this whenever the user asks what time or "
        "day it is, or when a calculation depends on now. Do not guess the time."
    ),
    args=GetTimeArgs,
)
def get_time(args: GetTimeArgs, ctx: ToolContext) -> str:
    zone = resolve_timezone(ctx.config)
    if args.timezone:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            zone = ZoneInfo(args.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolError(f"{args.timezone!r} is not a known timezone.") from exc

    now = datetime.now(zone)
    return now.strftime("%A %-d %B %Y, %-I:%M %p") + (f" ({now.tzname()})" if now.tzname() else "")


class SetTimerArgs(BaseModel):
    seconds: int = Field(..., gt=0, le=86_400, description="How long, in seconds.")
    label: str = Field("", description="What the timer is for, e.g. 'the pasta'.")


class TimerService:
    """Background timers that announce themselves when they finish.

    Timers live in memory only: a timer that survived a restart would fire at a
    moment the user has no context for, which is worse than losing it.
    """

    def __init__(self, announce: object = None) -> None:
        self._announce = announce
        self._timers: list[threading.Timer] = []
        self._lock = threading.Lock()

    def announce_with(self, speak: object) -> None:
        """Set how a finished timer makes itself heard.

        Set after construction because the thing that speaks is the loop, and
        the loop needs the timer service to exist before it is built.
        """
        self._announce = speak

    def schedule(self, seconds: float, label: str) -> datetime:
        due = datetime.now() + timedelta(seconds=seconds)

        def fire() -> None:
            message = f"Your {label} timer has finished." if label else "Your timer has finished."
            log.info("%s", message)
            if callable(self._announce):
                try:
                    self._announce(message)
                except Exception as exc:  # a failed announcement must not kill the thread
                    log.error("Could not announce a timer: %r", exc)

        timer = threading.Timer(seconds, fire)
        timer.daemon = True
        with self._lock:
            self._timers.append(timer)
        timer.start()
        return due

    def cancel_all(self) -> int:
        with self._lock:
            count = len(self._timers)
            for timer in self._timers:
                timer.cancel()
            self._timers.clear()
        return count

    @property
    def pending(self) -> int:
        with self._lock:
            return sum(1 for timer in self._timers if timer.is_alive())


@tool(
    name="set_timer",
    description=(
        "Set a countdown timer that announces itself out loud when it finishes. "
        "Use for 'remind me in ten minutes' or 'set a timer for the pasta'."
    ),
    args=SetTimerArgs,
)
def set_timer(args: SetTimerArgs, ctx: ToolContext) -> str:
    if ctx.timers is None:
        raise ToolError("Timers are not available in this session.")
    due = ctx.timers.schedule(args.seconds, args.label)
    return (
        f"Timer set for {_spoken_duration(args.seconds)}"
        f"{f' for {args.label}' if args.label else ''}. "
        f"It will finish at {due.strftime('%-I:%M %p')}."
    )


def _spoken_duration(seconds: int) -> str:
    """Render a duration the way it would be said aloud."""
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        if remainder:
            return f"{minutes} minute{'s' if minutes != 1 else ''} and {remainder} seconds"
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, minutes = divmod(minutes, 60)
    if minutes:
        return f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minutes"
    return f"{hours} hour{'s' if hours != 1 else ''}"
