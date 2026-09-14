"""Building the system prompt.

A voice assistant needs different instructions from a chat window, and most of
the difference is about what *cannot* be spoken. Markdown, bullet lists, code
blocks, tables and URLs are all either unspeakable or excruciating to listen to.
Length matters more too: a paragraph that reads fine takes forty seconds to say.

The prompt is assembled stable-part-first. Nothing depends on that yet, but
prompt caching matches on a prefix, so putting the timestamp last is what will
let the persona be cached once tools and memory make these requests expensive.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jarvis.config import Config
from jarvis.logging_setup import get_logger

log = get_logger("agent.prompt")

SPEECH_RULES = """\
Everything you say is converted to speech and played aloud. Nothing you write is
ever read on a screen, so:

- Be brief. One or two sentences unless genuinely more is needed. A list of five
  things takes half a minute to listen to; offer the top two and stop.
- Write plain prose. No markdown, no bullet points, no numbered lists, no
  headings, no code blocks, no tables, no emoji.
- Expand anything that does not read aloud well. Say "twenty past three", not
  "15:20". Say "twenty-one degrees", not "21C". Avoid URLs entirely; describe
  the source instead.
- Never narrate what you are about to do. Do it, then report what happened.
- If you do not know something, say so in one short sentence.
- If a request is ambiguous, ask one short question rather than guessing at
  length."""


def build_system_prompt(
    cfg: Config,
    *,
    now: datetime | None = None,
    facts: list[str] | None = None,
) -> str:
    """Assemble the system prompt.

    Args:
        cfg: Loaded configuration. Supplies the persona, the assistant's name,
            and the preferred units.
        now: Current time, for testing. Defaults to the real clock in the
            configured timezone.
        facts: Remembered facts to inject. Empty until milestone 5.
    """
    sections: list[str] = [
        cfg.general.persona.strip(),
        SPEECH_RULES,
        f"Your name is {cfg.general.name}.",
        _units_line(cfg),
    ]

    if facts:
        remembered = "\n".join(f"- {fact}" for fact in facts)
        sections.append("Things you have been told before and should treat as true:\n" + remembered)

    # Volatile last: everything above is stable across turns.
    sections.append(_time_line(cfg, now))

    return "\n\n".join(section for section in sections if section)


def _units_line(cfg: Config) -> str:
    if cfg.general.units == "metric":
        return "Use metric units and degrees Celsius unless asked otherwise."
    return "Use imperial units and degrees Fahrenheit unless asked otherwise."


def _time_line(cfg: Config, now: datetime | None) -> str:
    moment = now if now is not None else datetime.now(resolve_timezone(cfg))
    return (
        f"The current date and time is {moment.strftime('%A %-d %B %Y, %-I:%M %p')}"
        f"{_timezone_suffix(moment)}."
    )


def _timezone_suffix(moment: datetime) -> str:
    name = moment.tzname()
    return f" ({name})" if name else ""


def resolve_timezone(cfg: Config) -> ZoneInfo | None:
    """The configured timezone, or None to use the system's.

    An unknown timezone name is a warning rather than an error: being an hour
    out is a far better failure than refusing to start.
    """
    if cfg.general.timezone is None:
        return None
    try:
        return ZoneInfo(cfg.general.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning(
            "general.timezone %r is not a known timezone; using the system clock instead.",
            cfg.general.timezone,
        )
        return None
