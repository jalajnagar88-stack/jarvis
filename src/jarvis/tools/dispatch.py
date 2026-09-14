"""Executing tool calls.

The dispatcher is where confirmation is enforced. A tool declares that it needs
approval; it does not get to decide whether that approval happened. Keeping the
check here means a new tool cannot accidentally bypass it, and there is one
place to read to know what the rule is.

Nothing raises out of :meth:`ToolDispatcher.execute`. A failure becomes a tool
result marked as an error, which the model sees and can explain or work around.
A tool that blew up should not end the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.errors import SafetyViolation, ToolError
from jarvis.logging_setup import get_logger
from jarvis.tools.registry import REGISTRY, Tool, ToolContext, ToolRegistry

log = get_logger("tools.dispatch")


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of one tool call, ready to send back to the model."""

    content: str
    is_error: bool = False
    summary: str = ""
    """One line for the status window. Not the full output."""


class ToolDispatcher:
    """Validates, gates, and runs tool calls."""

    def __init__(
        self,
        cfg: Config,
        audit: AuditLog,
        *,
        registry: ToolRegistry | None = None,
        memory: Any = None,
        timers: Any = None,
    ) -> None:
        self._cfg = cfg
        self._audit = audit
        self._registry = registry if registry is not None else REGISTRY
        self._context = ToolContext(cfg, audit, memory=memory, timers=timers)

    @property
    def context(self) -> ToolContext:
        return self._context

    def enabled_tools(self) -> list[Tool[Any]]:
        """Local tools named in config, in config order.

        Server-side tools are filtered out first: they run on Anthropic's
        infrastructure and have no handler here, so asking the registry about
        them would warn about a tool that is working perfectly well.
        """
        from jarvis.tools.server_tools import SERVER_TOOL_NAMES

        local = [name for name in self._cfg.tools.enabled if name not in SERVER_TOOL_NAMES]
        return self._registry.enabled(local)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions for the API: local tools, plus web search if enabled."""
        definitions = [tool.schema() for tool in self.enabled_tools()]
        if "web_search" in self._cfg.tools.enabled:
            from jarvis.tools.server_tools import web_search_definition

            definitions.append(web_search_definition(self._cfg))
        return definitions

    def lookup(self, name: str) -> Tool[Any] | None:
        return self._registry.get(name)

    def needs_confirmation(self, name: str) -> bool:
        tool = self._registry.get(name)
        return tool is not None and tool.requires_confirmation

    def describe(self, name: str, raw_args: dict[str, Any]) -> str:
        """The confirmation question for a pending call.

        Falls back to naming the tool if the arguments do not validate -- the
        user still gets asked, and the failure surfaces when it runs.
        """
        tool = self._registry.get(name)
        if tool is None:
            return f"Run the unknown tool {name}?"
        try:
            return tool.describe_call(tool.parse(raw_args))
        except ToolError:
            return f"Run {name}?"

    def execute(
        self, name: str, raw_args: dict[str, Any], *, approved: bool | None = None
    ) -> ToolOutcome:
        """Run one tool call.

        Args:
            name: The tool the model asked for.
            raw_args: Its arguments, unvalidated.
            approved: The user's decision, for tools that require one. ``None``
                means no decision was collected, which counts as a refusal --
                the safe default when a confirmation is somehow skipped.
        """
        tool = self._registry.get(name)
        if tool is None:
            self._audit.record("tool.unknown", outcome="error", detail=name)
            return ToolOutcome(
                f"There is no tool called {name}.", is_error=True, summary=f"unknown tool {name}"
            )

        if name not in self._cfg.tools.enabled:
            self._audit.record("tool.disabled", outcome="denied", detail=name)
            return ToolOutcome(
                f"The {name} tool is turned off in this user's configuration.",
                is_error=True,
                summary=f"{name} is disabled",
            )

        if tool.requires_confirmation and approved is not True:
            self._audit.record(
                "tool.declined",
                outcome="declined",
                detail=name,
                arguments=_safe_args(raw_args),
            )
            log.info("The user declined %s.", name)
            return ToolOutcome(
                f"The user declined to run {name}. Do not try it again unless they ask.",
                is_error=False,
                summary=f"{name} declined",
            )

        try:
            args = tool.parse(raw_args)
        except ToolError as exc:
            return ToolOutcome(str(exc.message), is_error=True, summary="invalid arguments")

        try:
            result = tool.handler(args, self._context)
        except SafetyViolation as exc:
            self._audit.record(
                "tool.blocked",
                outcome="denied",
                detail=name,
                reason=exc.message,
                arguments=_safe_args(raw_args),
            )
            log.warning("Blocked %s: %s", name, exc.message)
            return ToolOutcome(exc.message, is_error=True, summary="blocked by the safety layer")
        except ToolError as exc:
            self._audit.record("tool.failed", outcome="error", detail=name, reason=exc.message)
            return ToolOutcome(exc.message, is_error=True, summary=exc.message[:80])
        except Exception as exc:  # a broken tool must not end the conversation
            log.exception("The %s tool raised an unexpected error.", name)
            self._audit.record("tool.crashed", outcome="error", detail=name, error=repr(exc))
            return ToolOutcome(
                f"The {name} tool failed unexpectedly: {exc}",
                is_error=True,
                summary=f"{name} crashed",
            )

        if tool.requires_confirmation:
            self._audit.record(
                "tool.confirmed", outcome="confirmed", detail=name, arguments=_safe_args(raw_args)
            )
        return ToolOutcome(result, summary=_first_line(result))


def _first_line(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:80]


def _safe_args(raw: dict[str, Any]) -> dict[str, Any]:
    """Trim arguments for the audit log.

    File contents in particular can be enormous, and an audit log nobody can
    read is not an audit log.
    """
    trimmed: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and len(value) > 200:
            trimmed[key] = value[:200] + f"… ({len(value)} characters)"
        else:
            trimmed[key] = value
    return trimmed
