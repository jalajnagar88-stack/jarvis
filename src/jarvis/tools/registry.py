"""Tool registration.

Adding a tool is one file: define a pydantic model for its arguments, write a
function, decorate it. The decorator generates the JSON schema the API needs
from the model, so the schema can never drift from the code that reads it.

    @tool(name="get_time", description="...")
    def get_time(args: GetTimeArgs, ctx: ToolContext) -> str:
        ...

Tools return a string, because that is what goes back to the model as a tool
result. Raising :class:`ToolError` is the way to report failure: the message is
handed back to the model so it can explain itself or try something else, rather
than the turn collapsing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from jarvis.errors import ToolError
from jarvis.logging_setup import get_logger

log = get_logger("tools")

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class ToolContext:
    """Everything a tool is allowed to reach.

    Passed explicitly rather than imported, so a tool cannot quietly acquire a
    dependency on the rest of the system, and so tests can hand it fakes.
    """

    def __init__(
        self,
        config: Any,
        audit: Any,
        *,
        memory: Any = None,
        timers: Any = None,
    ) -> None:
        self.config = config
        self.audit = audit
        self.memory = memory
        self.timers = timers


@dataclass(slots=True)
class Tool(Generic[ArgsT]):
    """One registered tool."""

    name: str
    description: str
    args_model: type[ArgsT]
    handler: Callable[[ArgsT, ToolContext], str]
    requires_confirmation: bool = False
    """If true, the user must approve each call before it runs. Enforced by the
    dispatcher, not by the tool, so a tool cannot opt itself out."""
    confirmation_prompt: Callable[[ArgsT], str] | None = None
    """Builds the spoken question. Must state exactly what will happen."""
    destructive: bool = False
    """Marks tools that change the machine, for the audit log and the UI."""

    def schema(self) -> dict[str, Any]:
        """The tool definition sent to the API."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        # `strict` requires this, and it stops the model inventing extra keys.
        schema["additionalProperties"] = False
        schema.setdefault("properties", {})
        schema.setdefault("required", [])
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
            "strict": True,
        }

    def parse(self, raw: dict[str, Any]) -> ArgsT:
        """Validate the model's arguments.

        Raises:
            ToolError: the arguments do not fit the schema. The message goes
                back to the model, which can then correct itself.
        """
        try:
            return self.args_model.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'argument'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolError(f"Invalid arguments for {self.name}: {problems}") from exc

    def describe_call(self, args: ArgsT) -> str:
        """The confirmation question for this call."""
        if self.confirmation_prompt is not None:
            return self.confirmation_prompt(args)
        return f"Run {self.name}?"


@dataclass
class ToolRegistry:
    """Every known tool, keyed by name."""

    tools: dict[str, Tool[Any]] = field(default_factory=dict)

    def register(self, tool_obj: Tool[Any]) -> None:
        if tool_obj.name in self.tools:
            raise ValueError(f"Two tools are both named {tool_obj.name!r}.")
        self.tools[tool_obj.name] = tool_obj

    def get(self, name: str) -> Tool[Any] | None:
        return self.tools.get(name)

    def enabled(self, names: list[str]) -> list[Tool[Any]]:
        """The subset named in config, in config order.

        A name with no matching tool is logged and skipped rather than raising:
        the health check already warns about it, and losing one tool is not a
        reason to refuse to start.
        """
        selected: list[Tool[Any]] = []
        for name in names:
            found = self.tools.get(name)
            if found is None:
                log.warning("tools.enabled lists %r, which is not a known tool.", name)
                continue
            selected.append(found)
        return selected

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: object) -> bool:
        return name in self.tools


REGISTRY = ToolRegistry()
"""The process-wide registry. Populated by importing ``jarvis.tools``."""


def tool(
    *,
    name: str,
    description: str,
    args: type[ArgsT],
    requires_confirmation: bool = False,
    confirmation_prompt: Callable[[ArgsT], str] | None = None,
    destructive: bool = False,
    registry: ToolRegistry | None = None,
) -> Callable[[Callable[[ArgsT, ToolContext], str]], Callable[[ArgsT, ToolContext], str]]:
    """Register a function as a tool.

    Args:
        name: What the model calls it. Must be unique.
        description: What it does, written for the model. Say when *not* to use
            it as well as when to.
        args: A pydantic model describing the arguments.
        requires_confirmation: Ask the user before every call.
        confirmation_prompt: Builds the question. State exactly what will
            happen -- "Shall I run: rm build" beats "Run a command?".
        destructive: Records that this changes the machine.
        registry: Override the global registry. For tests.
    """

    def decorate(
        handler: Callable[[ArgsT, ToolContext], str],
    ) -> Callable[[ArgsT, ToolContext], str]:
        target = registry if registry is not None else REGISTRY
        target.register(
            Tool(
                name=name,
                description=description,
                args_model=args,
                handler=handler,
                requires_confirmation=requires_confirmation,
                confirmation_prompt=confirmation_prompt,
                destructive=destructive,
            )
        )
        return handler

    return decorate
