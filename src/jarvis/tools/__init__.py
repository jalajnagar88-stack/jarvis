"""Tools, one file per capability, registered by decorator.

Importing this package registers every built-in tool. Which of them are actually
offered to the model is decided by ``tools.enabled`` in config.yaml, so removing
a capability is a one-line change and the tool never appears in the schema at
all.

Adding a tool: write a module under ``builtin/``, decorate the function with
``@tool``, and import it below.
"""

from __future__ import annotations

from jarvis.tools.builtin import (  # noqa: F401 - imported for the side effect of registering
    clock,
    files,
    memory_tools,
    shell,
    system,
    weather,
)
from jarvis.tools.registry import REGISTRY, Tool, ToolContext, ToolRegistry, tool

__all__ = ["REGISTRY", "Tool", "ToolContext", "ToolRegistry", "tool"]
