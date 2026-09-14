"""Anthropic's server-side tools.

Web search runs on Anthropic's infrastructure: the request is declared in the
``tools`` list and the results come back inside the same response, with no
client-side execution. That is why it is here rather than in ``builtin/`` -- it
has no handler, and the dispatcher never sees it.

Using a scraper instead would mean fetching pages from the user's own address,
parsing hostile HTML, and maintaining it forever. This is one dictionary.
"""

from __future__ import annotations

from typing import Any

from jarvis.config import Config

WEB_SEARCH_TYPE = "web_search_20260209"
"""The dynamic-filtering variant. Supported by the Sonnet 5 and Opus 4.6+
families; older models need the basic `web_search_20250305`."""

_BASIC_WEB_SEARCH_TYPE = "web_search_20250305"

# Models predating the dynamic-filtering tool.
_OLDER_MODELS = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5", "claude-3")


def web_search_definition(cfg: Config) -> dict[str, Any]:
    """Build the web search tool declaration from config."""
    search = cfg.tools.web_search
    definition: dict[str, Any] = {
        "type": _search_type_for(cfg.llm.model),
        "name": "web_search",
        "max_uses": search.max_uses,
    }
    # The API rejects a request carrying both; config validation catches it too.
    if search.allowed_domains:
        definition["allowed_domains"] = list(search.allowed_domains)
    elif search.blocked_domains:
        definition["blocked_domains"] = list(search.blocked_domains)
    return definition


def _search_type_for(model: str) -> str:
    return _BASIC_WEB_SEARCH_TYPE if model.startswith(_OLDER_MODELS) else WEB_SEARCH_TYPE


SERVER_TOOL_NAMES = frozenset({"web_search"})
"""Names the dispatcher must not try to execute locally."""
