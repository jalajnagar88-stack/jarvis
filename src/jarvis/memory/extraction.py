"""Pulling durable facts out of a finished exchange.

Runs after the reply, on a background thread, with a cheap model: this happens
after every single turn, so it must not add latency the user can feel and must
not cost much.

The prompt is deliberately conservative. A memory that fills with "the user
asked about the weather" is worse than one that stays empty -- it crowds out the
things that actually matter and makes recall noisier every turn.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from jarvis.config import Config
from jarvis.interfaces.memory import MemoryStore
from jarvis.logging_setup import get_logger

log = get_logger("memory.extraction")

SYSTEM_PROMPT = """\
You extract durable facts about a user from a conversation, for an assistant's
long-term memory.

Store only what will still be true and useful in a month:
- stable preferences ("prefers metric units", "dislikes being called sir")
- names and relationships ("the user's sister is called Priya")
- standing instructions ("always confirm before deleting anything")
- enduring circumstances ("lives in Edinburgh", "works as a vet")

Do not store:
- anything about this conversation ("asked about the weather")
- anything transient ("is going out later", "is tired today")
- anything the assistant said, or anything you inferred rather than were told
- facts already implied by what you are given as already known

Write each fact in the third person, self-contained, under fifteen words.
Reply with a JSON array of strings, and nothing else. Return [] if there is
nothing worth keeping, which is the common case."""


class FactExtractor:
    """Extracts and stores facts in the background."""

    def __init__(
        self,
        cfg: Config,
        memory: MemoryStore,
        *,
        client_factory: Any = None,
    ) -> None:
        self._cfg = cfg
        self._memory = memory
        self._client_factory = client_factory
        self._client: Any = None
        self._threads: list[threading.Thread] = []

    def consider(self, user_text: str, assistant_text: str) -> None:
        """Queue an exchange for extraction. Returns immediately."""
        if not self._cfg.memory.extraction.enabled:
            return
        if not user_text.strip() or not assistant_text.strip():
            return

        thread = threading.Thread(
            target=self._extract_quietly,
            args=(user_text, assistant_text),
            daemon=True,
            name="fact-extractor",
        )
        self._threads.append(thread)
        thread.start()

    def wait(self, timeout: float = 5.0) -> None:
        """Block until queued extractions finish. For tests and shutdown."""
        for thread in list(self._threads):
            thread.join(timeout)
        self._threads = [t for t in self._threads if t.is_alive()]

    def _extract_quietly(self, user_text: str, assistant_text: str) -> None:
        """Never let a background failure reach the user.

        Losing a fact is a minor degradation; a traceback printed over a
        conversation is not.
        """
        try:
            for fact in self.extract(user_text, assistant_text):
                self._memory.add_fact(fact, source="extracted")
        except Exception as exc:
            log.debug("Fact extraction failed quietly: %r", exc)

    def extract(self, user_text: str, assistant_text: str) -> list[str]:
        """Ask the cheap model what is worth remembering."""
        client = self._ensure_client()
        if client is None:
            return []

        known = [fact.text for fact in self._memory.all_facts()[:40]]
        transcript = (
            f"Already known:\n{json.dumps(known)}\n\n"
            f"User said: {user_text}\n"
            f"Assistant replied: {assistant_text}"
        )

        response = client.messages.create(
            model=self._cfg.memory.extraction.model,
            max_tokens=self._cfg.memory.extraction.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": transcript}],
        )
        return _parse(_text_of(response))

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client

        if not self._cfg.secrets.anthropic_api_key:
            return None
        try:
            import anthropic
        except ImportError:
            return None

        kwargs: dict[str, Any] = {
            "api_key": self._cfg.secrets.anthropic_api_key,
            "timeout": self._cfg.llm.timeout_seconds,
            "max_retries": 1,
        }
        if self._cfg.secrets.anthropic_base_url:
            kwargs["base_url"] = self._cfg.secrets.anthropic_base_url
        self._client = anthropic.Anthropic(**kwargs)
        return self._client


def _text_of(message: Any) -> str:
    parts = [
        getattr(block, "text", "")
        for block in (getattr(message, "content", None) or [])
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts).strip()


def _parse(reply: str) -> list[str]:
    """Read the JSON array, tolerating a model that wrapped it in prose."""
    if not reply:
        return []
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        log.debug("Fact extractor returned unparseable JSON: %r", reply[:200])
        return []

    if not isinstance(parsed, list):
        return []
    return [
        " ".join(item.split())
        for item in parsed
        if isinstance(item, str) and 0 < len(item.strip()) <= 200
    ]
