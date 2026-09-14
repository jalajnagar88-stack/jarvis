"""The brain: the only module in JARVIS that talks to Anthropic.

Streaming is not a performance nicety here, it is the whole design. The reply
is split into sentences as it arrives and each one is handed to speech the
moment it is complete, so JARVIS starts talking roughly when the first sentence
lands rather than when the last one does. On a three-sentence answer that is the
difference between a half-second pause and a three-second one, which is most of
what makes an assistant feel present rather than sluggish.

The turn is a generator. That matters for milestone 6: when the user speaks over
JARVIS, the loop simply stops consuming it, and closing the generator tears down
the HTTP stream. There is no cancellation flag to get wrong.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator, Iterator
from typing import Any

from jarvis.agent.prompt import build_system_prompt
from jarvis.config import Config
from jarvis.interfaces.agent import (
    AgentEvent,
    AgentFailed,
    Brain,
    ConfirmationRequired,
    SentenceComplete,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
)
from jarvis.logging_setup import get_logger
from jarvis.tts.sentences import SentenceSplitter

log = get_logger("agent")


class AnthropicBrain(Brain):
    """Turns user text into a stream of events, backed by the Anthropic API."""

    def __init__(
        self,
        cfg: Config,
        *,
        dispatcher: Any = None,
        memory: Any = None,
        extractor: Any = None,
    ) -> None:
        self._cfg = cfg
        self._client: Any = None
        self._messages: list[dict[str, Any]] = []
        self._dispatcher = dispatcher
        self._decisions: dict[str, bool] = {}
        self._memory = memory
        self._extractor = extractor
        self._session_id = uuid.uuid4().hex[:12]

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def reset(self) -> None:
        """Drop the working conversation. Stored memory is unaffected."""
        self._messages.clear()
        self._decisions.clear()
        log.debug("Conversation history cleared.")

    def confirm(self, call_id: str, approved: bool) -> None:
        """Answer a :class:`ConfirmationRequired` raised during :meth:`respond`.

        Call this before resuming the event iterator. A decision that never
        arrives counts as a refusal, which is the safe default.
        """
        self._decisions[call_id] = approved

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - anthropic is a core dependency
            from jarvis.errors import DependencyMissingError

            raise DependencyMissingError(
                "The anthropic package is not installed.", remedy="uv sync"
            ) from exc

        if not self._cfg.secrets.anthropic_api_key:
            from jarvis.errors import AuthError

            raise AuthError(
                "ANTHROPIC_API_KEY is not set, so JARVIS can listen and speak but cannot think.",
                remedy="cp .env.example .env  # then paste your key from console.anthropic.com",
            )

        kwargs: dict[str, Any] = {
            "api_key": self._cfg.secrets.anthropic_api_key,
            "timeout": self._cfg.llm.timeout_seconds,
            "max_retries": self._cfg.llm.max_retries,
        }
        if self._cfg.secrets.anthropic_base_url:
            kwargs["base_url"] = self._cfg.secrets.anthropic_base_url

        self._client = anthropic.Anthropic(**kwargs)
        log.debug("Anthropic client ready (model %s)", self._cfg.llm.model)
        return self._client

    # -- one turn ----------------------------------------------------------- #

    def respond(self, user_text: str) -> Iterator[AgentEvent]:
        """Handle one user turn, yielding events as they happen."""
        text = user_text.strip()
        if not text:
            yield TurnFinished(text="", stop_reason="empty")
            return

        try:
            client = self._ensure_client()
        except Exception as exc:
            yield self._failure(exc)
            return

        yield from self._run_turn(client, text)

    def _build_request(self, messages: list[dict[str, Any]], user_text: str) -> dict[str, Any]:
        """Assemble the request body for one leg of a turn."""
        facts = self._recall(user_text)

        request: dict[str, Any] = {
            "model": self._cfg.llm.model,
            "max_tokens": self._cfg.llm.max_tokens,
            "system": build_system_prompt(self._cfg, facts=facts),
            "messages": messages,
            "output_config": {"effort": self._cfg.llm.effort},
        }
        # Sonnet 5 accepts adaptive or disabled; budget_tokens was removed.
        request["thinking"] = {"type": self._cfg.llm.thinking}

        if self._dispatcher is not None:
            schemas = self._dispatcher.schemas()
            if schemas:
                request["tools"] = schemas
        return request

    def _trimmed_history(self) -> list[dict[str, Any]]:
        """The recent conversation, bounded by ``llm.history_turns``.

        A turn is one exchange, so the message limit is twice that. Trimming
        starts at a user message whose content is not a tool result: beginning
        the history with an assistant reply, or with results for a tool call the
        model can no longer see, leaves it dangling.
        """
        limit = self._cfg.llm.history_turns * 2
        if limit <= 0 or len(self._messages) <= limit:
            return list(self._messages)

        trimmed = self._messages[-limit:]
        while trimmed and not _is_plain_user_message(trimmed[0]):
            trimmed.pop(0)
        return trimmed

    def _run_turn(self, client: Any, user_text: str) -> Iterator[AgentEvent]:
        """One user turn, including any tool calls it needs.

        Loops until the model stops asking for tools or the configured ceiling
        is reached. The ceiling exists so a confused model cannot run up a bill
        while the user stands there listening to silence.
        """
        messages: list[dict[str, Any]] = [
            *self._trimmed_history(),
            {"role": "user", "content": user_text},
        ]
        splitter = SentenceSplitter()
        collected: list[str] = []
        max_rounds = self._cfg.tools.max_iterations if self._dispatcher is not None else 1

        for round_index in range(max_rounds):
            request = self._build_request(messages, user_text)

            try:
                if self._cfg.llm.streaming:
                    final = yield from self._stream_leg(client, request, splitter, collected)
                else:
                    final = yield from self._single_leg(client, request, splitter, collected)
            except _LegFailed as failure:
                yield failure.event
                return

            stop_reason = getattr(final, "stop_reason", None)

            refusal = self._refusal_event(final, stop_reason)
            if refusal is not None:
                yield refusal
                return

            if stop_reason == "pause_turn":
                # A server-side tool ran long and the turn was paused. Resending
                # the paused assistant turn continues it; dropping it here would
                # silently truncate the answer with no error.
                messages.append({"role": "assistant", "content": final.content})
                continue

            tool_calls = [
                block
                for block in (getattr(final, "content", None) or [])
                if getattr(block, "type", None) == "tool_use"
            ]
            if stop_reason != "tool_use" or not tool_calls or self._dispatcher is None:
                break

            messages.append({"role": "assistant", "content": final.content})
            results = yield from self._run_tools(tool_calls)
            messages.append({"role": "user", "content": results})

            if round_index == max_rounds - 1:
                log.warning("Hit the %d-round tool limit; answering with what we have.", max_rounds)
                yield AgentFailed(
                    spoken="I got stuck going round in circles on that one.",
                    detail=f"reached tools.max_iterations ({max_rounds})",
                )
                return
        else:  # pragma: no cover - the loop always breaks or returns
            return

        remainder = splitter.flush()
        if remainder:
            yield SentenceComplete(text=remainder)

        reply = "".join(collected).strip()
        self._remember_exchange(user_text, reply)
        yield TurnFinished(
            text=reply, stop_reason=getattr(final, "stop_reason", None), usage=_usage_of(final)
        )

    def _run_tools(
        self, tool_calls: list[Any]
    ) -> Generator[AgentEvent, None, list[dict[str, Any]]]:
        """Execute a batch of tool calls, asking the user where required.

        Every result is returned, including failures: dropping one leaves the
        model with a tool_use block that has no answer, which the API rejects.
        """
        assert self._dispatcher is not None
        results: list[dict[str, Any]] = []

        for call in tool_calls:
            name = getattr(call, "name", "")
            call_id = getattr(call, "id", "")
            raw_args = dict(getattr(call, "input", None) or {})

            yield ToolCallStarted(name=name, arguments=raw_args, call_id=call_id)

            approved: bool | None = None
            if self._dispatcher.needs_confirmation(name):
                yield ConfirmationRequired(
                    name=name,
                    call_id=call_id,
                    prompt=self._dispatcher.describe(name, raw_args),
                    details=_render_arguments(raw_args),
                )
                # The caller answers by calling confirm() before resuming us.
                approved = self._decisions.pop(call_id, None)

            outcome = self._dispatcher.execute(name, raw_args, approved=approved)
            yield ToolCallFinished(
                name=name, call_id=call_id, ok=not outcome.is_error, summary=outcome.summary
            )

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": outcome.content,
                    **({"is_error": True} if outcome.is_error else {}),
                }
            )

        return results

    def _stream_leg(
        self,
        client: Any,
        request: dict[str, Any],
        splitter: SentenceSplitter,
        collected: list[str],
    ) -> Generator[AgentEvent, None, Any]:
        """Stream one request, emitting sentences as they complete."""
        try:
            with client.messages.stream(**request) as stream:
                for fragment in stream.text_stream:
                    if not fragment:
                        continue
                    collected.append(fragment)
                    yield TextDelta(text=fragment)
                    for sentence in splitter.feed(fragment):
                        yield SentenceComplete(text=sentence)
                return stream.get_final_message()
        except GeneratorExit:
            # The caller stopped consuming: an interruption, not an error. The
            # `with` block has already closed the HTTP stream.
            log.debug("Turn abandoned by the caller.")
            raise
        except Exception as exc:
            raise _LegFailed(self._failure(exc)) from exc

    def _single_leg(
        self,
        client: Any,
        request: dict[str, Any],
        splitter: SentenceSplitter,
        collected: list[str],
    ) -> Generator[AgentEvent, None, Any]:
        """Non-streaming path, for ``llm.streaming: false``.

        Kept because the setting exists, and a setting that silently does
        nothing is worse than no setting. Speech cannot begin until the whole
        reply has arrived, which is exactly the latency streaming avoids.
        """
        try:
            final = client.messages.create(**request)
        except Exception as exc:
            raise _LegFailed(self._failure(exc)) from exc

        text = _text_of(final)
        if text:
            collected.append(text)
            yield TextDelta(text=text)
            for sentence in splitter.feed(text):
                yield SentenceComplete(text=sentence)
        return final

    def _refusal_event(self, final: Any, stop_reason: str | None) -> AgentFailed | None:
        """Turn a safety refusal into something speakable.

        The API returns HTTP 200 with ``stop_reason: "refusal"`` and usually no
        usable content, so reading the text and speaking it would produce
        silence with no explanation.
        """
        if stop_reason != "refusal":
            return None

        details = getattr(final, "stop_details", None)
        category = getattr(details, "category", None)
        log.warning("The model declined the request (category %s).", category)
        return AgentFailed(
            spoken="I'm afraid I can't help with that one.",
            detail=f"stop_reason=refusal category={category}",
        )

    def _recall(self, user_text: str) -> list[str] | None:
        """The stored facts most relevant to this turn.

        Never lets a memory failure break a conversation: JARVIS with no recall
        is far better than JARVIS that stops answering.
        """
        if self._memory is None:
            return None
        try:
            facts = self._memory.recall(
                user_text,
                top_k=self._cfg.memory.top_k,
                min_similarity=self._cfg.memory.min_similarity,
            )
        except Exception as exc:
            log.error("Could not recall facts (%r); continuing without them.", exc)
            return None
        if facts:
            log.debug("Recalled %d fact(s) for this turn.", len(facts))
        return [fact.text for fact in facts]

    def _remember_exchange(self, user_text: str, reply: str) -> None:
        """Append the exchange to the working conversation.

        Only complete exchanges are stored. A failed or abandoned turn leaves no
        trace, so the next request cannot inherit a dangling user message with
        no answer.
        """
        if not reply:
            return
        self._messages.append({"role": "user", "content": user_text})
        self._messages.append({"role": "assistant", "content": reply})

        if self._memory is not None:
            try:
                self._memory.add_message("user", user_text, session_id=self._session_id)
                self._memory.add_message("assistant", reply, session_id=self._session_id)
            except Exception as exc:
                log.error("Could not write to the transcript (%r).", exc)

        if self._extractor is not None:
            # Background: the user should not wait for this.
            self._extractor.consider(user_text, reply)

    def _failure(self, exc: Exception) -> AgentFailed:
        from jarvis.errors import JarvisError

        spoken, detail = describe_failure(exc)
        # A setup problem is about to be said aloud, printed, and written to the
        # audit log. Shouting it into the console log as well just interrupts
        # the conversation; the log file still gets it at debug level.
        if isinstance(exc, JarvisError):
            log.debug("%s (%s)", detail, type(exc).__name__)
        else:
            log.error("%s (%s)", detail, type(exc).__name__)
        return AgentFailed(spoken=spoken, detail=detail)


# --------------------------------------------------------------------------- #
# Failure messages
# --------------------------------------------------------------------------- #


def describe_failure(exc: Exception) -> tuple[str, str]:
    """Map an exception to (what JARVIS says aloud, what goes in the log).

    The spoken half is short, blames nothing, and says whether it is worth
    trying again -- standing in silence wondering is worse than a one-line
    apology. The logged half keeps the detail.
    """
    from jarvis.errors import AuthError, DependencyMissingError, JarvisError

    if isinstance(exc, AuthError):
        # Short enough to speak, specific enough to act on. The full remedy is
        # in the detail, which the terminal prints underneath.
        return "My API key isn't set, so I can't think just now.", str(exc)
    if isinstance(exc, DependencyMissingError):
        return "Part of my installation is missing.", str(exc)
    if isinstance(exc, JarvisError):
        return "I can't reach my reasoning just now.", str(exc)

    name = type(exc).__name__
    spoken = _SPOKEN_BY_EXCEPTION.get(name)
    if spoken is not None:
        return spoken, f"{name}: {exc}"

    status = getattr(exc, "status_code", None)
    if status is not None:
        if status == 529:
            return "The service is busy. Ask me again shortly.", f"{name}: {exc}"
        if status >= 500:
            return "Something went wrong at the other end. Try again shortly.", f"{name}: {exc}"
        return "That request was refused.", f"{name}: {exc}"

    return "Something went wrong while I was thinking.", f"{name}: {exc}"


# Keyed by class name so this module never imports anthropic just to catch it.
_SPOKEN_BY_EXCEPTION = {
    "AuthenticationError": "My API key was rejected. Please check it.",
    "PermissionDeniedError": "My API key isn't permitted to do that.",
    "NotFoundError": "That model isn't available to me.",
    "RateLimitError": "I'm being rate limited. Try again in a moment.",
    "APIConnectionError": "I can't reach the network just now.",
    "APITimeoutError": "That took too long, so I gave up.",
    "BadRequestError": "Something about that request was invalid.",
    "InternalServerError": "Something went wrong at the other end. Try again shortly.",
}


def _text_of(message: Any) -> str:
    """Concatenate the text blocks of a non-streaming response."""
    parts: list[str] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def _usage_of(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        field: value
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
        if isinstance(value := getattr(usage, field, None), int)
    }


class _LegFailed(Exception):
    """Internal: carries an AgentFailed out of a nested generator."""

    def __init__(self, event: AgentFailed) -> None:
        super().__init__(event.detail)
        self.event = event


def _is_plain_user_message(message: dict[str, Any]) -> bool:
    """True for a user message that is text, not a batch of tool results."""
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    return not any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content or []
    )


def _render_arguments(raw: dict[str, Any]) -> str:
    """Arguments as a readable block, for the confirmation prompt on screen."""
    lines = []
    for key, value in raw.items():
        text = str(value)
        if len(text) > 300:
            text = text[:300] + f"… ({len(text)} characters)"
        lines.append(f"{key}: {text}")
    return "\n".join(lines)
