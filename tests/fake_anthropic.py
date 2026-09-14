"""A stand-in for the Anthropic SDK's streaming surface.

Mirrors the shape the brain actually uses: ``client.messages.stream(...)`` as a
context manager exposing ``text_stream`` and ``get_final_message()``, plus
``client.messages.create(...)`` for the non-streaming path. Fragments are
delivered one at a time so the sentence splitting is genuinely exercised on a
stream rather than on a finished string.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any


@dataclass
class FakeUsage:
    input_tokens: int = 11
    output_tokens: int = 23
    cache_read_input_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeStopDetails:
    type: str = "refusal"
    category: str | None = "cyber"
    explanation: str | None = None


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeMessage:
    content: list[Any] = field(default_factory=list)
    stop_reason: str | None = "end_turn"
    stop_details: FakeStopDetails | None = None
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeStream:
    def __init__(
        self,
        fragments: list[str],
        final: FakeMessage,
        raise_at: tuple[int, Exception] | None = None,
    ) -> None:
        self._fragments = fragments
        self._final = final
        self._raise_at = raise_at
        self.closed = False

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.closed = True

    @property
    def text_stream(self) -> Iterator[str]:
        for index, fragment in enumerate(self._fragments):
            if self._raise_at is not None and self._raise_at[0] == index:
                raise self._raise_at[1]
            yield fragment

    def get_final_message(self) -> FakeMessage:
        return self._final


class FakeMessages:
    def __init__(self, owner: FakeAnthropic) -> None:
        self._owner = owner

    def stream(self, **kwargs: Any) -> FakeStream:
        self._owner.requests.append(kwargs)
        if self._owner.raise_on_call is not None:
            raise self._owner.raise_on_call
        fragments = self._owner.next_fragments()
        stream = FakeStream(
            fragments, self._owner.next_final(fragments), self._owner.raise_mid_stream
        )
        self._owner.streams.append(stream)
        return stream

    def create(self, **kwargs: Any) -> FakeMessage:
        self._owner.requests.append(kwargs)
        if self._owner.raise_on_call is not None:
            raise self._owner.raise_on_call
        fragments = self._owner.next_fragments()
        return self._owner.next_final(fragments)


class FakeAnthropic:
    """Records every request and replays scripted replies."""

    def __init__(
        self,
        replies: list[list[str]] | None = None,
        *,
        raise_on_call: Exception | None = None,
        raise_mid_stream: tuple[int, Exception] | None = None,
        stop_reason: str = "end_turn",
        stop_reasons: list[str] | None = None,
        stop_details: FakeStopDetails | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.client_kwargs = client_kwargs
        self.requests: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.raise_on_call = raise_on_call
        self.raise_mid_stream = raise_mid_stream
        self._replies = list(replies or [["Very good, sir."]])
        self._stop_reason = stop_reason
        self._stop_reasons = list(stop_reasons or [])
        self._stop_details = stop_details
        self._tool_calls: list[FakeToolUseBlock] = []
        self._tool_forever = False
        self.messages = FakeMessages(self)

    def script_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        call_id: str = "call_1",
        forever: bool = False,
    ) -> None:
        """Make the next response ask for this tool.

        With ``forever``, every response asks for it again -- which is how the
        round limit is exercised.
        """
        self._tool_calls.append(FakeToolUseBlock(id=call_id, name=name, input=arguments))
        self._tool_forever = forever

    def next_fragments(self) -> list[str]:
        if self._replies:
            return self._replies.pop(0)
        return [""]

    def next_final(self, fragments: list[str]) -> FakeMessage:
        text = "".join(fragments)
        content: list[Any] = [FakeTextBlock(text=text)] if text else []

        pending = None
        if self._tool_forever and self._tool_calls:
            pending = self._tool_calls[0]
        elif self._tool_calls:
            pending = self._tool_calls.pop(0)

        if pending is not None:
            content.append(pending)
            return FakeMessage(content=content, stop_reason="tool_use")

        if self._stop_reasons:
            return FakeMessage(
                content=content,
                stop_reason=self._stop_reasons.pop(0),
                stop_details=self._stop_details,
            )
        return FakeMessage(
            content=content,
            stop_reason=self._stop_reason,
            stop_details=self._stop_details,
        )


def install(monkeypatch: Any, fake: FakeAnthropic) -> None:
    """Make ``import anthropic`` inside the brain yield this fake."""
    import sys

    module = type(sys)("anthropic")
    module.Anthropic = lambda **kwargs: _configure(fake, kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


def _configure(fake: FakeAnthropic, kwargs: dict[str, Any]) -> FakeAnthropic:
    fake.client_kwargs = kwargs
    return fake
