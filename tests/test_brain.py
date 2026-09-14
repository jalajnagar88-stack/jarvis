"""The Anthropic brain.

The SDK is replaced with a fake that mirrors the streaming surface the brain
actually uses, so these tests cover request construction, sentence-by-sentence
emission, history management, and -- most importantly -- what JARVIS says out
loud when the API will not cooperate.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from jarvis.agent.anthropic_brain import AnthropicBrain, describe_failure
from jarvis.agent.echo import EchoBrain
from jarvis.agent.prompt import build_system_prompt
from jarvis.config import Config
from jarvis.errors import AuthError
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

from . import fake_anthropic
from .fake_anthropic import FakeAnthropic, FakeStopDetails


@pytest.fixture
def keyed(cfg: Config) -> Config:
    cfg.secrets.anthropic_api_key = "sk-ant-test"
    return cfg


def events_of(brain: Brain, text: str) -> list[AgentEvent]:
    return list(brain.respond(text))


def sentences_of(events: Sequence[AgentEvent]) -> list[str]:
    return [e.text for e in events if isinstance(e, SentenceComplete)]


class TestRequestConstruction:
    def test_the_configured_model_and_limits_are_sent(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed), "hello")

        request = fake.requests[0]
        assert request["model"] == "claude-sonnet-5"
        assert request["max_tokens"] == keyed.llm.max_tokens

    def test_effort_goes_inside_output_config(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not top-level: the API rejects that."""
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed), "hello")

        assert fake.requests[0]["output_config"] == {"effort": keyed.llm.effort}
        assert "effort" not in fake.requests[0]

    def test_thinking_uses_the_adaptive_form(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """budget_tokens was removed on this model generation and 400s."""
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed), "hello")

        assert fake.requests[0]["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in fake.requests[0]["thinking"]

    def test_thinking_can_be_disabled(self, keyed: Config, monkeypatch: pytest.MonkeyPatch) -> None:
        keyed.llm.thinking = "disabled"
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed), "hello")
        assert fake.requests[0]["thinking"] == {"type": "disabled"}

    def test_the_system_prompt_is_sent(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed), "hello")

        system = fake.requests[0]["system"]
        assert "JARVIS" in system
        assert "converted to speech" in system

    def test_the_client_gets_the_configured_timeout_and_retries(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed), "hello")

        assert fake.client_kwargs["timeout"] == keyed.llm.timeout_seconds
        assert fake.client_kwargs["max_retries"] == keyed.llm.max_retries
        assert fake.client_kwargs["api_key"] == "sk-ant-test"

    def test_a_base_url_override_is_passed_through(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keyed.secrets.anthropic_base_url = "https://proxy.internal"
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed), "hello")
        assert fake.client_kwargs["base_url"] == "https://proxy.internal"


class TestStreaming:
    def test_each_sentence_is_emitted_as_it_completes(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of milestone 3."""
        fake = FakeAnthropic([["Good evening", ", sir.", " The kettle", " is on."]])
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed), "hello")
        assert sentences_of(events) == ["Good evening, sir.", "The kettle is on."]

    def test_a_sentence_is_emitted_before_the_stream_ends(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the first sentence only arrived at the end, nothing would be gained."""
        fake = FakeAnthropic([["One. ", "Two. ", "Three."]])
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed), "hello")
        first_sentence = next(i for i, e in enumerate(events) if isinstance(e, SentenceComplete))
        turn_finished = next(i for i, e in enumerate(events) if isinstance(e, TurnFinished))
        assert first_sentence < turn_finished

    def test_text_deltas_are_forwarded_verbatim(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic([["Good ", "evening", "."]])
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed), "hello")
        deltas = [e.text for e in events if isinstance(e, TextDelta)]
        assert deltas == ["Good ", "evening", "."]

    def test_an_unterminated_reply_is_still_emitted(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise a reply with no full stop would never be spoken."""
        fake = FakeAnthropic([["no full stop here"]])
        fake_anthropic.install(monkeypatch, fake)
        assert sentences_of(events_of(AnthropicBrain(keyed), "hi")) == ["no full stop here"]

    def test_the_turn_reports_usage(self, keyed: Config, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeAnthropic([["Done."]])
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed), "hello")
        finished = next(e for e in events if isinstance(e, TurnFinished))
        assert finished.usage["output_tokens"] == 23
        assert finished.text == "Done."

    def test_empty_input_short_circuits(self, keyed: Config) -> None:
        events = events_of(AnthropicBrain(keyed), "   ")
        assert len(events) == 1
        assert isinstance(events[0], TurnFinished)

    def test_abandoning_the_turn_closes_the_stream(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This is milestone 6's interrupt path: stop consuming, and it tears down."""
        fake = FakeAnthropic([["One. ", "Two. ", "Three. ", "Four."]])
        fake_anthropic.install(monkeypatch, fake)

        # The interface promises only an Iterator; every real implementation
        # returns a generator, and closing it is what tears down the HTTP stream.
        generator = cast(Generator[AgentEvent, None, None], AnthropicBrain(keyed).respond("hello"))
        next(generator)  # start it
        generator.close()

        assert fake.streams[0].closed


class TestNonStreaming:
    def test_it_still_produces_sentences(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`llm.streaming: false` must work, not silently do nothing."""
        keyed.llm.streaming = False
        fake = FakeAnthropic([["Good evening. The kettle is on."]])
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed), "hello")
        assert sentences_of(events) == ["Good evening.", "The kettle is on."]

    def test_it_records_the_exchange(self, keyed: Config, monkeypatch: pytest.MonkeyPatch) -> None:
        keyed.llm.streaming = False
        fake_anthropic.install(monkeypatch, FakeAnthropic([["Certainly."]]))
        brain = AnthropicBrain(keyed)
        events_of(brain, "hello")
        assert len(brain.history) == 2


class TestHistory:
    def test_an_exchange_is_remembered(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_anthropic.install(monkeypatch, FakeAnthropic([["Certainly."]]))
        brain = AnthropicBrain(keyed)
        events_of(brain, "put the kettle on")

        assert brain.history == [
            {"role": "user", "content": "put the kettle on"},
            {"role": "assistant", "content": "Certainly."},
        ]

    def test_history_is_sent_on_the_next_turn(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic([["First."], ["Second."]])
        fake_anthropic.install(monkeypatch, fake)
        brain = AnthropicBrain(keyed)
        events_of(brain, "one")
        events_of(brain, "two")

        assert fake.requests[1]["messages"] == [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "First."},
            {"role": "user", "content": "two"},
        ]

    def test_history_is_trimmed_to_the_configured_length(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keyed.llm.history_turns = 2
        fake = FakeAnthropic([[f"Reply {i}."] for i in range(5)])
        fake_anthropic.install(monkeypatch, fake)
        brain = AnthropicBrain(keyed)
        for index in range(5):
            events_of(brain, f"question {index}")

        # 2 turns of history, plus the new user message.
        assert len(fake.requests[-1]["messages"]) == 5

    def test_trimming_never_starts_on_an_assistant_message(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dangling reply to a question the model cannot see confuses it."""
        keyed.llm.history_turns = 1
        fake = FakeAnthropic([[f"Reply {i}."] for i in range(4)])
        fake_anthropic.install(monkeypatch, fake)
        brain = AnthropicBrain(keyed)
        for index in range(4):
            events_of(brain, f"question {index}")

        assert fake.requests[-1]["messages"][0]["role"] == "user"

    def test_reset_clears_the_conversation(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_anthropic.install(monkeypatch, FakeAnthropic([["Certainly."]]))
        brain = AnthropicBrain(keyed)
        events_of(brain, "hello")
        brain.reset()
        assert brain.history == []

    def test_a_failed_turn_leaves_no_trace(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dangling user message with no answer would poison the next request."""
        fake_anthropic.install(monkeypatch, FakeAnthropic(raise_on_call=RuntimeError("boom")))
        brain = AnthropicBrain(keyed)
        events_of(brain, "hello")
        assert brain.history == []


class TestFailures:
    def test_a_missing_key_is_reported_aloud_not_raised(self, cfg: Config) -> None:
        """JARVIS should still hear you and explain itself, not crash."""
        cfg.secrets.anthropic_api_key = None
        events = events_of(AnthropicBrain(cfg), "hello")
        assert isinstance(events[0], AgentFailed)
        assert events[0].spoken

    def test_an_api_error_becomes_a_spoken_apology(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_anthropic.install(monkeypatch, FakeAnthropic(raise_on_call=RuntimeError("kaboom")))
        events = events_of(AnthropicBrain(keyed), "hello")
        assert isinstance(events[-1], AgentFailed)

    def test_a_mid_stream_failure_is_caught(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The connection can drop after the first sentence has been spoken."""
        fake = FakeAnthropic(
            [["Good evening. ", "The kettle", " is on."]],
            raise_mid_stream=(1, RuntimeError("connection reset")),
        )
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed), "hello")
        assert sentences_of(events) == ["Good evening."]
        assert isinstance(events[-1], AgentFailed)

    def test_a_refusal_is_explained_rather_than_played_as_silence(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal is HTTP 200 with no usable text; speaking it would be silence."""
        fake = FakeAnthropic(
            [[""]], stop_reason="refusal", stop_details=FakeStopDetails(category="cyber")
        )
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed), "do something dodgy")
        failure = next(e for e in events if isinstance(e, AgentFailed))
        assert "can't help" in failure.spoken
        assert "refusal" in failure.detail

    @pytest.mark.parametrize(
        ("exception_name", "expected"),
        [
            ("AuthenticationError", "key was rejected"),
            ("RateLimitError", "rate limited"),
            ("APIConnectionError", "reach the network"),
            ("APITimeoutError", "too long"),
            ("NotFoundError", "isn't available"),
        ],
    )
    def test_each_api_error_gets_its_own_spoken_message(
        self, exception_name: str, expected: str
    ) -> None:
        """A generic apology tells the user nothing about whether to retry."""
        exc = type(exception_name, (Exception,), {})("detail")
        spoken, _ = describe_failure(exc)
        assert expected in spoken

    def test_an_overloaded_response_suggests_retrying(self) -> None:
        exc = type("APIStatusError", (Exception,), {})("busy")
        exc.status_code = 529
        spoken, _ = describe_failure(exc)
        assert "busy" in spoken

    def test_a_jarvis_error_keeps_its_detail(self) -> None:
        spoken, detail = describe_failure(AuthError("no key", remedy="set it"))
        assert spoken
        assert "set it" in detail

    def test_an_unknown_exception_still_produces_speech(self) -> None:
        spoken, detail = describe_failure(ValueError("something odd"))
        assert spoken
        assert "ValueError" in detail


class TestSystemPrompt:
    def test_it_forbids_markdown(self, cfg: Config) -> None:
        assert "No markdown" in build_system_prompt(cfg)

    def test_it_includes_the_persona(self, cfg: Config) -> None:
        assert "British butler" in build_system_prompt(cfg)

    def test_the_time_comes_last(self, cfg: Config) -> None:
        """Stable content first, so the prefix can be cached later."""
        prompt = build_system_prompt(cfg, now=datetime(2026, 9, 14, 15, 20))
        assert prompt.rstrip().endswith(".")
        assert "current date and time" in prompt.split("\n\n")[-1]

    def test_facts_are_injected_when_present(self, cfg: Config) -> None:
        prompt = build_system_prompt(cfg, facts=["sister is called Priya"])
        assert "sister is called Priya" in prompt

    def test_no_facts_section_when_there_are_none(self, cfg: Config) -> None:
        assert "Things you have been told" not in build_system_prompt(cfg)

    def test_units_follow_the_config(self, cfg: Config) -> None:
        assert "metric" in build_system_prompt(cfg)
        cfg.general.units = "imperial"
        assert "imperial" in build_system_prompt(cfg)

    def test_an_unknown_timezone_warns_rather_than_failing(self, cfg: Config) -> None:
        """Being an hour out is a better failure than refusing to start."""
        cfg.general.timezone = "Mars/Olympus_Mons"
        assert "current date and time" in build_system_prompt(cfg)


class TestEchoBrain:
    def test_it_repeats_the_input(self) -> None:
        events = events_of(EchoBrain(), "what time is it")
        assert sentences_of(events) == ["what time is it"]

    def test_it_splits_multiple_sentences(self) -> None:
        events = events_of(EchoBrain(), "One. Two.")
        assert sentences_of(events) == ["One.", "Two."]

    def test_it_needs_no_api_key(self) -> None:
        """Which is the whole reason it exists."""
        events = events_of(EchoBrain(), "hello")
        assert not any(isinstance(e, AgentFailed) for e in events)

    def test_empty_input_short_circuits(self) -> None:
        events = events_of(EchoBrain(), "  ")
        assert len(events) == 1
        assert isinstance(events[0], TurnFinished)

    def test_it_keeps_history(self) -> None:
        brain = EchoBrain()
        list(brain.respond("hello"))
        assert len(brain.history) == 2
        brain.reset()
        assert brain.history == []


class TestToolLoop:
    """The brain's side of tool use: asking, gating, and feeding results back."""

    @pytest.fixture
    def dispatcher(self, keyed: Config):  # type: ignore[no-untyped-def]
        import jarvis.tools  # noqa: F401 - registers the built-ins
        from jarvis.audit import NullAuditLog
        from jarvis.tools.dispatch import ToolDispatcher

        keyed.tools.filesystem.workspace.mkdir(parents=True, exist_ok=True)
        return ToolDispatcher(keyed, NullAuditLog())

    def test_tool_schemas_are_sent(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed, dispatcher=dispatcher), "hello")

        names = [schema["name"] for schema in fake.requests[0]["tools"]]
        assert "get_time" in names
        assert "web_search" in names

    def test_no_tools_key_when_none_are_enabled(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)
        events_of(AnthropicBrain(keyed), "hello")
        assert "tools" not in fake.requests[0]

    def test_a_tool_call_is_executed_and_fed_back(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic([[""], ["It is Monday."]])
        fake.script_tool_call("get_time", {}, call_id="call_1")
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed, dispatcher=dispatcher), "what day is it")

        assert any(isinstance(e, ToolCallStarted) for e in events)
        assert any(isinstance(e, ToolCallFinished) and e.ok for e in events)
        # The second request carries the tool result.
        results = fake.requests[1]["messages"][-1]["content"]
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "call_1"

    def test_a_confirmed_tool_runs(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic([[""], ["Done."]])
        fake.script_tool_call("write_file", {"path": "x.txt", "content": "hi"}, call_id="call_w")
        fake_anthropic.install(monkeypatch, fake)

        brain = AnthropicBrain(keyed, dispatcher=dispatcher)
        collected = []
        for event in brain.respond("write a file"):
            collected.append(event)
            if isinstance(event, ConfirmationRequired):
                brain.confirm(event.call_id, True)

        assert any(isinstance(e, ConfirmationRequired) for e in collected)
        assert (keyed.tools.filesystem.workspace / "x.txt").read_text() == "hi"

    def test_a_declined_tool_does_not_run(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic([[""], ["Understood."]])
        fake.script_tool_call("write_file", {"path": "x.txt", "content": "hi"}, call_id="call_w")
        fake_anthropic.install(monkeypatch, fake)

        brain = AnthropicBrain(keyed, dispatcher=dispatcher)
        for event in brain.respond("write a file"):
            if isinstance(event, ConfirmationRequired):
                brain.confirm(event.call_id, False)

        assert not (keyed.tools.filesystem.workspace / "x.txt").exists()

    def test_never_answering_the_confirmation_counts_as_no(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The safe default when a decision is somehow never collected."""
        fake = FakeAnthropic([[""], ["Understood."]])
        fake.script_tool_call("write_file", {"path": "x.txt", "content": "hi"}, call_id="call_w")
        fake_anthropic.install(monkeypatch, fake)

        events_of(AnthropicBrain(keyed, dispatcher=dispatcher), "write a file")
        assert not (keyed.tools.filesystem.workspace / "x.txt").exists()

    def test_the_confirmation_prompt_quotes_the_command(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeAnthropic([[""], ["Done."]])
        fake.script_tool_call("run_shell", {"command": "git push --force"}, call_id="call_s")
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed, dispatcher=dispatcher), "push")
        prompt = next(e for e in events if isinstance(e, ConfirmationRequired)).prompt
        assert "git push --force" in prompt

    def test_the_tool_round_limit_is_enforced(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A confused model must not loop up a bill while the user waits."""
        keyed.tools.max_iterations = 3
        fake = FakeAnthropic([[""]] * 10)
        fake.script_tool_call("get_time", {}, call_id="call_x", forever=True)
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed, dispatcher=dispatcher), "loop")
        assert isinstance(events[-1], AgentFailed)
        assert "circles" in events[-1].spoken
        assert len(fake.requests) == 3

    def test_a_paused_turn_is_resumed(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server tool pausing must not silently truncate the answer."""
        fake = FakeAnthropic(
            [["Searching"], [" and done."]], stop_reasons=["pause_turn", "end_turn"]
        )
        fake_anthropic.install(monkeypatch, fake)

        events = events_of(AnthropicBrain(keyed, dispatcher=dispatcher), "search")
        assert len(fake.requests) == 2
        assert isinstance(events[-1], TurnFinished)

    def test_history_trimming_never_starts_on_tool_results(
        self, keyed: Config, dispatcher: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Results for a tool_use block the model cannot see are rejected."""
        brain = AnthropicBrain(keyed, dispatcher=dispatcher)
        keyed.llm.history_turns = 1
        brain._messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}],
            },
            {"role": "assistant", "content": "second"},
        ]
        trimmed = brain._trimmed_history()
        assert trimmed == [] or trimmed[0]["role"] == "user"
        assert not any(
            isinstance(m.get("content"), list)
            and m["content"]
            and m["content"][0].get("type") == "tool_result"
            for m in trimmed[:1]
        )


class TestMemoryIntegration:
    """Facts in, transcript out, extraction in the background."""

    @pytest.fixture
    def store(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        from jarvis.memory.store import SqliteMemory

        memory = SqliteMemory(tmp_path / "m.db")
        memory.initialise()
        return memory

    def test_recalled_facts_reach_the_system_prompt(
        self, keyed: Config, store: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_fact("the user's sister is called Priya")
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)

        events_of(AnthropicBrain(keyed, memory=store), "what is my sister called")
        assert "Priya" in fake.requests[0]["system"]

    def test_irrelevant_facts_are_not_injected(
        self, keyed: Config, store: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add_fact("the user's sister is called Priya")
        fake = FakeAnthropic()
        fake_anthropic.install(monkeypatch, fake)

        events_of(AnthropicBrain(keyed, memory=store), "what is the capital of Peru")
        assert "Priya" not in fake.requests[0]["system"]

    def test_the_exchange_is_written_to_the_transcript(
        self, keyed: Config, store: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_anthropic.install(monkeypatch, FakeAnthropic([["Certainly."]]))
        events_of(AnthropicBrain(keyed, memory=store), "put the kettle on")

        texts = [m.text for m in store.recent_messages(10)]
        assert texts == ["put the kettle on", "Certainly."]

    def test_a_memory_failure_does_not_break_the_conversation(
        self, keyed: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JARVIS with no recall is far better than JARVIS that stops answering."""

        class BrokenMemory:
            def recall(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("database is on fire")

            def add_message(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("still on fire")

        fake_anthropic.install(monkeypatch, FakeAnthropic([["Very good."]]))
        events = events_of(AnthropicBrain(keyed, memory=BrokenMemory()), "hello")
        assert isinstance(events[-1], TurnFinished)
        assert events[-1].text == "Very good."

    def test_extraction_is_offered_the_finished_exchange(
        self, keyed: Config, store: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, str]] = []

        class Recorder:
            def consider(self, user_text: str, reply: str) -> None:
                seen.append((user_text, reply))

        fake_anthropic.install(monkeypatch, FakeAnthropic([["Noted."]]))
        events_of(AnthropicBrain(keyed, memory=store, extractor=Recorder()), "I prefer metric")
        assert seen == [("I prefer metric", "Noted.")]

    def test_a_failed_turn_is_not_offered_for_extraction(
        self, keyed: Config, store: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[str, str]] = []

        class Recorder:
            def consider(self, user_text: str, reply: str) -> None:
                seen.append((user_text, reply))

        fake_anthropic.install(monkeypatch, FakeAnthropic(raise_on_call=RuntimeError("x")))
        events_of(AnthropicBrain(keyed, memory=store, extractor=Recorder()), "hello")
        assert seen == []
