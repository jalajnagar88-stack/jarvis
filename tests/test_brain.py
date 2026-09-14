"""The Anthropic brain.

The SDK is replaced with a fake that mirrors the streaming surface the brain
actually uses, so these tests cover request construction, sentence-by-sentence
emission, history management, and -- most importantly -- what JARVIS says out
loud when the API will not cooperate.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from datetime import datetime
from typing import cast

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
    SentenceComplete,
    TextDelta,
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
