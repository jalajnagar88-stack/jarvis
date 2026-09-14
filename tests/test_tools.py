"""The tool layer: registry, dispatcher, and each built-in tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

import jarvis.tools  # noqa: F401 - registers the built-ins
from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.errors import SafetyViolation, ToolError
from jarvis.tools.builtin.clock import TimerService, _spoken_duration
from jarvis.tools.dispatch import ToolDispatcher
from jarvis.tools.registry import REGISTRY, ToolContext, ToolRegistry, tool
from jarvis.tools.server_tools import web_search_definition


@pytest.fixture
def audit(cfg: Config) -> AuditLog:
    return AuditLog(cfg.logging.audit_file)


@pytest.fixture
def dispatcher(cfg: Config, audit: AuditLog) -> ToolDispatcher:
    return ToolDispatcher(cfg, audit, timers=TimerService())


@pytest.fixture
def workspace(cfg: Config) -> Path:
    cfg.tools.filesystem.workspace.mkdir(parents=True, exist_ok=True)
    return cfg.tools.filesystem.workspace


class TestRegistry:
    def test_every_configured_tool_exists(self, cfg: Config) -> None:
        """config.yaml and the code must not drift apart."""
        missing = [
            name for name in cfg.tools.enabled if name not in REGISTRY and name != "web_search"
        ]
        assert not missing, f"config.yaml enables unregistered tools: {missing}"

    def test_schemas_forbid_extra_properties(self) -> None:
        for name, registered in REGISTRY.tools.items():
            schema = registered.schema()
            assert schema["input_schema"]["additionalProperties"] is False, name
            assert schema["strict"] is True, name

    def test_every_tool_has_a_description(self) -> None:
        """The description is the only thing telling the model when to use it."""
        for name, registered in REGISTRY.tools.items():
            assert len(registered.description) > 40, name

    def test_duplicate_names_are_rejected(self) -> None:
        registry = ToolRegistry()

        class Args(BaseModel):
            pass

        @tool(name="dup", description="x" * 50, args=Args, registry=registry)
        def first(args: Args, ctx: ToolContext) -> str:
            return ""

        with pytest.raises(ValueError, match="named 'dup'"):

            @tool(name="dup", description="y" * 50, args=Args, registry=registry)
            def second(args: Args, ctx: ToolContext) -> str:
                return ""

    def test_unknown_names_in_config_are_skipped_not_fatal(self) -> None:
        registry = ToolRegistry()
        assert registry.enabled(["nope"]) == []

    def test_bad_arguments_produce_a_readable_error(self) -> None:
        registered = REGISTRY.get("set_timer")
        assert registered is not None
        with pytest.raises(ToolError, match="Invalid arguments"):
            registered.parse({"seconds": "soon"})


class TestConfirmationGate:
    """The dispatcher enforces confirmation; a tool cannot waive it."""

    def test_writes_require_confirmation(self, dispatcher: ToolDispatcher) -> None:
        assert dispatcher.needs_confirmation("write_file")
        assert dispatcher.needs_confirmation("run_shell")
        assert dispatcher.needs_confirmation("open_app")

    def test_reads_do_not(self, dispatcher: ToolDispatcher) -> None:
        assert not dispatcher.needs_confirmation("get_time")
        assert not dispatcher.needs_confirmation("read_file")

    def test_an_unapproved_call_does_not_run(
        self, dispatcher: ToolDispatcher, workspace: Path
    ) -> None:
        outcome = dispatcher.execute(
            "write_file", {"path": "x.txt", "content": "hello"}, approved=False
        )
        assert "declined" in outcome.content
        assert not (workspace / "x.txt").exists()

    def test_a_missing_decision_counts_as_refusal(
        self, dispatcher: ToolDispatcher, workspace: Path
    ) -> None:
        """If the confirmation is somehow skipped, the safe default is no."""
        outcome = dispatcher.execute("write_file", {"path": "x.txt", "content": "hello"})
        assert "declined" in outcome.content
        assert not (workspace / "x.txt").exists()

    def test_an_approved_call_runs(self, dispatcher: ToolDispatcher, workspace: Path) -> None:
        outcome = dispatcher.execute(
            "write_file", {"path": "x.txt", "content": "hello"}, approved=True
        )
        assert not outcome.is_error
        assert (workspace / "x.txt").read_text() == "hello"

    def test_the_prompt_quotes_the_command_verbatim(self, dispatcher: ToolDispatcher) -> None:
        """The user agrees to this exact string, not to a paraphrase."""
        prompt = dispatcher.describe("run_shell", {"command": "git push --force"})
        assert "git push --force" in prompt

    def test_the_prompt_warns_about_compound_commands(self, dispatcher: ToolDispatcher) -> None:
        prompt = dispatcher.describe("run_shell", {"command": "ls | xargs rm"})
        assert "shell operators" in prompt

    def test_the_write_prompt_names_the_file(self, dispatcher: ToolDispatcher) -> None:
        prompt = dispatcher.describe("write_file", {"path": "notes.md", "content": "hi"})
        assert "notes.md" in prompt

    def test_a_declined_call_is_audited(self, dispatcher: ToolDispatcher, cfg: Config) -> None:
        dispatcher.execute("run_shell", {"command": "ls"}, approved=False)
        assert "declined" in cfg.logging.audit_file.read_text()


class TestDispatcher:
    def test_a_disabled_tool_is_refused(self, cfg: Config, audit: AuditLog) -> None:
        cfg.tools.enabled = ["get_time"]
        outcome = ToolDispatcher(cfg, audit).execute("run_shell", {"command": "ls"}, approved=True)
        assert outcome.is_error
        assert "turned off" in outcome.content

    def test_an_unknown_tool_is_reported_not_raised(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("teleport", {})
        assert outcome.is_error
        assert "no tool called" in outcome.content

    def test_a_crashing_tool_does_not_end_the_conversation(
        self, cfg: Config, audit: AuditLog
    ) -> None:
        registry = ToolRegistry()

        class Args(BaseModel):
            pass

        @tool(name="explode", description="d" * 50, args=Args, registry=registry)
        def explode(args: Args, ctx: ToolContext) -> str:
            raise RuntimeError("boom")

        cfg.tools.enabled = ["explode"]
        outcome = ToolDispatcher(cfg, audit, registry=registry).execute("explode", {})
        assert outcome.is_error
        assert "failed unexpectedly" in outcome.content

    def test_a_safety_violation_is_returned_as_an_error(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("run_shell", {"command": "rm -rf /"}, approved=True)
        assert outcome.is_error
        assert "blocked" in outcome.content.lower()

    def test_a_blocked_command_is_audited_as_denied(
        self, dispatcher: ToolDispatcher, cfg: Config
    ) -> None:
        dispatcher.execute("run_shell", {"command": "rm -rf /"}, approved=True)
        assert "tool.blocked" in cfg.logging.audit_file.read_text()

    def test_schemas_include_web_search_when_enabled(self, dispatcher: ToolDispatcher) -> None:
        names = [schema["name"] for schema in dispatcher.schemas()]
        assert "web_search" in names
        assert "get_time" in names

    def test_schemas_omit_web_search_when_disabled(self, cfg: Config, audit: AuditLog) -> None:
        cfg.tools.enabled = ["get_time"]
        names = [schema["name"] for schema in ToolDispatcher(cfg, audit).schemas()]
        assert names == ["get_time"]

    def test_long_arguments_are_trimmed_in_the_audit_log(
        self, dispatcher: ToolDispatcher, cfg: Config
    ) -> None:
        """An audit log nobody can read is not an audit log."""
        dispatcher.execute("write_file", {"path": "big.txt", "content": "x" * 5000}, approved=False)
        assert "x" * 5000 not in cfg.logging.audit_file.read_text()


class TestWebSearchDefinition:
    def test_it_uses_the_current_tool_type(self, cfg: Config) -> None:
        assert web_search_definition(cfg)["type"] == "web_search_20260209"

    def test_older_models_get_the_basic_variant(self, cfg: Config) -> None:
        cfg.llm.model = "claude-haiku-4-5"
        assert web_search_definition(cfg)["type"] == "web_search_20250305"

    def test_max_uses_is_passed_through(self, cfg: Config) -> None:
        assert web_search_definition(cfg)["max_uses"] == cfg.tools.web_search.max_uses

    def test_never_sends_both_domain_lists(self, cfg: Config) -> None:
        """The API rejects a request carrying both."""
        cfg.tools.web_search.allowed_domains = ["example.com"]
        cfg.tools.web_search.blocked_domains = ["spam.example"]
        definition = web_search_definition(cfg)
        assert "allowed_domains" in definition
        assert "blocked_domains" not in definition


class TestFileTools:
    def test_reading_a_file(self, dispatcher: ToolDispatcher, workspace: Path) -> None:
        (workspace / "notes.md").write_text("the kettle is on")
        outcome = dispatcher.execute("read_file", {"path": "notes.md"})
        assert outcome.content == "the kettle is on"

    def test_reading_outside_the_workspace_is_blocked(
        self, dispatcher: ToolDispatcher, workspace: Path
    ) -> None:
        outcome = dispatcher.execute("read_file", {"path": "../../etc/passwd"})
        assert outcome.is_error
        assert "outside the workspace" in outcome.content

    def test_a_missing_file_is_a_readable_error(
        self, dispatcher: ToolDispatcher, workspace: Path
    ) -> None:
        outcome = dispatcher.execute("read_file", {"path": "absent.md"})
        assert outcome.is_error
        assert "no file at" in outcome.content

    def test_reading_a_directory_lists_it(
        self, dispatcher: ToolDispatcher, workspace: Path
    ) -> None:
        (workspace / "sub").mkdir()
        (workspace / "sub" / "a.txt").write_text("a")
        outcome = dispatcher.execute("read_file", {"path": "sub"})
        assert "a.txt" in outcome.content

    def test_a_large_file_is_truncated(self, dispatcher: ToolDispatcher, workspace: Path) -> None:
        (workspace / "big.txt").write_text("y" * 9000)
        outcome = dispatcher.execute("read_file", {"path": "big.txt"})
        assert "truncated" in outcome.content

    def test_appending(self, dispatcher: ToolDispatcher, workspace: Path) -> None:
        (workspace / "log.txt").write_text("one\n")
        dispatcher.execute(
            "write_file",
            {"path": "log.txt", "content": "two\n", "append": True},
            approved=True,
        )
        assert (workspace / "log.txt").read_text() == "one\ntwo\n"

    def test_writing_outside_the_workspace_is_blocked_even_when_approved(
        self, dispatcher: ToolDispatcher, tmp_path: Path
    ) -> None:
        """Approval is not permission to leave the sandbox."""
        target = tmp_path / "escaped.txt"
        outcome = dispatcher.execute(
            "write_file", {"path": str(target), "content": "x"}, approved=True
        )
        assert outcome.is_error
        assert not target.exists()

    def test_notes_are_dated_and_appended(
        self, dispatcher: ToolDispatcher, workspace: Path
    ) -> None:
        dispatcher.execute("add_note", {"text": "buy milk"})
        dispatcher.execute("add_note", {"text": "call Priya"})
        content = (workspace / "notes.md").read_text()
        assert "buy milk" in content and "call Priya" in content
        assert content.count("\n") == 2


class TestClockTools:
    def test_get_time_returns_something_speakable(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("get_time", {})
        assert not outcome.is_error
        assert any(
            day in outcome.content
            for day in (
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            )
        )

    def test_an_unknown_timezone_is_a_readable_error(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("get_time", {"timezone": "Mars/Olympus"})
        assert outcome.is_error
        assert "not a known timezone" in outcome.content

    def test_a_timer_is_scheduled(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("set_timer", {"seconds": 600, "label": "the pasta"})
        assert "ten minutes" in outcome.content or "10 minutes" in outcome.content
        assert "the pasta" in outcome.content

    def test_a_timer_announces_itself(self) -> None:
        heard: list[str] = []
        service = TimerService(heard.append)
        service.schedule(0.01, "the pasta")
        import time

        time.sleep(0.2)
        assert heard == ["Your the pasta timer has finished."]

    def test_a_failing_announcer_does_not_kill_the_thread(self) -> None:
        def explode(message: str) -> None:
            raise RuntimeError("no speaker")

        service = TimerService(explode)
        service.schedule(0.01, "")
        import time

        time.sleep(0.2)  # must not raise into the test

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (1, "1 second"),
            (30, "30 seconds"),
            (60, "1 minute"),
            (600, "10 minutes"),
            (3600, "1 hour"),
            (5400, "1 hour and 30 minutes"),
        ],
    )
    def test_durations_are_spoken_naturally(self, seconds: int, expected: str) -> None:
        assert _spoken_duration(seconds) == expected


class TestShellTool:
    def test_a_simple_command_runs(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("run_shell", {"command": "echo hello"}, approved=True)
        assert "hello" in outcome.content

    def test_a_failing_command_reports_its_status(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("run_shell", {"command": "exit 3"}, approved=True)
        assert "status 3" in outcome.content

    def test_output_is_truncated(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute(
            "run_shell", {"command": "python3 -c \"print('z'*5000)\""}, approved=True
        )
        assert "truncated" in outcome.content

    def test_a_timeout_is_reported(self, cfg: Config, audit: AuditLog) -> None:
        cfg.tools.shell.timeout_seconds = 0.3
        outcome = ToolDispatcher(cfg, audit).execute(
            "run_shell", {"command": "sleep 5"}, approved=True
        )
        assert outcome.is_error
        assert "still running" in outcome.content

    def test_it_runs_in_the_configured_directory(
        self, dispatcher: ToolDispatcher, cfg: Config
    ) -> None:
        outcome = dispatcher.execute("run_shell", {"command": "pwd"}, approved=True)
        assert str(cfg.tools.shell.cwd.resolve()) in outcome.content

    def test_the_denylist_is_rechecked_inside_the_tool(self, cfg: Config, audit: AuditLog) -> None:
        """A tool that is safe only because of its caller is not safe."""
        from jarvis.tools.builtin.shell import RunShellArgs, run_shell

        context = ToolContext(cfg, audit)
        with pytest.raises(SafetyViolation):
            run_shell(RunShellArgs(command="rm -rf /"), context)


class TestWeatherTool:
    def test_it_describes_the_weather(
        self, dispatcher: ToolDispatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get(url: str, params: dict[str, Any], timeout: float) -> Any:
            payload: dict[str, Any]
            if "geocoding" in url:
                payload = {
                    "results": [
                        {
                            "latitude": 55.95,
                            "longitude": -3.19,
                            "name": "Edinburgh",
                            "country": "United Kingdom",
                        }
                    ]
                }
            else:
                payload = {
                    "current": {
                        "temperature_2m": 12.4,
                        "apparent_temperature": 9.0,
                        "weather_code": 61,
                        "wind_speed_10m": 25.0,
                    },
                    "daily": {
                        "temperature_2m_max": [14.0],
                        "temperature_2m_min": [7.0],
                        "precipitation_probability_max": [80],
                    },
                }
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        outcome = dispatcher.execute("get_weather", {"location": "Edinburgh"})

        assert "Edinburgh" in outcome.content
        assert "12 degrees" in outcome.content
        assert "raining" in outcome.content
        assert "80 percent" in outcome.content

    def test_an_unknown_place_is_a_readable_error(
        self, dispatcher: ToolDispatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            httpx,
            "get",
            lambda url, **k: httpx.Response(
                200, json={"results": []}, request=httpx.Request("GET", url)
            ),
        )
        outcome = dispatcher.execute("get_weather", {"location": "Atlantis"})
        assert outcome.is_error
        assert "couldn't find" in outcome.content

    def test_no_location_and_no_default_asks_rather_than_guessing(
        self, dispatcher: ToolDispatcher
    ) -> None:
        outcome = dispatcher.execute("get_weather", {})
        assert outcome.is_error
        assert "Ask the user" in outcome.content

    def test_a_network_failure_is_a_readable_error(
        self, dispatcher: ToolDispatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: Any, **kwargs: Any) -> Any:
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(httpx, "get", explode)
        outcome = dispatcher.execute("get_weather", {"location": "Edinburgh"})
        assert outcome.is_error
        assert "unreachable" in outcome.content


class TestSystemTools:
    def test_open_app_degrades_on_an_unknown_platform(
        self, dispatcher: ToolDispatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("jarvis.tools.builtin.system._system", lambda: "Plan9")
        outcome = dispatcher.execute("open_app", {"name": "Safari"}, approved=True)
        assert outcome.is_error
        assert "don't know how" in outcome.content

    def test_volume_requires_exactly_one_argument(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("control_volume", {"level": 50, "mute": True})
        assert outcome.is_error
        assert "exactly one" in outcome.content

    def test_volume_degrades_without_a_backend(
        self, dispatcher: ToolDispatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("jarvis.tools.builtin.system._system", lambda: "Linux")
        monkeypatch.setattr("jarvis.tools.builtin.system.shutil.which", lambda name: None)
        outcome = dispatcher.execute("control_volume", {"level": 50})
        assert outcome.is_error
        assert "pactl" in outcome.content

    def test_macos_volume_uses_osascript(
        self, dispatcher: ToolDispatcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr("jarvis.tools.builtin.system._system", lambda: "Darwin")

        def record(command: list[str]) -> str:
            calls.append(command)
            return "40"

        monkeypatch.setattr("jarvis.tools.builtin.system._run", record)
        outcome = dispatcher.execute("control_volume", {"level": 30})
        assert "30 percent" in outcome.content
        assert any("osascript" in c[0] for c in calls)


class TestServerToolsAreNotDispatchedLocally:
    def test_web_search_is_not_looked_up_in_the_registry(
        self, dispatcher: ToolDispatcher, caplog: pytest.LogCaptureFixture
    ) -> None:
        """It runs on Anthropic's servers; warning about it would be wrong."""
        import logging

        with caplog.at_level(logging.WARNING):
            names = [t.name for t in dispatcher.enabled_tools()]
        assert "web_search" not in names
        assert "web_search" not in caplog.text
