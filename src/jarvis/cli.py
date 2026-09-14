"""Command line entry point.

``python -m jarvis`` with no arguments runs the health check, which is the
milestone 1 deliverable: it inspects every subsystem and reports, in plain
language, what is ready and what is missing.

Exit codes: 0 when everything needed is present (warnings included), 1 when at
least one check failed, 2 when configuration could not be loaded at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from jarvis import __version__, factory, runner
from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.errors import JarvisError
from jarvis.health import (
    Check,
    HealthReport,
    Status,
    environment_summary,
    list_audio_devices,
    run_health_checks,
)
from jarvis.logging_setup import get_logger, setup_logging

_STATUS_MARKER = {
    Status.OK: ("[ ok ]", "green"),
    Status.WARN: ("[warn]", "yellow"),
    Status.FAIL: ("[FAIL]", "red"),
    Status.SKIP: ("[skip]", "dim"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="A local-first voice assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m jarvis                 run the health check\n"
            "  python -m jarvis run             start listening for the wake word\n"
            "  python -m jarvis run --text      text REPL, no microphone needed\n"
            "  python -m jarvis run --no-brain  echo mode; no API key needed\n"
            "  python -m jarvis say 'Good evening.'   audition the voice\n"
            "  python -m jarvis devices         list microphones and speakers\n"
            "  python -m jarvis health --json   machine-readable health report\n"
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        help="path to config.yaml (default: ./config.yaml, or $JARVIS_CONFIG)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="override logging.level for this run",
    )
    parser.add_argument("-V", "--version", action="version", version=f"jarvis {__version__}")

    sub = parser.add_subparsers(dest="command")

    health = sub.add_parser("health", help="check every subsystem and report (default)")
    health.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    sub.add_parser("devices", help="list audio input and output devices")

    run = sub.add_parser("run", help="start the assistant")
    run.add_argument(
        "--text",
        action="store_true",
        help="run as a text REPL instead of listening (no microphone needed)",
    )
    run.add_argument(
        "--no-brain",
        action="store_true",
        help="echo what you said instead of thinking about it; needs no API key",
    )

    speak = sub.add_parser("say", help="speak one phrase and exit (for auditioning a voice)")
    speak.add_argument("text", nargs="+", help="what to say")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    try:
        cfg = Config.load(args.config)
    except JarvisError as exc:
        console.print(Text("Configuration error", style="bold red"))
        console.print(str(exc))
        return 2

    if args.log_level:
        cfg.logging.level = args.log_level

    setup_logging(cfg.logging, console=Console(stderr=True))
    log = get_logger("cli")
    log.debug("Loaded configuration from %s", cfg.source_path)

    command = args.command or "health"
    if command == "health":
        return _cmd_health(cfg, console, as_json=getattr(args, "json", False))
    if command == "devices":
        return _cmd_devices(console)
    if command == "run":
        return _cmd_run(cfg, console, text_mode=args.text, no_brain=args.no_brain)
    if command == "say":
        return runner.say(cfg, console, " ".join(args.text))

    parser.error(f"unknown command {command!r}")


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #


def _cmd_health(cfg: Config, console: Console, *, as_json: bool) -> int:
    report = run_health_checks(cfg)

    if as_json:
        console.print_json(json.dumps(_report_to_dict(cfg, report)))
        return 1 if report.overall is Status.FAIL else 0

    _render_report(cfg, report, console)

    audit = AuditLog(cfg.logging.audit_file)
    audit.record(
        "health.check",
        outcome="ok" if report.overall is not Status.FAIL else "error",
        detail=f"{report.overall.value}: "
        f"{sum(1 for c in report.checks if c.status is Status.FAIL)} failed",
    )
    return 1 if report.overall is Status.FAIL else 0


def _render_report(cfg: Config, report: HealthReport, console: Console) -> None:
    console.print()
    console.print(Text(f"  {cfg.general.name}", style="bold"), end="")
    console.print(Text(f"  v{__version__}", style="dim"))

    env = environment_summary()
    console.print(Text(f"  {env['platform']} | Python {env['python']}", style="dim"))
    console.print(Text(f"  config: {cfg.source_path}", style="dim"))
    console.print()

    # One small table per subsystem. A single wide table looks tidy at 120
    # columns and unreadable at 80, and a health check is exactly the thing
    # people run in a cramped terminal.
    for subsystem, checks in _group_by_subsystem(report.checks):
        console.print(Text(f"  {subsystem}", style="bold"))
        table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
        table.add_column(width=8, no_wrap=True)
        table.add_column(width=24)
        table.add_column(ratio=1)
        for check in checks:
            marker, style = _STATUS_MARKER[check.status]
            muted = "dim" if check.status is Status.SKIP else ""
            table.add_row(
                Text(f"  {marker}", style=style),
                Text(check.name, style=muted),
                Text(check.detail, style=muted),
            )
        console.print(table)
        console.print()

    _render_capabilities(report, console)
    _render_remedies(report, console)
    _render_verdict(report, console)


def _group_by_subsystem(checks: list[Check]) -> list[tuple[str, list[Check]]]:
    """Preserve the order checks were produced in, grouping consecutive runs."""
    grouped: list[tuple[str, list[Check]]] = []
    for check in checks:
        if grouped and grouped[-1][0] == check.subsystem:
            grouped[-1][1].append(check)
        else:
            grouped.append((check.subsystem, [check]))
    return grouped


def _render_capabilities(report: HealthReport, console: Console) -> None:
    console.print(Text("  What works right now", style="bold"))
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    table.add_column(width=8, no_wrap=True)
    table.add_column(width=28)
    table.add_column(ratio=1)
    for name, (available, why) in report.capabilities.items():
        marker, style = ("[ ok ]", "green") if available else ("[ -- ]", "yellow")
        table.add_row(
            Text(f"  {marker}", style=style),
            Text(name, style="" if available else "dim"),
            Text(why, style="dim"),
        )
    console.print(table)
    console.print()


def _render_remedies(report: HealthReport, console: Console) -> None:
    actionable = [c for c in report.checks if c.remedy and c.status in (Status.FAIL, Status.WARN)]
    if not actionable:
        return
    console.print(Text("  To fix", style="bold"))
    seen: set[str] = set()
    for check in actionable:
        assert check.remedy is not None
        if check.remedy in seen:
            continue
        seen.add(check.remedy)
        label = "required" if check.status is Status.FAIL else "optional"
        console.print(
            Text(f"  {label:>10}  ", style="red" if label == "required" else "yellow"), end=""
        )
        console.print(Text(check.remedy, overflow="ignore"), crop=False)
    console.print()


def _render_verdict(report: HealthReport, console: Console) -> None:
    failed = sum(1 for c in report.checks if c.status is Status.FAIL)
    warned = sum(1 for c in report.checks if c.status is Status.WARN)
    if report.overall is Status.FAIL:
        console.print(Text(f"  {failed} check(s) failed — see 'To fix' above.", style="bold red"))
        console.print(Text("  JARVIS will still run whatever is ready.", style="dim"))
    elif warned:
        console.print(
            Text(f"  All required checks passed, with {warned} warning(s).", style="bold yellow")
        )
    else:
        console.print(Text("  Every subsystem is ready.", style="bold green"))
    console.print()


def _report_to_dict(cfg: Config, report: HealthReport) -> dict[str, object]:
    return {
        "version": __version__,
        "config": str(cfg.source_path),
        "environment": environment_summary(),
        "overall": report.overall.value,
        "capabilities": {
            name: {"available": available, "note": note}
            for name, (available, note) in report.capabilities.items()
        },
        "checks": [_check_to_dict(c) for c in report.checks],
    }


def _check_to_dict(check: Check) -> dict[str, object]:
    return {
        "id": check.id,
        "subsystem": check.subsystem,
        "name": check.name,
        "status": check.status.value,
        "detail": check.detail,
        "remedy": check.remedy,
    }


# --------------------------------------------------------------------------- #
# devices
# --------------------------------------------------------------------------- #


def _cmd_devices(console: Console) -> int:
    devices = list_audio_devices()
    if not devices:
        console.print()
        console.print(Text("  No audio devices found.", style="bold yellow"))
        console.print(
            "  Either sounddevice is not installed (uv sync --extra voice) or PortAudio\n"
            "  cannot see any hardware. On macOS, check that Terminal has microphone\n"
            "  access under System Settings > Privacy & Security > Microphone."
        )
        console.print()
        return 1

    table = Table(title="Audio devices", header_style="bold", title_justify="left")
    table.add_column("Index", justify="right")
    table.add_column("Name")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Default")

    for dev in devices:
        default = ", ".join(
            part
            for part, flag in (("input", dev.is_default_input), ("output", dev.is_default_output))
            if flag
        )
        table.add_row(
            str(dev.index),
            dev.name,
            str(dev.max_input_channels),
            str(dev.max_output_channels),
            f"{dev.default_sample_rate:.0f}",
            default or "",
        )
    console.print()
    console.print(table)
    console.print()
    console.print(
        Text(
            "  Set audio.input_device / audio.output_device in config.yaml to either an\n"
            "  index from this table or any substring of a device name.",
            style="dim",
        )
    )
    console.print()
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def _cmd_run(cfg: Config, console: Console, *, text_mode: bool, no_brain: bool = False) -> int:
    """Start the assistant."""
    brain = factory.build_echo_brain() if no_brain else None
    if no_brain:
        console.print()
        console.print(
            Text("  Echo mode: repeating what you say, not thinking about it.", style="yellow")
        )

    if text_mode:
        return runner.run_text(cfg, console, brain=brain)

    blockers = _voice_blockers(cfg, need_brain=not no_brain)
    if blockers:
        console.print()
        console.print(Text("  Not ready to listen yet.", style="bold red"))
        for detail, remedy in blockers:
            console.print(Text(f"  - {detail}", style="red"))
            if remedy:
                console.print(Text(f"      {remedy}", style="dim"))
        console.print()
        console.print(
            Text("  `python -m jarvis run --text` works without any of this.", style="dim")
        )
        console.print()
        return 1

    return runner.run_voice(cfg, console)


def _voice_blockers(cfg: Config, *, need_brain: bool = True) -> list[tuple[str, str | None]]:
    """Checks that must pass before listening is even worth attempting.

    Starting the loop and letting it explode on the first missing model would
    technically work, but the failure would arrive after a ten-second model load
    and would name a file rather than a fix.

    A missing API key is deliberately *not* a blocker: JARVIS still wakes,
    listens and transcribes, then says out loud that it cannot think. Refusing
    to start would hide which half is broken.
    """
    report = run_health_checks(cfg)
    required = [
        "audio.backend",
        "audio.input",
        "audio.output",
        "wake.package",
        "wake.model",
        "stt.package",
        "stt.model",
        "tts.engine",
        "tts.voice",
    ]
    if need_brain:
        required.append("brain.sdk")
    blockers: list[tuple[str, str | None]] = []
    for check_id in required:
        check = report.by_id(check_id)
        if check is not None and check.status is Status.FAIL:
            blockers.append((f"{check.name}: {check.detail}", check.remedy))
    return blockers


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
