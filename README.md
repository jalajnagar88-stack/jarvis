# JARVIS

A local-first, always-listening desktop voice assistant. You say a wake word, it
listens, thinks, optionally acts on your machine or the web, and replies out
loud in a calm British-butler voice. It remembers things across sessions.

Built for macOS on Apple Silicon. The architecture is OS-agnostic; only two
tools (`open_app`, `control_volume`) are platform-specific, and they degrade
gracefully elsewhere.

## What leaves your machine

This matters, so it is stated first and precisely.

| Stage | Where it runs | What leaves the machine |
| --- | --- | --- |
| Wake word (openWakeWord) | Local | Nothing |
| Transcription (faster-whisper) | Local | Nothing |
| Reasoning (Anthropic API) | Cloud | **The transcribed text of your request**, the conversation history, and any retrieved facts |
| Speech (Piper) | Local | Nothing |
| Speech (ElevenLabs, opt-in) | Cloud | The reply text |
| Memory (SQLite) | Local | Nothing |

**Your audio never leaves the machine.** No recording is ever uploaded. What is
sent to Anthropic is the text that Whisper produced locally, exactly as it would
be if you had typed it. If you use the optional ElevenLabs voice instead of
Piper, the reply text is also sent to ElevenLabs — the default Piper setup sends
nothing anywhere.

Everything JARVIS does on your behalf is appended to a local audit log
(`logs/audit.log`) with timestamps.

## Requirements

- macOS 13 or newer on Apple Silicon (Intel works; Linux and Windows are
  supported by the audio layer but untested)
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- An [Anthropic API key](https://console.anthropic.com/settings/keys)
- About 1.5 GB of disk for models (2.5 GB if you install semantic memory)

## Setup

```bash
git clone <your-fork> jarvis
cd jarvis

# 1. System library for audio capture.
brew install portaudio

# 2. Dependencies.
#    `uv sync` alone gives you a working text-mode JARVIS in seconds.
#    Add the extras when you want it to actually listen and speak.
uv sync --extra voice --extra memory --extra dev

# 3. Secrets.
cp .env.example .env
$EDITOR .env          # paste your ANTHROPIC_API_KEY

# 4. Models (~1.4 GB, a few minutes).
./scripts/download_models.sh

# 5. Check everything.
uv run python -m jarvis
```

Step 5 prints a health check of every subsystem. Anything missing is reported in
plain language with the command that fixes it.

The first time JARVIS opens the microphone, macOS will prompt for permission. If
you miss the prompt, grant it manually under **System Settings > Privacy &
Security > Microphone** for whichever terminal you launch from.

### Installing less

The heavy dependencies are deliberately optional, because a 2 GB torch download
should not be the price of running the test suite.

| Command | What you get |
| --- | --- |
| `uv sync` | Config, logging, health check, and the Anthropic client. Text mode. |
| `uv sync --extra voice` | Adds sounddevice, openWakeWord, faster-whisper, Piper. |
| `uv sync --extra memory` | Adds sentence-transformers for semantic recall. |
| `uv sync --extra elevenlabs` | Adds the optional cloud voice. |
| `uv sync --extra full --extra dev` | Everything, plus the test toolchain. |

Without `--extra memory`, memory still works — facts are stored and retrieved,
just by keyword rather than by meaning.

## Downloading models

`./scripts/download_models.sh` fetches all of them. To fetch one at a time:

```bash
./scripts/download_models.sh wakeword     # openWakeWord, ~15 MB
./scripts/download_models.sh whisper      # faster-whisper base.en, ~140 MB
./scripts/download_models.sh piper        # en_GB-alan-medium voice, ~60 MB
./scripts/download_models.sh embeddings   # all-MiniLM-L6-v2, ~90 MB (needs --extra memory)
```

Everything lands under `models/`, which is gitignored. To use a different Piper
voice, browse [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices),
set `tts.piper.voice` in `config.yaml`, and re-run the script.

## Usage

```bash
uv run python -m jarvis                  # health check (the default)
uv run python -m jarvis health --json    # machine-readable report
uv run python -m jarvis devices          # list microphones and speakers
uv run python -m jarvis run              # start listening
uv run python -m jarvis run --text       # text REPL, no microphone needed
```

The `--text` REPL exists so you can develop and test the brain, the tools, and
memory without talking to your laptop.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Healthy, warnings allowed |
| 1 | At least one subsystem failed its check |
| 2 | `config.yaml` is missing or invalid |

## Configuration

Two files, and the split is strict:

- **`config.yaml`** — every behavioural setting, each one commented. Safe to
  commit.
- **`.env`** — API keys only. Gitignored. Never logged, never in the audit file.

Relative paths in `config.yaml` resolve against the file's own directory, so
JARVIS behaves identically no matter where you launch it from. Unknown keys are
rejected rather than ignored, so a typo is an error at startup instead of a
silently missing feature.

## Architecture

```
audio_in -> wake_word -> stt -> agent -> [tools] -> tts -> audio_out
                                  |
                                memory
```

Each stage is an abstract base class in `src/jarvis/interfaces/` with one
concrete implementation in a sibling package. No implementation imports another
implementation's internals — they communicate through the interfaces and the
dataclasses defined alongside them, and are wired together only in
`src/jarvis/cli.py`. Swapping Piper for a different synthesiser, or
faster-whisper for a different recogniser, means writing one class.

The agent is the only module that talks to Anthropic.

```
src/jarvis/
  interfaces/      abstract base classes — the contracts
  audio/           sounddevice capture and playback
  wake/            openWakeWord detector
  stt/             faster-whisper transcriber
  agent/           Anthropic client, streaming, tool dispatch
  tools/           one file per tool, registered by decorator
  tts/             Piper and ElevenLabs synthesisers
  memory/          SQLite store and embedding recall
  ui/              status window
  config.py        pydantic models for config.yaml and .env
  health.py        subsystem checks
  audit.py         append-only action log
  cli.py           argument parsing and wiring
```

## Safety

These are enforced in code, not in configuration, and cannot be turned off from
`config.yaml`:

- **`run_shell` and `write_file` always require explicit confirmation.** The
  proposed command or the full destination path is shown before you are asked.
- **A denylist blocks destructive patterns outright** — `rm -rf`, `dd`, `mkfs`,
  fork bombs, `curl | sh`, and friends. These are refused; there is no
  confirmation path that lets them through.
- **File tools cannot escape the workspace.** Paths are fully resolved, with
  symlinks followed, and checked against the real workspace path before any
  operation. String matching alone would not catch a symlink pointing outward.
- **Everything is audited.** Every tool call, confirmation, and refusal is
  appended to `logs/audit.log` as timestamped JSON.

## Development

```bash
uv run pytest                      # full suite; mocks the API and audio devices
uv run pytest -m "not hardware"    # explicitly skip anything needing real hardware
uv run ruff check src tests
uv run mypy src
```

The test suite mocks the Anthropic client and the audio devices, so it runs in
CI with no microphone, no speakers, no downloaded models, and no API key.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Skeleton: repo, deps, config, logging, health check | **Done** |
| 2 | Voice loop with no brain: wake -> record -> transcribe -> speak back | Next |
| 3 | Brain: streaming Anthropic client, speak on first sentence | |
| 4 | Tools: time, weather, web search, shell, files, timers, notes, OS control | |
| 5 | Memory: SQLite facts with embeddings, injected each turn | |
| 6 | Interrupt handling: barge-in cuts TTS immediately | |
| 7 | Polish: always-on-top status window and live transcript | |

## Licence

MIT.
