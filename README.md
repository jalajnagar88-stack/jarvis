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
./scripts/download_models.sh wakeword     # openWakeWord, ~10 MB
./scripts/download_models.sh whisper      # faster-whisper base.en, ~140 MB
./scripts/download_models.sh piper        # en_GB-alan-medium voice, ~60 MB
./scripts/download_models.sh embeddings   # all-MiniLM-L6-v2, ~90 MB (needs --extra memory)
```

Everything lands under `models/`, which is gitignored. The script reads
`config.yaml` to decide *which* wake word, Whisper size, and voice to fetch, so
the two can never disagree: change the config, re-run the script.

To use a different Piper voice, browse
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices), set
`tts.piper.voice` in `config.yaml`, and re-run. Re-running is always safe --
anything already downloaded is left alone.

## Usage

```bash
uv run python -m jarvis                      # health check (the default)
uv run python -m jarvis health --json        # machine-readable report
uv run python -m jarvis devices              # list microphones and speakers
uv run python -m jarvis run                  # start listening for the wake word
uv run python -m jarvis run --text           # text REPL, no microphone needed
uv run python -m jarvis run --no-brain       # echo mode; no API key needed
uv run python -m jarvis say 'Good evening.'  # audition the voice
```

### Talking to it

Run `python -m jarvis run`, say **"hey Jarvis"**, wait for `[listening]`, then
speak. It records until you stop talking, transcribes locally, sends the text to
Claude, and starts speaking the reply as soon as the first sentence is ready.

The first time it opens the microphone, macOS will ask for permission.

### Why it starts talking before it has finished thinking

The reply is streamed. As each sentence completes it is handed straight to
speech, so JARVIS begins talking roughly when the *first* sentence lands rather
than when the last one does. On a three-sentence answer that is the difference
between a half-second pause and a three-second one, which is most of what makes
an assistant feel present rather than sluggish.

Sentence boundaries are found on the stream, which means not splitting after
"Dr.", "3 p.m." or the "3." of "3.14" -- cutting a sentence in the wrong place
is more jarring than waiting a moment longer.

### Echo mode

`run --no-brain` skips Claude entirely and repeats what you said. It needs no
API key and costs nothing. Use it to check the audio path in isolation: if echo
sounds right but real replies do not, the problem is reasoning, not microphones.

### What a conversation costs

Every turn is one API call. With the default `claude-sonnet-5` and the short
replies the system prompt asks for, a typical exchange is well under a penny,
but it is not free and there is no local fallback for the thinking step. To
change model or spend, edit the `llm` section of `config.yaml`:

- `model` -- `claude-haiku-4-5` is cheaper and faster; `claude-opus-5` is better
  at multi-step reasoning but noticeably slower to first word.
- `effort` -- `low` by default, because you are standing there waiting. Raise it
  if replies feel careless.
- `history_turns` -- how much conversation is re-sent each turn. Lower it to
  spend less; raise it if JARVIS forgets what you just said.

### Working without a microphone

`run --text` gives you the same pipeline with the keyboard standing in for the
microphone. Replies are still spoken aloud, so it is also the quickest way to
audition a voice. If no speaker is available either, it degrades to printing
rather than failing.

### Watching what it is doing

```bash
uv run python -m jarvis run --ui tui      # a panel in the terminal
uv run python -m jarvis run --ui window   # a small always-on-top window
```

Both show the current state — idle, listening, thinking, speaking — and a live
transcript, including the reply as it streams in and a note when a tool runs.
The terminal panel is the more portable of the two: it works over SSH and needs
nothing installed. Set `ui.mode` in `config.yaml` to make either the default.

Neither is on by default. This is a voice interface; a window is opt-in.

### Interrupting it

Start talking while JARVIS is speaking and it stops mid-word and listens. You do
not need the wake word again — it goes straight back to recording, because you
are already talking.

The hard part is that JARVIS's own voice comes out of the speakers and back into
the microphone, so a plain level threshold would cut it off every time. Three
things prevent that: interruption is judged against a higher threshold than
ordinary speech, the level has to hold for `min_duration_seconds` rather than a
single loud block, and the first `grace_seconds` of each reply are ignored
entirely — the opening of a sentence is the loudest part of its own output.

This is not echo cancellation. At a sensible speaker volume it works, and it
fails in the safe direction: a missed interruption means waiting for the sentence
to finish, not JARVIS talking over itself indefinitely. If it cuts itself off,
raise `interrupt.threshold`; if it ignores you, lower it.

### Tuning the listening

These live under `audio.silence` in `config.yaml`. The defaults suit a quiet
room; a noisy one usually needs the threshold raised.

| Symptom | Setting | Direction |
| --- | --- | --- |
| Cuts you off mid-sentence | `duration_seconds` | up |
| Waits too long after you finish | `duration_seconds` | down |
| Records constantly in a noisy room | `threshold` | up |
| Misses quiet speech | `threshold` | down |
| Triggers on background noise | `wake_word.threshold` | up (try 0.6-0.7) |
| Misses the wake word | `wake_word.threshold` | down |
| Clips the first word of your command | `preroll_seconds` | up |
| Cuts itself off while speaking | `interrupt.threshold` | up |
| Ignores you when you talk over it | `interrupt.threshold` | down |

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

The agent is the only module that talks to Anthropic. It hands the rest of the
system a stream of events -- text deltas, completed sentences, a finished turn,
a failure -- so the voice loop, the text REPL and (later) the status window all
consume the same thing.

A turn is a generator, which is how interruption will work in milestone 6: the
loop stops consuming it and the HTTP stream tears itself down. There is no
cancellation flag to get wrong.

```
src/jarvis/
  interfaces/      abstract base classes — the contracts
  audio/           sounddevice capture and playback
  wake/            openWakeWord detector
  stt/             faster-whisper transcriber
  agent/           Anthropic client, streaming, the system prompt, echo brain
  tools/           one file per tool, registered by decorator
  tts/             Piper and ElevenLabs synthesisers
  memory/          SQLite store and embedding recall
  ui/              status window
  config.py        pydantic models for config.yaml and .env
  health.py        subsystem checks
  audit.py         append-only action log
  cli.py           argument parsing and wiring
```

## Tools

JARVIS can tell the time, check the weather (Open-Meteo, no key), search the web,
read and write files in a sandboxed workspace, run shell commands, set timers,
take notes, open applications, and change the volume. Which of those are offered
is decided by `tools.enabled` in `config.yaml` — remove a name and the tool never
appears in the schema at all.

Adding one is a single file under `src/jarvis/tools/builtin/`: a pydantic model
for the arguments, a function, a decorator. The JSON schema is generated from
the model, so it cannot drift from the code that reads it.

## Memory

JARVIS keeps two things in a local SQLite file: an append-only transcript, and
short durable facts about you. After each exchange a cheap background call
decides whether anything is worth keeping — preferences, names, standing
instructions — and stores it with an embedding. On each turn the most relevant
few are retrieved and put into the system prompt.

You can correct it out loud: "remember that I prefer metric" and "forget that my
sister is called Priya" both work. To see everything it has:

```bash
uv run python -m jarvis facts              # list what it remembers
uv run python -m jarvis facts --forget-all # delete the lot (asks first)
```

Memory that cannot be inspected is memory you cannot trust, which is why that is
a command rather than an invitation to open SQLite by hand.

Without `--extra memory` it still works, retrieving by keyword instead of by
meaning. Extraction is deliberately conservative — a memory full of "asked about
the weather" is worse than an empty one, because it crowds out what matters and
makes every later recall noisier. Forgetting is stricter still: an ambiguous
request removes nothing, since you can repeat a fact it failed to forget but
cannot recover one it forgot by mistake.

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

The denylist and the confirmation gate are deliberately separate. Confirmation
protects against JARVIS doing something you did not intend; the denylist
protects against a spoken "yes" that was a misheard word, a television in the
background, or a sentence that happened to contain "sure". Anything that could
destroy a machine is simply not reachable through this program.

Spoken confirmation is strict in the same direction: anything that is not a
clear yes is a no, including silence, and "yes, don't do that" is a refusal.

## Development

```bash
uv run pytest                      # full suite; mocks the API and audio devices
uv run pytest -m "not hardware"    # explicitly skip anything needing real hardware
uv run pytest --cov                # with coverage
uv run ruff check src tests
uv run ruff format src tests
uv run mypy src tests
```

The suite runs with no microphone, no speakers, no downloaded models, and no API
key. That is enforced rather than hoped for: PortAudio is replaced with a fake
module whose audio callbacks the tests drive by hand, so the real queueing,
mixing and cancellation code is genuinely executed rather than mocked past.

Tests marked `hardware` run against real downloaded models and skip themselves
when those models are absent.

### When the API will not cooperate

Every failure mode has its own spoken sentence, because "sorry, something went
wrong" tells you nothing about whether to try again. A rate limit says to wait a
moment; a missing key says so and prints the fix; a dropped connection says it
could not reach the network. Anything already spoken and audited is logged at
debug rather than shouted into the middle of the conversation.

A failed turn is not written to the conversation history, so a dangling question
with no answer cannot poison the next request.

### Swapping an implementation

Every stage is an interface in `src/jarvis/interfaces/` with its implementation
chosen in `src/jarvis/factory.py`. To replace one -- a different recogniser, a
different voice engine -- write a class against the interface and change one
function there. Nothing else needs to know.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Skeleton: repo, deps, config, logging, health check | **Done** |
| 2 | Voice loop with no brain: wake -> record -> transcribe -> speak back | **Done** |
| 3 | Brain: streaming Anthropic client, speak on first sentence | **Done** |
| 4 | Tools: time, weather, web search, shell, files, timers, notes, OS control | **Done** |
| 5 | Memory: SQLite facts with embeddings, injected each turn | **Done** |
| 6 | Interrupt handling: barge-in cuts TTS immediately | **Done** |
| 7 | Polish: always-on-top status window and live transcript | **Done** |

All seven are built. What has *not* been exercised is the physical layer: no
microphone, speaker, or GPU exists in the environment this was written in, and
the Anthropic API was mocked throughout rather than called. The test suite
covers the logic around all of it; you are the first to run it against real
hardware and a real key.

## Licence

MIT.
