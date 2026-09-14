"""Download one model family. Called by download_models.sh.

Each library owns its own download layout, so this defers to each one rather
than hand-building URLs that would rot. The value it adds is uniform, readable
failure: a missing package, a blocked network, or a name that does not exist
should each produce one clear line and a non-zero exit, never a traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def info(message: str) -> None:
    print(f"    {message}")


def fail(message: str, remedy: str | None = None) -> int:
    print(f"{RED}    {message}{RESET}", file=sys.stderr)
    if remedy:
        print(f"    {remedy}", file=sys.stderr)
    return 1


def _network_hint(exc: Exception) -> str:
    return (
        f"Download failed: {exc}\n"
        "    Check your network connection and any proxy settings, then re-run."
    )


def wakeword(model: str, target: Path) -> int:
    try:
        import openwakeword.utils as utils
    except ImportError:
        return fail("openwakeword is not installed.", "uv sync --extra voice")

    suffix = ".onnx"
    if list(target.glob(f"{model}*{suffix}")) and (target / f"melspectrogram{suffix}").exists():
        info(f"already have {model}")
        return 0

    info(f"downloading '{model}' and its feature extractors into {target}")
    try:
        utils.download_models(model_names=[model], target_directory=str(target))
    except Exception as exc:
        return fail(_network_hint(exc))

    if not list(target.glob(f"{model}*{suffix}")):
        return fail(
            f"'{model}' is not a known openWakeWord model.",
            "Valid names: alexa, hey_mycroft, hey_jarvis, hey_rhasspy, timer, weather. "
            "Set wake_word.model in config.yaml.",
        )
    info(f"{GREEN}done{RESET}")
    return 0


def whisper(model: str, target: Path) -> int:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return fail("faster-whisper is not installed.", "uv sync --extra voice")

    if list(target.rglob("model.bin")):
        info(f"already have {model}")
        return 0

    info(f"downloading '{model}' into {target} (a few hundred MB)")
    try:
        WhisperModel(model, device="cpu", compute_type="int8", download_root=str(target))
    except Exception as exc:
        return fail(_network_hint(exc), f"Check that stt.model ('{model}') is a real model name.")
    info(f"{GREEN}done{RESET}")
    return 0


def piper(voice: str, target: Path) -> int:
    try:
        from piper.download_voices import download_voice
    except ImportError:
        return fail("piper-tts is not installed.", "uv sync --extra voice")

    if (target / f"{voice}.onnx").exists() and (target / f"{voice}.onnx.json").exists():
        info(f"already have {voice}")
        return 0

    info(f"downloading '{voice}' into {target}")
    try:
        download_voice(voice, target)
    except Exception as exc:
        return fail(
            _network_hint(exc),
            f"If '{voice}' does not exist, browse the available voices at "
            "https://huggingface.co/rhasspy/piper-voices and set tts.piper.voice.",
        )
    info(f"{GREEN}done{RESET}")
    return 0


def embeddings(model: str, target: Path) -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Optional by design: memory still works, recall just falls back to keywords.
        info("sentence-transformers is not installed - skipping.")
        info("Memory still works; recall falls back to keyword matching.")
        info("Install it with: uv sync --extra memory")
        return 0

    if target.is_dir() and any(target.iterdir()):
        info(f"already have {model}")
        return 0

    info(f"downloading '{model}' into {target}")
    try:
        SentenceTransformer(model, cache_folder=str(target))
    except Exception as exc:
        return fail(_network_hint(exc))
    info(f"{GREEN}done{RESET}")
    return 0


HANDLERS = {
    "wakeword": wakeword,
    "whisper": whisper,
    "piper": piper,
    "embeddings": embeddings,
}


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: _download.py <target> <name> <directory>", file=sys.stderr)
        return 2

    target, name, directory = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    handler = HANDLERS.get(target)
    if handler is None:
        return fail(f"unknown target '{target}'")
    return handler(name, directory)


if __name__ == "__main__":
    sys.exit(main())
