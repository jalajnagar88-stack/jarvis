#!/usr/bin/env bash
#
# Download the models JARVIS needs. Everything lands under ./models/, which is
# gitignored. Safe to re-run: existing files are left alone.
#
#   ./scripts/download_models.sh                 # everything
#   ./scripts/download_models.sh piper whisper   # just these
#
# Targets: wakeword whisper piper embeddings
#
# Which model, voice and Whisper size are fetched is read out of config.yaml, so
# this script and the application can never disagree about what is wanted.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS_DIR="$REPO_ROOT/models"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m    %s\033[0m\n' "$*" >&2; exit 1; }

# Run python from the project venv when there is one, so the libraries that
# download their own models are the same ones JARVIS imports at runtime.
py() {
    if command -v uv >/dev/null 2>&1 && [ -d "$REPO_ROOT/.venv" ]; then
        uv run --quiet python "$@"
    else
        python3 "$@"
    fi
}

config_get() {
    py "$REPO_ROOT/scripts/_config_get.py" "$1"
}

# --------------------------------------------------------------------------- #

download_wakeword() {
    say "openWakeWord model"
    local model target
    model="$(config_get wake_word.model)"; model="${model:-hey_jarvis}"
    target="$MODELS_DIR/openwakeword"
    mkdir -p "$target"
    info "wake word: $model"
    # openWakeWord knows which shared feature extractors each wake model needs,
    # so let its own downloader do the work rather than guessing URLs.
    py "$REPO_ROOT/scripts/_download.py" wakeword "$model" "$target"
}

download_whisper() {
    say "faster-whisper model"
    local model target
    model="$(config_get stt.model)"; model="${model:-base.en}"
    target="$MODELS_DIR/whisper"
    mkdir -p "$target"
    info "model: $model"
    py "$REPO_ROOT/scripts/_download.py" whisper "$model" "$target"
}

download_piper() {
    say "Piper voice"
    local voice target
    voice="$(config_get tts.piper.voice)"; voice="${voice:-en_GB-alan-medium}"
    target="$MODELS_DIR/piper"
    mkdir -p "$target"
    info "voice: $voice"
    py "$REPO_ROOT/scripts/_download.py" piper "$voice" "$target"
}

download_embeddings() {
    say "sentence-transformers embedding model"
    local model target
    model="$(config_get memory.embedding_model)"; model="${model:-all-MiniLM-L6-v2}"
    target="$MODELS_DIR/embeddings"
    mkdir -p "$target"
    info "model: $model"
    py "$REPO_ROOT/scripts/_download.py" embeddings "$model" "$target"
}

# --------------------------------------------------------------------------- #

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    targets=(wakeword whisper piper embeddings)
fi

for target in "${targets[@]}"; do
    case "$target" in
        wakeword)   download_wakeword ;;
        whisper)    download_whisper ;;
        piper)      download_piper ;;
        embeddings) download_embeddings ;;
        *)          die "unknown target '$target' (want: wakeword whisper piper embeddings)" ;;
    esac
done

say "Done"
info "Verify with: uv run python -m jarvis"
