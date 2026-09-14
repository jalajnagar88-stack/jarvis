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
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS_DIR="$REPO_ROOT/models"

# Read a value out of config.yaml so this script and the app never disagree
# about which voice or model is wanted.
config_get() {
    python3 - "$1" <<'PY'
import sys, pathlib, yaml
keys = sys.argv[1].split(".")
data = yaml.safe_load(pathlib.Path("config.yaml").read_text())
for key in keys:
    data = (data or {}).get(key)
print(data if data is not None else "")
PY
}

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m    %s\033[0m\n' "$*" >&2; exit 1; }

# Run python from the project venv if there is one, so that the libraries which
# download their own models are the same ones JARVIS will import at runtime.
py() {
    if command -v uv >/dev/null 2>&1 && [ -d "$REPO_ROOT/.venv" ]; then
        uv run python "$@"
    else
        python3 "$@"
    fi
}

fetch() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        info "already have $(basename "$dest")"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    info "fetching $(basename "$dest")"
    # --fail turns an HTML error page into a non-zero exit instead of a
    # corrupt "model" file that fails confusingly at load time.
    if ! curl --fail --location --progress-bar --output "$dest.partial" "$url"; then
        rm -f "$dest.partial"
        die "download failed: $url
    Check your network, then re-run. If the URL 404s, the voice name in
    config.yaml may not exist — browse https://huggingface.co/rhasspy/piper-voices"
    fi
    mv "$dest.partial" "$dest"
}

# --------------------------------------------------------------------------- #

download_wakeword() {
    say "openWakeWord models"
    local target="$MODELS_DIR/openwakeword"
    mkdir -p "$target"
    # openWakeWord ships its own downloader and knows which shared feature
    # extractors each wake model needs, so let it do the work.
    py - "$target" <<'PY'
import sys
from pathlib import Path

target = Path(sys.argv[1])
try:
    import openwakeword.utils as utils
except ImportError:
    sys.exit("openwakeword is not installed. Run: uv sync --extra voice")

print(f"    downloading pretrained models into {target}")
utils.download_models(target_directory=str(target))
onnx = sorted(p.name for p in target.glob("*.onnx"))
print(f"    {len(onnx)} model file(s): {', '.join(onnx) or 'none'}")
PY
}

download_whisper() {
    say "faster-whisper model"
    local model target
    model="$(config_get stt.model)"; model="${model:-base.en}"
    target="$MODELS_DIR/whisper"
    mkdir -p "$target"
    py - "$model" "$target" <<'PY'
import sys
model_name, target = sys.argv[1], sys.argv[2]
try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("faster-whisper is not installed. Run: uv sync --extra voice")

print(f"    downloading '{model_name}' into {target}")
# Instantiating the model downloads and converts it; we discard it immediately.
WhisperModel(model_name, device="cpu", compute_type="int8", download_root=target)
print("    done")
PY
}

download_piper() {
    say "Piper voice"
    local voice target
    voice="$(config_get tts.piper.voice)"; voice="${voice:-en_GB-alan-medium}"
    target="$MODELS_DIR/piper"

    # Voice names are <locale>-<name>-<quality>, and the HuggingFace repo is
    # laid out as <lang>/<locale>/<name>/<quality>/.
    local locale name quality lang base
    locale="${voice%%-*}"
    quality="${voice##*-}"
    name="${voice#*-}"; name="${name%-*}"
    lang="${locale%%_*}"
    base="https://huggingface.co/rhasspy/piper-voices/resolve/main/${lang}/${locale}/${name}/${quality}"

    info "voice: $voice"
    fetch "$base/$voice.onnx"      "$target/$voice.onnx"
    fetch "$base/$voice.onnx.json" "$target/$voice.onnx.json"
}

download_embeddings() {
    say "sentence-transformers embedding model"
    local model target
    model="$(config_get memory.embedding_model)"; model="${model:-all-MiniLM-L6-v2}"
    target="$MODELS_DIR/embeddings"
    mkdir -p "$target"
    py - "$model" "$target" <<'PY'
import sys
model_name, target = sys.argv[1], sys.argv[2]
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("    sentence-transformers is not installed — skipping.")
    print("    Memory still works; recall falls back to keyword matching.")
    print("    Install it with: uv sync --extra memory")
    raise SystemExit(0)

print(f"    downloading '{model_name}' into {target}")
SentenceTransformer(model_name, cache_folder=target)
print("    done")
PY
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
