#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
URL="http://${HOST}:${PORT}"

export ASR_MODEL_ID="${ASR_MODEL_ID:-whisper-small-en}"
export WHISPER_MODEL="${WHISPER_MODEL:-small.en}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cpu}"
export WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8}"
export WHISPER_CHUNK_SECONDS="${WHISPER_CHUNK_SECONDS:-3}"
export WHISPER_BEAM_SIZE="${WHISPER_BEAM_SIZE:-1}"
export MODEL_LOCAL_FILES_ONLY="${MODEL_LOCAL_FILES_ONLY:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found."
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "==> Syncing Python environment"
uv sync --python 3.11 --inexact

echo "==> Downloading and warming local models for ${ASR_MODEL_ID}"
MODEL_LOCAL_FILES_ONLY=0 uv run pdf-speech-download-models --asr-model "$ASR_MODEL_ID"

if [[ "${SKIP_INDEX:-0}" != "1" ]]; then
  echo "==> Building PDF semantic index"
  uv run pdf-speech-index --rebuild
else
  echo "==> Skipping index rebuild because SKIP_INDEX=1"
fi

if command -v lsof >/dev/null 2>&1; then
  EXISTING_PID="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
  if [[ -n "$EXISTING_PID" ]]; then
    echo "Port ${PORT} is already in use by PID(s): ${EXISTING_PID}"
    echo "Stop that process first, or run with another port:"
    echo "  PORT=8001 ./run.sh"
    exit 1
  fi
fi

echo "==> Starting app at ${URL}"
HOST="$HOST" PORT="$PORT" uv run pdf-speech-search &
SERVER_PID="$!"

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "==> Waiting for server"
for _ in $(seq 1 60); do
  if curl -fsS "${URL}/api/status" >/dev/null 2>&1; then
    echo "==> Ready: ${URL}"
    if [[ "${OPEN_BROWSER:-1}" != "0" ]] && command -v open >/dev/null 2>&1; then
      open "$URL" >/dev/null 2>&1 || true
    fi
    wait "$SERVER_PID"
    exit $?
  fi

  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Server exited before becoming ready."
    wait "$SERVER_PID"
    exit $?
  fi

  sleep 1
done

echo "Timed out waiting for ${URL}"
exit 1
