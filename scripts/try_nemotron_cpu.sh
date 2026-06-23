#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found."
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if command -v brew >/dev/null 2>&1; then
  if ! brew list libsndfile >/dev/null 2>&1; then
    echo "Note: if soundfile/audio loading fails on macOS, run:"
    echo "  brew install libsndfile ffmpeg"
  fi
fi

echo "==> Syncing base project environment"
uv sync --python 3.11

echo "==> Installing local Nemotron CPU dependencies into .venv"
uv pip install Cython packaging soundfile torchaudio

if [[ "${NEMOTRON_NEMO_SOURCE:-pypi}" == "github" ]]; then
  uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
else
  uv pip install "nemo_toolkit[asr]"
fi

echo "==> Running Nemotron CPU test"
".venv/bin/python" scripts/try_nemotron_cpu.py "$@"
