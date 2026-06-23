# PDF Speech Search

Local app for English lecture captions and semantic slide-page retrieval over PDFs in `mlsc_slide/`.

## Quick Start

One-command setup and launch:

```bash
./run.sh
```

Then open:

```text
http://127.0.0.1:8000
```

If port 8000 is occupied:

```bash
PORT=8001 ./run.sh
```

What `run.sh` does:

- creates or updates the `uv` Python 3.11 environment
- downloads and warms local search and Whisper models
- rebuilds the PDF semantic index from `mlsc_slide/`
- starts the local server
- opens the app in the browser on macOS

Useful launch options:

```bash
# Start on a different port
PORT=8001 ./run.sh

# Do not open a browser automatically
OPEN_BROWSER=0 ./run.sh

# Skip rebuilding the PDF index
SKIP_INDEX=1 ./run.sh

# Faster, less accurate speech recognition
WHISPER_MODEL=base.en ./run.sh

# Slower, more accurate speech recognition
WHISPER_MODEL=medium.en WHISPER_CHUNK_SECONDS=6 WHISPER_BEAM_SIZE=5 ./run.sh
```

Stop the app with `Ctrl+C` in the terminal running `./run.sh`.

## Manual Setup

Use this only if you do not want the one-command script:

```bash
uv sync --python 3.11
uv run pdf-speech-download-models
uv run pdf-speech-index --rebuild
uv run pdf-speech-search
```

Then open:

```text
http://127.0.0.1:8000
```

The first setup downloads the local embedding model, reranker, and Whisper model.

To download and warm all models before opening the app:

```bash
uv run pdf-speech-download-models
```

To pre-download only Whisper:

```bash
uv run pdf-speech-download-models --skip-search
```

## Search

Retrieval is now local semantic search, not keyword-first search.

The index is built per PDF page and stores:

- dense embeddings from `BAAI/bge-small-en-v1.5`
- lexical TF-IDF word and character n-gram scores as fallback
- LSA scores over TF-IDF
- page text and nearby-page context for slide decks

At query time:

1. The transcript window is expanded for common ML/RL terms and ASR variants.
2. Dense semantic similarity retrieves candidate pages.
3. `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks candidate pages against the actual page text.
4. The app opens the best PDF page with `#page=<number>`.

This is meant to find the page explaining the spoken concept, not the first page where a word appears.

## Local Whisper

The default speech mode is `Local Whisper`. It does not use the browser Web Speech API.

The browser only captures microphone audio and streams 16 kHz mono PCM to the local FastAPI server. Transcription runs locally with `faster-whisper`.

`Stop` stops microphone capture only. The server keeps processing buffered audio, sends any remaining transcript, then marks transcription finished.

Low-latency default:

```text
WHISPER_MODEL=small.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CHUNK_SECONDS=3
WHISPER_BEAM_SIZE=1
```

For even faster but less accurate captions:

```bash
export WHISPER_MODEL=base.en
uv run pdf-speech-search
```

For higher accuracy but slower captions:

```bash
export WHISPER_MODEL=medium.en
export WHISPER_CHUNK_SECONDS=6
export WHISPER_BEAM_SIZE=5
uv run pdf-speech-search
```

## Configuration

Environment variables:

```text
PDF_DIR=mlsc_slide
INDEX_PATH=.cache/pdf_index.pkl
HOST=127.0.0.1
PORT=8000
AUTO_BUILD_INDEX=1

SEMANTIC_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENABLE_RERANKER=1
RERANK_CANDIDATES=48
MODEL_LOCAL_FILES_ONLY=1

WHISPER_MODEL=small.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CHUNK_SECONDS=3
WHISPER_BEAM_SIZE=1
```
