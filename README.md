# PDF Speech Search

A local app that transcribes speech in real time and quickly finds the matching PDF slide page based on what you say.

Put your PDFs in `mlsc_slide/`, start the app, speak into the microphone, and it will caption your speech and jump to the most relevant slide.

## Getting Started

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
./run.sh
```

Open in your browser: **http://127.0.0.1:8000**

The first run downloads the NVIDIA Nemotron ASR checkpoint and builds the PDF index, which may take a few minutes. Later starts reuse the local cache.

Stop the app with `Ctrl+C` in the terminal running `./run.sh`.

### Common Options

```bash
PORT=8001 ./run.sh              # Use a different port
OPEN_BROWSER=0 ./run.sh         # Do not open the browser automatically
SKIP_INDEX=1 ./run.sh           # Skip rebuilding the PDF index
FORCE_INDEX=1 ./run.sh          # Force rebuild even if PDFs did not change
```

## Supported Models

### Speech Recognition (ASR)

| Model | Notes |
|-------|-------|
| **NVIDIA Nemotron** | Local CPU ASR, ~2.4 GB `.nemo` checkpoint |

`./run.sh` uses `ASR_MODEL_ID=nemotron-0-6b` by default. If the checkpoint already exists in `.cache/huggingface/`, startup skips the download.

Press the microphone button to start capture. Press it again to stop microphone capture; transcription continues on the already-buffered audio and finishes when the server sends the final text.

### PDF Search

| Model | Purpose |
|-------|---------|
| `BAAI/bge-small-en-v1.5` | Semantic embeddings to find relevant slide pages |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking to improve search accuracy |

## Data Storage

Models and the index are stored in `.cache/` inside the project (not committed to git):

- `.cache/huggingface/` — speech and search models
- `.cache/pdf_index.pkl` — PDF index

Keep real lecture slides in `mlsc_slide/` locally. The repo only includes `mock.pdf` as a demo.
