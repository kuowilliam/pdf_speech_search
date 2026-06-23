from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load local Nemotron ASR with NeMo on CPU and optionally transcribe an audio file."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("NEMOTRON_MODEL", DEFAULT_MODEL),
        help=f"Hugging Face model id or local .nemo path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Explicit local .nemo checkpoint path. This avoids Hugging Face metadata calls.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional WAV/audio file to transcribe. If omitted, the script only loads the model.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only cached Hugging Face files. Fails if the model has not been downloaded.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=int(os.getenv("NEMOTRON_TORCH_THREADS", "0")),
        help="Optional torch CPU thread count.",
    )
    return parser.parse_args()


def coerce_transcript(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return str(result.text)
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def candidate_cache_roots() -> list[Path]:
    roots: list[Path] = []
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser())
    roots.extend(
        [
            ROOT_DIR / ".cache" / "huggingface",
            Path.home() / ".cache" / "huggingface",
        ]
    )
    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def find_cached_nemo(model: str) -> Path | None:
    model_path = Path(model).expanduser()
    if model_path.exists() and model_path.suffix == ".nemo":
        return model_path.resolve()

    if "/" not in model:
        return None

    owner, name = model.split("/", 1)
    cache_dir_name = f"models--{owner}--{name}"
    checkpoint_name = f"{name}.nemo"

    for root in candidate_cache_roots():
        hub_dir = root / "hub" / cache_dir_name
        if not hub_dir.exists():
            continue
        snapshots = sorted(
            hub_dir.glob(f"snapshots/*/{checkpoint_name}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return snapshots[0].resolve()

        any_nemo = sorted(
            hub_dir.glob("snapshots/**/*.nemo"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if any_nemo:
            return any_nemo[0].resolve()

    return None


def main() -> int:
    args = parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except Exception as exc:
        print(f"Missing Nemotron local dependencies: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Run: ./scripts/try_nemotron_cpu.sh", file=sys.stderr)
        return 2

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    if args.audio and not args.audio.exists():
        print(f"Audio file does not exist: {args.audio}", file=sys.stderr)
        return 2

    local_nemo = args.model_path.resolve() if args.model_path else find_cached_nemo(args.model)
    if args.model_path and not args.model_path.exists():
        print(f"Local model path does not exist: {args.model_path}", file=sys.stderr)
        return 2

    print(f"Loading Nemotron ASR model on CPU: {local_nemo or args.model}")
    started = time.time()
    if local_nemo:
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(local_nemo),
            map_location="cpu",
        )
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=args.model,
            map_location="cpu",
        )
    model = model.to("cpu")
    model.eval()
    print(f"Loaded in {time.time() - started:.1f}s")

    if args.audio is None:
        print("No --audio provided. Model load test passed.")
        return 0

    print(f"Transcribing: {args.audio}")
    started = time.time()
    outputs = model.transcribe([str(args.audio)], batch_size=1)
    elapsed = time.time() - started

    if not outputs:
        print("No transcript returned.", file=sys.stderr)
        return 1

    print("\nTranscript:")
    print(coerce_transcript(outputs[0]).strip())
    print(f"\nTranscribed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
