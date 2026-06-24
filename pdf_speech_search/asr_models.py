from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from pdf_speech_search.settings import settings


Engine = Literal["nemotron"]


@dataclass(frozen=True)
class AsrModelSpec:
    id: str
    label: str
    detail: str
    engine: Engine
    model_name: str
    repo_id: str
    chunk_seconds: float
    beam_size: int = 1
    default: bool = False


ASR_MODELS: tuple[AsrModelSpec, ...] = (
    AsrModelSpec(
        id="nemotron-0-6b",
        label="NVIDIA Nemotron",
        detail="Local CPU ASR",
        engine="nemotron",
        model_name="nvidia/nemotron-speech-streaming-en-0.6b",
        repo_id="nvidia/nemotron-speech-streaming-en-0.6b",
        chunk_seconds=5.0,
        beam_size=1,
        default=True,
    ),
)


def get_asr_model(model_id: str | None) -> AsrModelSpec:
    normalized = model_id or settings.asr_model_id
    aliases = {
        "nemotron": "nemotron-0-6b",
    }
    normalized = aliases.get(normalized, normalized)
    for spec in ASR_MODELS:
        if spec.id == normalized:
            return spec
    raise KeyError(normalized)


def default_asr_model() -> AsrModelSpec:
    try:
        return get_asr_model(settings.asr_model_id)
    except KeyError:
        return next(spec for spec in ASR_MODELS if spec.default)


def cache_roots() -> list[Path]:
    roots = [
        settings.model_cache_dir,
        Path.home() / ".cache" / "huggingface",
    ]
    seen: set[Path] = set()
    result: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def repo_cache_dir(repo_id: str, root: Path) -> Path:
    return root / "hub" / f"models--{repo_id.replace('/', '--')}"


def hf_repo_cached(repo_id: str) -> bool:
    for root in cache_roots():
        snapshots = repo_cache_dir(repo_id, root) / "snapshots"
        if snapshots.exists() and any(path.is_dir() for path in snapshots.iterdir()):
            return True
    return False


def find_cached_nemo(model_name: str) -> Path | None:
    model_path = Path(model_name).expanduser()
    if model_path.exists() and model_path.suffix == ".nemo":
        return model_path.resolve()
    if "/" not in model_name:
        return None

    checkpoint_name = f"{model_name.split('/', 1)[1]}.nemo"
    for root in cache_roots():
        hub_dir = repo_cache_dir(model_name, root)
        if not hub_dir.exists():
            continue
        matches = sorted(
            hub_dir.glob(f"snapshots/*/{checkpoint_name}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            matches = sorted(
                hub_dir.glob("snapshots/**/*.nemo"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        if matches:
            return matches[0].resolve()
    return None


def runtime_available(spec: AsrModelSpec) -> tuple[bool, str | None]:
    modules = ["nemo", "soundfile", "torch", "huggingface_hub"]
    missing = [module for module in modules if find_spec(module) is None]
    if missing:
        return False, f"Missing Python package: {', '.join(missing)}"
    return True, None


def model_installed(spec: AsrModelSpec) -> bool:
    return find_cached_nemo(spec.model_name) is not None


def download_model(spec: AsrModelSpec) -> None:
    if model_installed(spec):
        return

    from huggingface_hub import hf_hub_download

    checkpoint_name = f"{spec.model_name.split('/', 1)[1]}.nemo"
    hf_hub_download(
        repo_id=spec.repo_id,
        filename=checkpoint_name,
    )


def model_status(spec: AsrModelSpec) -> dict[str, object]:
    runtime_ok, runtime_reason = runtime_available(spec)
    installed = model_installed(spec)
    return {
        "id": spec.id,
        "label": spec.label,
        "detail": spec.detail,
        "engine": spec.engine,
        "model": spec.model_name,
        "repo_id": spec.repo_id,
        "chunk_seconds": spec.chunk_seconds,
        "beam_size": spec.beam_size,
        "default": spec.default,
        "installed": installed,
        "runtime_available": runtime_ok,
        "available": installed and runtime_ok,
        "reason": None if runtime_ok else runtime_reason,
    }
