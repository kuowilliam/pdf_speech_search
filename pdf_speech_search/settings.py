from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XDG_CACHE_DIR = ROOT_DIR / ".cache" / "xdg"
DEFAULT_MPL_CACHE_DIR = ROOT_DIR / ".cache" / "matplotlib"

_model_cache_raw = os.getenv("MODEL_CACHE_DIR")
if _model_cache_raw:
    DEFAULT_MODEL_CACHE_DIR = Path(_model_cache_raw).expanduser()
    if not DEFAULT_MODEL_CACHE_DIR.is_absolute():
        DEFAULT_MODEL_CACHE_DIR = ROOT_DIR / DEFAULT_MODEL_CACHE_DIR
    DEFAULT_MODEL_CACHE_DIR = DEFAULT_MODEL_CACHE_DIR.resolve()
else:
    DEFAULT_MODEL_CACHE_DIR = ROOT_DIR / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_XDG_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CACHE_DIR))


def bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    pdf_dir: Path = path_env("PDF_DIR", ROOT_DIR / "mlsc_slide")
    index_path: Path = path_env("INDEX_PATH", ROOT_DIR / ".cache" / "pdf_index.pkl")
    model_cache_dir: Path = path_env("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    auto_build_index: bool = bool_env("AUTO_BUILD_INDEX", True)

    semantic_model: str = os.getenv("SEMANTIC_MODEL", "BAAI/bge-small-en-v1.5")
    semantic_query_prefix: str = os.getenv(
        "SEMANTIC_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    enable_reranker: bool = bool_env("ENABLE_RERANKER", True)
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "48"))
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "16"))
    model_local_files_only: bool = bool_env("MODEL_LOCAL_FILES_ONLY", True)

    asr_model_id: str = os.getenv("ASR_MODEL_ID", "nemotron-0-6b")


settings = Settings()
