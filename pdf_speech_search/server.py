from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pdf_speech_search.asr_models import (
    ASR_MODELS,
    default_asr_model,
    download_model,
    get_asr_model,
    model_installed,
    model_status,
)
from pdf_speech_search.indexing import (
    PdfIndex,
    index_is_current,
    load_or_build_index,
    search_index,
)
from pdf_speech_search.settings import ROOT_DIR, settings
from pdf_speech_search.stt import nemotron_local, whisper_local


STATIC_DIR = Path(__file__).resolve().parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RebuildResponse(BaseModel):
    pages: int
    pdfs: int
    index_path: str


app = FastAPI(title="PDF Speech Search")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_index_lock = asyncio.Lock()
_index: PdfIndex | None = None
_download_lock = asyncio.Lock()
_download_jobs: dict[str, dict[str, Any]] = {}


async def get_index(force: bool = False) -> PdfIndex:
    global _index
    async with _index_lock:
        if force or _index is None or not index_is_current(_index, settings.pdf_dir):
            _index = await asyncio.to_thread(
                load_or_build_index,
                settings.pdf_dir,
                settings.index_path,
                force,
            )
        return _index


def asr_model_payload() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for spec in ASR_MODELS:
        payload = model_status(spec)
        job = _download_jobs.get(spec.id)
        if job:
            payload["download_status"] = job["status"]
            payload["download_message"] = job.get("message")
        else:
            payload["download_status"] = "ready" if payload["available"] else "missing"
            payload["download_message"] = None
        models.append(payload)
    return {
        "default_model_id": default_asr_model().id,
        "models": models,
    }


async def run_download_job(model_id: str) -> None:
    spec = get_asr_model(model_id)
    job = _download_jobs[model_id]
    try:
        job["message"] = f"Downloading {spec.label}"
        await asyncio.to_thread(download_model, spec)
        job["status"] = "ready"
        job["message"] = f"{spec.label} is ready"
    except Exception as exc:  # pragma: no cover - network/model runtime dependent
        job["status"] = "error"
        job["message"] = str(exc)


@app.on_event("startup")
async def startup() -> None:
    if settings.auto_build_index:
        await get_index(force=False)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    index = await get_index(force=False)
    return {
        "pdf_dir": str(settings.pdf_dir),
        "index_path": str(settings.index_path),
        "index_current": index_is_current(index, settings.pdf_dir),
        "pages": len(index.pages),
        "pdfs": len(index.doc_map),
        "built_at": index.built_at,
        "search": {
            "semantic_model": index.semantic_model_name,
            "reranker_enabled": settings.enable_reranker,
            "reranker_model": settings.reranker_model if settings.enable_reranker else None,
        },
        "asr": asr_model_payload(),
    }


@app.get("/api/asr/models")
async def api_asr_models() -> dict[str, Any]:
    return asr_model_payload()


@app.post("/api/asr/models/{model_id}/download")
async def download_asr_model(model_id: str) -> dict[str, Any]:
    try:
        spec = get_asr_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown ASR model") from None

    if model_installed(spec):
        _download_jobs[model_id] = {
            "status": "ready",
            "message": f"{spec.label} is ready",
        }
        return asr_model_payload()

    async with _download_lock:
        job = _download_jobs.get(model_id)
        if job and job.get("status") == "downloading":
            return asr_model_payload()
        _download_jobs[model_id] = {
            "status": "downloading",
            "message": f"Downloading {spec.label}",
        }
        asyncio.create_task(run_download_job(model_id))
    return asr_model_payload()


@app.post("/api/index/rebuild", response_model=RebuildResponse)
async def rebuild_index() -> RebuildResponse:
    index = await get_index(force=True)
    return RebuildResponse(
        pages=len(index.pages),
        pdfs=len(index.doc_map),
        index_path=str(settings.index_path),
    )


@app.post("/api/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    index = await get_index(force=False)
    results = await asyncio.to_thread(search_index, index, request.query, request.top_k)
    return {"query": request.query, "results": results}


@app.get("/api/search")
async def search_get(q: str, top_k: int = 5) -> dict[str, Any]:
    return await search(SearchRequest(query=q, top_k=top_k))


@app.get("/pdf/{doc_id}")
async def pdf(doc_id: str) -> FileResponse:
    index = await get_index(force=False)
    path_str = index.doc_map.get(doc_id)
    if path_str is None:
        raise HTTPException(status_code=404, detail="Unknown PDF")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF file no longer exists")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        content_disposition_type="inline",
    )


@app.websocket("/ws/asr/{model_id}")
async def ws_asr(websocket: WebSocket, model_id: str) -> None:
    await websocket.accept()
    try:
        spec = get_asr_model(model_id)
    except KeyError:
        await websocket.send_json({"type": "error", "message": "Unknown ASR model"})
        return

    try:
        if spec.engine == "whisper":
            await whisper_local.stream_websocket(websocket, spec)
        else:
            await nemotron_local.stream_websocket(websocket, spec)
    except WebSocketDisconnect:
        return


def main() -> None:
    uvicorn.run(
        "pdf_speech_search.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        app_dir=str(ROOT_DIR),
    )


if __name__ == "__main__":
    main()
