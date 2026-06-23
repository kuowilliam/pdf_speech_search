from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import wave
from typing import Any

from fastapi import WebSocket

from pdf_speech_search.asr_models import AsrModelSpec, find_cached_nemo


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

_models: dict[str, Any] = {}
_model_lock = threading.Lock()


def coerce_transcript(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return str(result.text)
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def pcm_to_wav_path(raw_audio: bytes) -> str:
    handle = tempfile.NamedTemporaryFile(prefix="nemotron-", suffix=".wav", delete=False)
    handle.close()
    with wave.open(handle.name, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(raw_audio)
    return handle.name


def get_model(spec: AsrModelSpec) -> Any:
    cached_nemo = find_cached_nemo(spec.model_name)
    if cached_nemo is None:
        raise FileNotFoundError(f"{spec.label} has not been downloaded")

    key = str(cached_nemo)
    if key not in _models:
        with _model_lock:
            if key not in _models:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                import nemo.collections.asr as nemo_asr

                model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=key,
                    map_location="cpu",
                )
                model = model.to("cpu")
                model.eval()
                _models[key] = model
    return _models[key]


def transcribe_chunk(raw_audio: bytes, spec: AsrModelSpec) -> str:
    if len(raw_audio) < int(0.5 * SAMPLE_RATE * BYTES_PER_SAMPLE):
        return ""

    wav_path = pcm_to_wav_path(raw_audio)
    try:
        model = get_model(spec)
        outputs = model.transcribe([wav_path], batch_size=1)
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass

    if not outputs:
        return ""
    return " ".join(coerce_transcript(outputs[0]).split())


async def stream_websocket(websocket: WebSocket, spec: AsrModelSpec) -> None:
    await websocket.send_json({"type": "loading", "message": f"Loading {spec.label}"})
    try:
        await asyncio.to_thread(get_model, spec)
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        await websocket.send_json({"type": "error", "message": f"Nemotron model load failed: {exc}"})
        return

    await websocket.send_json(
        {
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "model": spec.label,
            "chunk_seconds": spec.chunk_seconds,
        }
    )

    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=250)
    stop_event = asyncio.Event()
    chunk_bytes = int(spec.chunk_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
    min_flush_bytes = int(1.0 * SAMPLE_RATE * BYTES_PER_SAMPLE)

    async def receive_audio() -> None:
        try:
            while not stop_event.is_set():
                message = await websocket.receive()
                if "bytes" in message and message["bytes"] is not None:
                    try:
                        audio_queue.put_nowait(message["bytes"])
                    except asyncio.QueueFull:
                        pass
                elif message.get("text") == "stop":
                    break
        finally:
            stop_event.set()
            try:
                audio_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def process_audio() -> None:
        buffer = bytearray()

        async def flush(force: bool = False) -> None:
            if len(buffer) < min_flush_bytes:
                return
            if not force and len(buffer) < chunk_bytes:
                return
            raw = bytes(buffer)
            buffer.clear()
            try:
                text = await asyncio.to_thread(transcribe_chunk, raw, spec)
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                await websocket.send_json({"type": "error", "message": f"Nemotron failed: {exc}"})
                stop_event.set()
                return
            if text:
                await websocket.send_json({"type": "transcript", "text": text, "final": True})

        while not stop_event.is_set() or not audio_queue.empty():
            chunk = await audio_queue.get()
            if chunk is None:
                break
            buffer.extend(chunk)
            await flush(force=False)

        await flush(force=True)
        await websocket.send_json({"type": "done"})

    receive_task = asyncio.create_task(receive_audio())
    process_task = asyncio.create_task(process_audio())
    done, pending = await asyncio.wait(
        {receive_task, process_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    stop_event.set()
    try:
        audio_queue.put_nowait(None)
    except asyncio.QueueFull:
        pass
    if process_task in done:
        receive_task.cancel()
    else:
        await process_task
        receive_task.cancel()
    await asyncio.gather(receive_task, process_task, return_exceptions=True)
