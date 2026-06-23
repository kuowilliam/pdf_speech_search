from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import WebSocket

from pdf_speech_search.asr_models import AsrModelSpec
from pdf_speech_search.settings import settings


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

_models: dict[tuple[str, str, str], Any] = {}
_model_lock = threading.Lock()


@dataclass(frozen=True)
class WhisperStatus:
    configured: bool
    model: str
    device: str
    compute_type: str
    chunk_seconds: float
    reason: str | None = None


def status() -> WhisperStatus:
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:  # pragma: no cover - import depends on optional runtime wheels
        return WhisperStatus(
            False,
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
            settings.whisper_chunk_seconds,
            f"faster-whisper is unavailable: {exc}",
        )
    return WhisperStatus(
        True,
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
        settings.whisper_chunk_seconds,
    )


def get_model() -> Any:
    spec = AsrModelSpec(
        id="legacy-whisper",
        label="Whisper",
        detail="",
        engine="whisper",
        model_name=settings.whisper_model,
        repo_id="",
        chunk_seconds=settings.whisper_chunk_seconds,
        beam_size=settings.whisper_beam_size,
    )
    return get_model_for_spec(spec)


def get_model_for_spec(spec: AsrModelSpec) -> Any:
    key = (spec.model_name, settings.whisper_device, settings.whisper_compute_type)
    if key not in _models:
        with _model_lock:
            if key not in _models:
                from faster_whisper import WhisperModel

                _models[key] = WhisperModel(
                    spec.model_name,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
    return _models[key]


def pcm16_to_float32(raw_audio: bytes) -> np.ndarray:
    pcm = np.frombuffer(raw_audio, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def transcribe_chunk(raw_audio: bytes, prompt: str, spec: AsrModelSpec) -> str:
    audio = pcm16_to_float32(raw_audio)
    if audio.size < SAMPLE_RATE * 0.5:
        return ""

    model = get_model_for_spec(spec)
    segments, _info = model.transcribe(
        audio,
        language=settings.whisper_language,
        task="transcribe",
        beam_size=spec.beam_size,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=True,
        initial_prompt=prompt or settings.whisper_initial_prompt,
    )
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return " ".join(text.split())


async def stream_websocket(websocket: WebSocket, spec: AsrModelSpec | None = None) -> None:
    if spec is None:
        spec = AsrModelSpec(
            id="legacy-whisper",
            label="Whisper",
            detail="",
            engine="whisper",
            model_name=settings.whisper_model,
            repo_id="",
            chunk_seconds=settings.whisper_chunk_seconds,
            beam_size=settings.whisper_beam_size,
        )
    cfg = status()
    if not cfg.configured:
        await websocket.send_json({"type": "error", "message": cfg.reason})
        return

    await websocket.send_json(
        {
            "type": "loading",
            "message": f"Loading {spec.label}",
        }
    )
    try:
        await asyncio.to_thread(get_model_for_spec, spec)
    except Exception as exc:  # pragma: no cover - model download/runtime dependent
        await websocket.send_json({"type": "error", "message": f"Whisper model load failed: {exc}"})
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
    transcript_tail = ""

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
        nonlocal transcript_tail
        buffer = bytearray()

        async def flush(force: bool = False) -> None:
            nonlocal transcript_tail
            if len(buffer) < min_flush_bytes:
                return
            if not force and len(buffer) < chunk_bytes:
                return
            raw = bytes(buffer)
            buffer.clear()
            prompt = (settings.whisper_initial_prompt + " " + transcript_tail[-500:]).strip()
            try:
                text = await asyncio.to_thread(transcribe_chunk, raw, prompt, spec)
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                await websocket.send_json({"type": "error", "message": f"Whisper failed: {exc}"})
                stop_event.set()
                return
            if text:
                transcript_tail = (transcript_tail + " " + text).strip()
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
