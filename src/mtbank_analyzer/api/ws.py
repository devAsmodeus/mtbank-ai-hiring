"""Real-time транскрибация по WebSocket (бонусное задание).

Протокол ``/ws/transcribe``:

1. Клиент подключается и (опционально) шлёт текстовый кадр-конфиг:
   ``{"sample_rate": 16000}`` — PCM16 mono little-endian; по умолчанию 16 кГц.
2. Клиент стримит бинарные кадры PCM16.
3. Каждые ~2 секунды накопленного аудио сервер транскрибирует блок и шлёт
   ``{"type": "segment", "start": ..., "end": ..., "text": ...}``.
   Латентность ≈ блок (2 c) + инференс (~0.5–1 c на CPU) < 3 c.
4. Кадр ``{"type": "flush"}`` — сервер дорабатывает остаток, прогоняет
   диаризацию по всей записи и шлёт ``{"type": "done", "segments": [...]}``.

Ограничение прототипа: сегменты в стриме могут рваться на границах блоков;
финальный ответ ``done`` собирается по полной записи и от этого не страдает.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mtbank_analyzer.logging_setup import get_logger

logger = get_logger(__name__)

ws_router = APIRouter()

TARGET_SR = 16_000
BLOCK_SEC = 2.0
_MIN_TAIL_SEC = 0.3


def _pcm16_to_float32(data: bytes, source_sr: int) -> np.ndarray:
    waveform = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if source_sr == TARGET_SR or waveform.size == 0:
        return waveform
    target_len = int(len(waveform) * TARGET_SR / source_sr)
    positions = np.linspace(0, len(waveform) - 1, target_len)
    return np.interp(positions, np.arange(len(waveform)), waveform).astype(np.float32)


@ws_router.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()
    service = websocket.app.state.transcription
    source_sr = TARGET_SR
    block_samples = int(BLOCK_SEC * TARGET_SR)

    pending = np.zeros(0, dtype=np.float32)  # ещё не транскрибировано
    full_audio: list[np.ndarray] = []  # вся запись — для финальной диаризации
    processed_sec = 0.0

    async def process_block(block: np.ndarray) -> None:
        nonlocal processed_sec
        outcome = await asyncio.to_thread(service.transcriber.transcribe_waveform, block)
        for seg in outcome.segments:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "segment",
                        "start": round(seg.start + processed_sec, 2),
                        "end": round(seg.end + processed_sec, 2),
                        "text": seg.text,
                    },
                    ensure_ascii=False,
                )
            )
        processed_sec += len(block) / TARGET_SR

    async def finalize() -> None:
        """Остаток буфера + диаризация всей записи."""
        nonlocal pending
        if len(pending) >= int(_MIN_TAIL_SEC * TARGET_SR):
            await process_block(pending)
        pending = np.zeros(0, dtype=np.float32)

        waveform = np.concatenate(full_audio) if full_audio else np.zeros(0, dtype=np.float32)
        segments = []
        if waveform.size >= int(_MIN_TAIL_SEC * TARGET_SR):
            outcome = await asyncio.to_thread(service.transcriber.transcribe_waveform, waveform)
            segments = service.diarizer.diarize_mono(waveform, TARGET_SR, outcome.segments)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "done",
                    "duration_sec": round(waveform.size / TARGET_SR, 2),
                    "segments": [seg.model_dump() for seg in segments],
                },
                ensure_ascii=False,
            )
        )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes"):
                chunk = _pcm16_to_float32(message["bytes"], source_sr)
                pending = np.concatenate([pending, chunk])
                full_audio.append(chunk)
                while len(pending) >= block_samples:
                    await process_block(pending[:block_samples])
                    pending = pending[block_samples:]

            elif message.get("text"):
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": "ожидается JSON"})
                    )
                    continue
                if "sample_rate" in control:
                    source_sr = int(control["sample_rate"])
                    await websocket.send_text(
                        json.dumps({"type": "ready", "sample_rate": source_sr})
                    )
                if control.get("type") == "flush":
                    await finalize()
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("ws_session_closed", processed_sec=round(processed_sec, 1))
