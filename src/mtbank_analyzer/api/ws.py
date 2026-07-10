"""Real-time транскрибация по WebSocket (бонусное задание).

Протокол ``/ws/transcribe``:

1. Клиент подключается и (опционально) шлёт текстовый кадр-конфиг:
   ``{"sample_rate": 16000}`` - PCM16 mono little-endian; по умолчанию 16 кГц.
2. Клиент стримит бинарные кадры PCM16.
3. Каждые ~2 секунды накопленного аудио сервер транскрибирует блок и шлёт
   ``{"type": "segment", "start": ..., "end": ..., "text": ...}``.
   Латентность ≈ блок (2 c) + инференс (~0.5-1 c на CPU) < 3 c.
4. Кадр ``{"type": "flush"}`` - сервер дорабатывает остаток и шлёт
   ``{"type": "done", "segments": [...]}`` с диаризацией по накопленным блокам.

Ограничение прототипа: сегменты в стриме могут рваться на границах блоков;
финальная диаризация идёт по этим же сегментам, отдельного повторного прохода
ASR по всей записи нет (это удваивало бы CPU на каждый сеанс).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mtbank_analyzer.asr.transcriber import RawSegment
from mtbank_analyzer.logging_setup import get_logger

logger = get_logger(__name__)

ws_router = APIRouter()

TARGET_SR = 16_000
BLOCK_SEC = 2.0
_MIN_TAIL_SEC = 0.3


def _pcm16_to_float32(data: bytes, source_sr: int) -> np.ndarray:
    # нечётный хвост (обрыв на середине сэмпла) отбрасываем, иначе frombuffer падает
    if len(data) % 2:
        data = data[: len(data) - 1]
    waveform = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if source_sr == TARGET_SR or waveform.size == 0:
        return waveform
    target_len = int(len(waveform) * TARGET_SR / source_sr)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.linspace(0, len(waveform) - 1, target_len)
    resampled: np.ndarray = np.interp(positions, np.arange(len(waveform)), waveform)
    return resampled.astype(np.float32)


@ws_router.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()
    service = websocket.app.state.transcription
    max_sec = websocket.app.state.settings.max_audio_duration_sec
    source_sr = TARGET_SR
    block_samples = int(BLOCK_SEC * TARGET_SR)
    max_samples = int(max_sec * TARGET_SR)

    pending = np.zeros(0, dtype=np.float32)  # ещё не транскрибировано
    full_audio: list[np.ndarray] = []  # вся запись - для финальной диаризации
    accumulated: list[RawSegment] = []  # сегменты блоков с абсолютным временем
    processed_sec = 0.0
    total_samples = 0

    async def send(payload: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    async def process_block(block: np.ndarray) -> None:
        nonlocal processed_sec
        outcome = await service.transcribe_block(block)
        for seg in outcome.segments:
            abs_seg = RawSegment(
                start=round(seg.start + processed_sec, 2),
                end=round(seg.end + processed_sec, 2),
                text=seg.text,
            )
            accumulated.append(abs_seg)
            await send(
                {
                    "type": "segment",
                    "start": abs_seg.start,
                    "end": abs_seg.end,
                    "text": abs_seg.text,
                }
            )
        processed_sec += len(block) / TARGET_SR

    async def finalize() -> None:
        nonlocal pending
        if len(pending) >= int(_MIN_TAIL_SEC * TARGET_SR):
            await process_block(pending)
        pending = np.zeros(0, dtype=np.float32)

        waveform = np.concatenate(full_audio) if full_audio else np.zeros(0, dtype=np.float32)
        # диаризуем уже полученные сегменты по полной волне - без повторного ASR
        segments = service.diarizer.diarize_mono(waveform, TARGET_SR, accumulated)
        await send(
            {
                "type": "done",
                "duration_sec": round(waveform.size / TARGET_SR, 2),
                "segments": [seg.model_dump() for seg in segments],
            }
        )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes"):
                chunk = _pcm16_to_float32(message["bytes"], source_sr)
                total_samples += len(chunk)
                if total_samples > max_samples:
                    await send(
                        {
                            "type": "error",
                            "detail": f"Превышен лимит длительности {max_sec / 60:.0f} мин",
                        }
                    )
                    break
                pending = np.concatenate([pending, chunk])
                full_audio.append(chunk)
                while len(pending) >= block_samples:
                    await process_block(pending[:block_samples])
                    pending = pending[block_samples:]

            elif message.get("text"):
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send({"type": "error", "detail": "ожидается JSON"})
                    continue
                if not isinstance(control, dict):
                    await send({"type": "error", "detail": "ожидается JSON-объект"})
                    continue
                if "sample_rate" in control:
                    try:
                        rate = int(control["sample_rate"])
                    except (TypeError, ValueError):
                        await send({"type": "error", "detail": "sample_rate должен быть числом"})
                        continue
                    if not 4000 <= rate <= 192_000:
                        await send(
                            {"type": "error", "detail": "sample_rate вне диапазона 4000-192000"}
                        )
                        continue
                    source_sr = rate
                    await send({"type": "ready", "sample_rate": source_sr})
                if control.get("type") == "flush":
                    await finalize()
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("ws_session_closed", processed_sec=round(processed_sec, 1))
