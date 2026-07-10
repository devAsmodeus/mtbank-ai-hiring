"""Демо-клиент real-time транскрибации (бонус: WebSocket).

Стримит WAV-файл в ``/ws/transcribe`` кусками по 250 мс, имитируя живой
микрофон, и печатает сегменты по мере распознавания. В конце шлёт ``flush``
и получает финальный диаризованный транскрипт.

Запуск:
    python scripts/ws_client.py test_data/dialog_credit_16k.wav
    python scripts/ws_client.py audio.wav --url ws://localhost:8000/ws/transcribe --fast
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import websockets

CHUNK_SEC = 0.25


async def stream(path: Path, url: str, realtime: bool) -> None:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            sys.exit("нужен WAV PCM16 mono (например, test_data/dialog_credit_16k.wav)")
        sample_rate = wav.getframerate()
        frames_per_chunk = int(CHUNK_SEC * sample_rate)
        audio = wav.readframes(wav.getnframes())

    bytes_per_chunk = frames_per_chunk * 2
    total_sec = len(audio) / 2 / sample_rate
    print(f"стримим {path.name}: {total_sec:.1f} c @ {sample_rate} Гц → {url}")

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"sample_rate": sample_rate}))
        print("<<", await ws.recv())

        async def receiver() -> None:
            started = time.perf_counter()
            try:
                async for message in ws:
                    data = json.loads(message)
                    elapsed = time.perf_counter() - started
                    if data["type"] == "segment":
                        # латентность = момент получения минус конец сегмента
                        latency = elapsed - data["end"]
                        print(
                            f"<< [{data['start']:6.1f}-{data['end']:6.1f}] "
                            f"{data['text']}  (латентность ~{latency:.1f} c)"
                        )
                    elif data["type"] == "done":
                        print("\n== финальный транскрипт (с диаризацией) ==")
                        for seg in data["segments"]:
                            print(f"  [{seg['start']:6.1f}] {seg['speaker']}: {seg['text']}")
                        return
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        for offset in range(0, len(audio), bytes_per_chunk):
            await ws.send(audio[offset : offset + bytes_per_chunk])
            if realtime:
                await asyncio.sleep(CHUNK_SEC)

        await ws.send(json.dumps({"type": "flush"}))
        await recv_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="ws://localhost:8000/ws/transcribe")
    parser.add_argument(
        "--fast", action="store_true", help="слать без пауз (не имитировать реальное время)"
    )
    args = parser.parse_args()
    asyncio.run(stream(args.audio, args.url, realtime=not args.fast))
