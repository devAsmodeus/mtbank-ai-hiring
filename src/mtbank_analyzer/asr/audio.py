"""Загрузка и декодирование аудио.

Поддерживаемые контейнеры - WAV, MP3, OGG (требование ТЗ). Формат определяется
по магическим байтам, а не по расширению/Content-Type - им доверять нельзя.
Декодирование - через PyAV (зависимость faster-whisper), т.е. без внешнего
вызова ffmpeg; на выходе всегда mono float32 16 кГц + стерео-дорожки, если
запись действительно двухканальная (в колл-центрах оператор/клиент часто
пишутся в отдельные каналы - это даёт идеальную диаризацию).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import httpx
import numpy as np

from mtbank_analyzer.logging_setup import get_logger

logger = get_logger(__name__)

TARGET_SAMPLE_RATE = 16_000

SUPPORTED_FORMATS = ("wav", "mp3", "ogg")


class AudioError(ValueError):
    """Проблема с входным аудио - транслируется пользователю как 4xx."""


@dataclass(frozen=True)
class DecodedAudio:
    """Результат декодирования."""

    mono: np.ndarray  # float32, 16 кГц
    sample_rate: int
    duration_sec: float
    #: (левый, правый) канал - только если исходник реально стерео
    channels: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def is_stereo(self) -> bool:
        return self.channels is not None


def sniff_format(data: bytes) -> str | None:
    """Определяет формат по магическим байтам."""
    if len(data) < 12:
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:3] == b"ID3":
        return "mp3"
    # MP3 без ID3-тега: frame sync 0xFFEx/0xFFFx
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    return None


def validate_format(data: bytes) -> str:
    fmt = sniff_format(data)
    if fmt not in SUPPORTED_FORMATS:
        raise AudioError("Неподдерживаемый формат аудио. Поддерживаются: WAV, MP3, OGG.")
    return fmt


async def fetch_audio_from_url(url: str, max_bytes: int) -> bytes:
    """Скачивает аудио по URL с потоковым контролем размера."""
    if not url.lower().startswith(("http://", "https://")):
        raise AudioError("Поддерживаются только http(s) URL")
    try:
        async with (
            httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise AudioError(f"Файл больше лимита {max_bytes // (1024 * 1024)} МБ")
                chunks.append(chunk)
    except httpx.HTTPStatusError as exc:
        raise AudioError(f"Не удалось скачать аудио: HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise AudioError(f"Не удалось скачать аудио: {exc}") from exc
    data = b"".join(chunks)
    logger.info("audio_downloaded", url=url, size_bytes=len(data))
    return data


def probe_duration_sec(data: bytes) -> float | None:
    """Длительность из заголовка контейнера без декодирования (быстро, ~мс).

    Позволяет отсечь очень длинное аудио до полной материализации волны в
    памяти (mp3 с низким битрейтом раздувается в гигабайты PCM). Возвращает
    None, если контейнер не сообщает длительность (например, потоковый ответ).
    """
    import av

    try:
        with av.open(io.BytesIO(data)) as container:
            total = getattr(container, "duration", None)  # μs; есть только у InputContainer
            if total is not None:
                return float(total) / av.time_base
            audio_streams = container.streams.audio
            if audio_streams:
                stream = audio_streams[0]
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


def decode_bytes(data: bytes, sampling_rate: int = TARGET_SAMPLE_RATE) -> DecodedAudio:
    """Декодирует WAV/MP3/OGG в float32 16 кГц; определяет реальное стерео.

    Декодируем сразу с ``split_stereo=True``: для монозаписи PyAV даёт два
    идентичных канала - сравнением каналов отличаем настоящее стерео от
    апмикса без отдельного пробинга контейнера.
    """
    from faster_whisper.audio import decode_audio  # тяжёлый импорт - локально

    validate_format(data)
    try:
        left, right = decode_audio(io.BytesIO(data), sampling_rate=sampling_rate, split_stereo=True)
    except Exception as exc:
        raise AudioError(f"Не удалось декодировать аудио: {exc}") from exc

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.size == 0:
        raise AudioError("Пустая аудиодорожка")

    is_stereo = left.shape != right.shape or not np.allclose(left, right, atol=1e-4)
    mono = ((left + right) / 2.0).astype(np.float32) if is_stereo else left
    decoded = DecodedAudio(
        mono=mono,
        sample_rate=sampling_rate,
        duration_sec=round(len(mono) / sampling_rate, 2),
        channels=(left, right) if is_stereo else None,
    )
    logger.info(
        "audio_decoded",
        duration_sec=decoded.duration_sec,
        stereo=decoded.is_stereo,
        samples=len(mono),
    )
    return decoded
