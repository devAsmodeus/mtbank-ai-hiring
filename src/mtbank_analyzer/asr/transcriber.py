"""Обёртка faster-whisper.

Модель загружается лениво и один раз (потокобезопасно), инференс - синхронный
и CPU-bound: вызывающая сторона выносит его в тред-пул (``asyncio.to_thread``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from mtbank_analyzer.logging_setup import get_logger

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Word

logger = get_logger(__name__)


@dataclass(frozen=True)
class RawSegment:
    """Сегмент ASR до диаризации."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscribeOutcome:
    segments: list[RawSegment]
    language: str | None


def _split_by_word_gap(words: list[Word], gap_sec: float) -> list[RawSegment]:
    """Режет whisper-сегмент на реплики по паузам между словами."""
    segments: list[RawSegment] = []
    group: list[Word] = []
    for word in words:
        if group and word.start - group[-1].end >= gap_sec:
            segments.append(_words_to_segment(group))
            group = []
        group.append(word)
    if group:
        segments.append(_words_to_segment(group))
    return [s for s in segments if s.text]


def _words_to_segment(words: list[Word]) -> RawSegment:
    return RawSegment(
        start=round(words[0].start, 2),
        end=round(words[-1].end, 2),
        text="".join(w.word for w in words).strip(),
    )


class Transcriber:
    """Ленивая обёртка WhisperModel с фиксированными настройками инференса."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 2,
        language: str | None = "ru",
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._language = language
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    started = perf_counter()
                    logger.info(
                        "whisper_loading",
                        model=self.model_name,
                        device=self._device,
                        compute_type=self._compute_type,
                    )
                    self._model = WhisperModel(
                        self.model_name,
                        device=self._device,
                        compute_type=self._compute_type,
                    )
                    logger.info(
                        "whisper_loaded",
                        model=self.model_name,
                        load_sec=round(perf_counter() - started, 1),
                    )
        return self._model

    def warmup(self) -> None:
        """Прогрев на старте сервиса, чтобы первый запрос не ждал загрузку."""
        self._get_model()

    def transcribe_waveform(
        self, waveform: np.ndarray, split_on_word_gap: float | None = None
    ) -> TranscribeOutcome:
        """Транскрибация float32 16 кГц волны. Блокирующий вызов.

        ``split_on_word_gap`` - для дорожек с длинными паузами (отдельный канал
        стереозаписи): whisper после VAD склеивает далёкие реплики в один
        сегмент; включаем пословные таймстемпы и режем сегменты по паузам
        между словами, иначе при слиянии каналов ломается порядок реплик.
        """
        model = self._get_model()
        started = perf_counter()
        segments_iter, info = model.transcribe(
            waveform,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,
            word_timestamps=split_on_word_gap is not None,
            # телефонные записи: без опоры на предыдущий текст -
            # меньше галлюцинационных повторов на шумных участках
            condition_on_previous_text=False,
        )
        segments: list[RawSegment] = []
        for s in segments_iter:
            if split_on_word_gap is not None and s.words:
                segments.extend(_split_by_word_gap(s.words, split_on_word_gap))
            elif s.text.strip():
                segments.append(
                    RawSegment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
                )
        elapsed = perf_counter() - started
        audio_sec = len(waveform) / 16_000
        logger.info(
            "transcription_done",
            segments=len(segments),
            language=info.language,
            audio_sec=round(audio_sec, 1),
            asr_sec=round(elapsed, 1),
            rtf=round(elapsed / audio_sec, 2) if audio_sec else None,
        )
        return TranscribeOutcome(segments=segments, language=info.language)
