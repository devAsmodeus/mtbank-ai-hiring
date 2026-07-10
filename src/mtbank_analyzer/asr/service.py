"""TranscriptionService - фасад ASR: bytes → структурированный транскрипт."""

from __future__ import annotations

import asyncio

import numpy as np

from mtbank_analyzer.asr.audio import AudioError, DecodedAudio, decode_bytes, probe_duration_sec
from mtbank_analyzer.asr.diarizer import Diarizer
from mtbank_analyzer.asr.transcriber import TranscribeOutcome, Transcriber
from mtbank_analyzer.config import Settings
from mtbank_analyzer.logging_setup import get_logger
from mtbank_analyzer.schemas import TranscriptionResult, TranscriptSegment

logger = get_logger(__name__)


class TranscriptionService:
    """Оркестрирует декодирование → ASR → диаризацию.

    Инференс whisper CPU-bound и выполняется в тред-пуле; одновременно - одна
    транскрибация (семафор): на CPU параллельные прогоны только делят ядра
    и раздувают память, увеличивая латентность каждого запроса.
    """

    def __init__(
        self,
        settings: Settings,
        transcriber: Transcriber | None = None,
        diarizer: Diarizer | None = None,
    ) -> None:
        self._settings = settings
        self.transcriber = transcriber or Transcriber(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            language=settings.whisper_language,
        )
        self.diarizer = diarizer or Diarizer(enabled=settings.diarization_enabled)
        self._semaphore = asyncio.Semaphore(1)

    async def transcribe_block(self, waveform: np.ndarray) -> TranscribeOutcome:
        """Транскрибация одного блока под общим семафором (real-time WS).

        Ходит через тот же семафор, что и transcribe_bytes, поэтому WS-стрим и
        REST /analyze не запускают конкурентные whisper-инференсы на CPU.
        """
        async with self._semaphore:
            return await asyncio.to_thread(self.transcriber.transcribe_waveform, waveform)

    async def transcribe_bytes(self, data: bytes) -> TranscriptionResult:
        """Полный ASR-пайплайн для готового файла."""
        max_sec = self._settings.max_audio_duration_sec
        # Отсекаем длинное аудио по метаданным контейнера ДО декодирования,
        # чтобы низкобитрейтный файл не раздулся в память гигабайтами PCM.
        probed = await asyncio.to_thread(probe_duration_sec, data)
        if probed is not None and probed > max_sec:
            raise AudioError(f"Аудио длиннее лимита {max_sec / 60:.0f} мин")

        decoded: DecodedAudio = await asyncio.to_thread(decode_bytes, data)
        if decoded.duration_sec > max_sec:
            raise AudioError(f"Аудио длиннее лимита {max_sec / 60:.0f} мин")

        async with self._semaphore:
            if decoded.is_stereo and self.diarizer.enabled:
                segments, language = await self._transcribe_stereo(decoded)
            else:
                segments, language = await self._transcribe_mono(decoded)

        return TranscriptionResult(
            segments=segments,
            language=language,
            duration_sec=decoded.duration_sec,
            asr_model=self.transcriber.model_name,
        )

    async def _transcribe_mono(
        self, decoded: DecodedAudio
    ) -> tuple[list[TranscriptSegment], str | None]:
        outcome = await asyncio.to_thread(self.transcriber.transcribe_waveform, decoded.mono)
        segments = self.diarizer.diarize_mono(decoded.mono, decoded.sample_rate, outcome.segments)
        return segments, outcome.language

    # пауза между словами, по которой канал режется на реплики: у канала
    # одного говорящего длинные окна тишины (говорит собеседник) - whisper
    # склеил бы далёкие реплики, сломав порядок при слиянии каналов
    _STEREO_WORD_GAP_SEC = 0.6

    async def _transcribe_stereo(
        self, decoded: DecodedAudio
    ) -> tuple[list[TranscriptSegment], str | None]:
        """Каналы транскрибируются по отдельности - диаризация по построению."""
        assert decoded.channels is not None
        left, right = decoded.channels
        left_out = await asyncio.to_thread(
            self.transcriber.transcribe_waveform, left, self._STEREO_WORD_GAP_SEC
        )
        right_out = await asyncio.to_thread(
            self.transcriber.transcribe_waveform, right, self._STEREO_WORD_GAP_SEC
        )
        segments = self.diarizer.label_stereo(left_out.segments, right_out.segments)
        return segments, left_out.language or right_out.language
