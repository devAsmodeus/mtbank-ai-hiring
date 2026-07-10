"""Базовая диаризация: разделение реплик на «Оператор» / «Клиент».

Два пути (выбирается автоматически):

1. **Стерео** — в колл-центрах запись обычно двухканальная (оператор и клиент
   в отдельных каналах). Каналы транскрибируются раздельно — диаризация
   получается точной по построению.
2. **Моно** — лёгкая спикер-кластеризация без тяжёлых моделей: MFCC-статистики
   по каждому ASR-сегменту → иерархическая кластеризация (cosine, average
   linkage) на 2 кластера. Без gated-моделей (pyannote требует HF-токен) —
   ``docker compose up`` работает из коробки; продакшен-апгрейд описан в README.

Роли назначаются по содержанию реплик (маркеры речи оператора) с приором
«первым в входящем звонке говорит оператор».
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from mtbank_analyzer.asr.transcriber import RawSegment
from mtbank_analyzer.logging_setup import get_logger
from mtbank_analyzer.schemas import CLIENT, OPERATOR, TranscriptSegment

logger = get_logger(__name__)

# Маркеры речи оператора контакт-центра (для назначения ролей кластерам)
_OPERATOR_MARKERS = [
    r"мтбанк",
    r"меня зовут",
    r"чем могу помочь",
    r"вы позвонили",
    r"спасибо за обращение",
    r"оставайтесь на линии",
    r"могу (?:вам )?предложить",
    r"оформ(?:ить|им|лю) (?:вам|заявку)",
    r"ваша заявка",
    r"уточните,? пожалуйста",
]
_CLIENT_MARKERS = [
    r"хочу узнать",
    r"хотел[а]? бы",
    r"у меня вопрос",
    r"подскажите",
    r"мне нужн[оа]",
    r"я звоню",
    r"моя карта",
    r"мой счёт",
]

_MIN_SEGMENT_SEC = 0.25  # короче — не считаем эмбеддинг, наследуем спикера


def _marker_score(text: str, patterns: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for p in patterns if re.search(p, lowered))


def _operator_likeness(texts: list[str]) -> int:
    joined = " ".join(texts)
    return _marker_score(joined, _OPERATOR_MARKERS) - _marker_score(joined, _CLIENT_MARKERS)


# --------------------------------------------------------------------- MFCC


def mfcc_embedding(
    waveform: np.ndarray,
    sample_rate: int = 16_000,
    n_mels: int = 26,
    n_coeffs: int = 13,
) -> np.ndarray | None:
    """MFCC-статистики фрагмента: (mean ‖ std) по фреймам → вектор 2*n_coeffs.

    Реализация на numpy/scipy — без librosa/torch: для разделения двух голосов
    в телефонном канале статистик тембра достаточно (см. README, ограничения).
    """
    from scipy.fft import dct, rfft

    frame_len = int(0.025 * sample_rate)  # 25 мс
    hop = int(0.010 * sample_rate)  # 10 мс
    if len(waveform) < frame_len * 3:
        return None

    emphasized = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    n_frames = 1 + (len(emphasized) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = emphasized[idx] * np.hamming(frame_len)

    spectrum = np.abs(rfft(frames, n=512, axis=1)) ** 2 / 512
    mel_fb = _mel_filterbank(sample_rate, n_fft=512, n_mels=n_mels)
    mel_energy = np.log(spectrum @ mel_fb.T + 1e-10)
    mfcc = dct(mel_energy, type=2, axis=1, norm="ortho")[:, :n_coeffs]

    # энергичные фреймы (отсекаем тишину внутри сегмента)
    energy = mel_energy.mean(axis=1)
    voiced = mfcc[energy > np.percentile(energy, 30)]
    if len(voiced) < 3:
        voiced = mfcc
    embedding: np.ndarray = np.concatenate([voiced.mean(axis=0), voiced.std(axis=0)])
    return embedding


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz: float) -> float:
        return float(2595.0 * np.log10(1.0 + hz / 700.0))

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            if center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


def cluster_two_speakers(embeddings: np.ndarray) -> np.ndarray:
    """Иерархическая кластеризация эмбеддингов на 2 кластера (cosine/average)."""
    from scipy.cluster.hierarchy import fcluster, linkage

    normalized = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)
    links = linkage(normalized, method="average", metric="cosine")
    labels: np.ndarray = fcluster(links, t=2, criterion="maxclust") - 1  # метки 0/1
    return labels


# ----------------------------------------------------------------- Diarizer


@dataclass
class Diarizer:
    """Назначает спикеров ASR-сегментам."""

    enabled: bool = True
    max_speakers: int = 2

    # --- моно: кластеризация ---

    def diarize_mono(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        raw_segments: list[RawSegment],
    ) -> list[TranscriptSegment]:
        if not raw_segments:
            return []
        if not self.enabled or len(raw_segments) < 4:
            return self._single_speaker(raw_segments)

        embeddings: list[np.ndarray | None] = []
        for seg in raw_segments:
            if seg.end - seg.start < _MIN_SEGMENT_SEC:
                embeddings.append(None)
                continue
            piece = waveform[int(seg.start * sample_rate) : int(seg.end * sample_rate)]
            embeddings.append(mfcc_embedding(piece, sample_rate))

        valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]
        if len(valid) < 4:
            return self._single_speaker(raw_segments)

        labels_valid = cluster_two_speakers(np.stack([e for _, e in valid]))
        by_index = {i: int(label) for (i, _), label in zip(valid, labels_valid, strict=True)}
        # короткие сегменты (без эмбеддинга) наследуют спикера предыдущей реплики
        labels: list[int] = []
        for i in range(len(raw_segments)):
            labels.append(by_index.get(i, labels[-1] if labels else 0))

        if self._is_degenerate(labels, raw_segments):
            logger.info("diarization_collapsed_to_single_speaker")
            return self._single_speaker(raw_segments)

        roles = self._assign_roles_by_label(labels, raw_segments)
        result = [
            TranscriptSegment(speaker=roles[label], start=seg.start, end=seg.end, text=seg.text)
            for label, seg in zip(labels, raw_segments, strict=True)
        ]
        logger.info(
            "diarization_done",
            mode="mono_clustering",
            segments=len(result),
            operator_segments=sum(1 for s in result if s.speaker == OPERATOR),
        )
        return result

    # --- стерео: каналы уже разделены ---

    def label_stereo(
        self,
        left_segments: list[RawSegment],
        right_segments: list[RawSegment],
    ) -> list[TranscriptSegment]:
        """Каналы → роли: маркеры речи + приор «оператор говорит первым»."""
        left_texts = [s.text for s in left_segments]
        right_texts = [s.text for s in right_segments]

        left_score = _operator_likeness(left_texts)
        right_score = _operator_likeness(right_texts)
        if left_score != right_score:
            left_is_operator = left_score > right_score
        else:
            first_left = left_segments[0].start if left_segments else float("inf")
            first_right = right_segments[0].start if right_segments else float("inf")
            left_is_operator = first_left <= first_right

        def to_transcript(segs: list[RawSegment], role: str) -> list[TranscriptSegment]:
            return [
                TranscriptSegment(speaker=role, start=s.start, end=s.end, text=s.text) for s in segs
            ]

        merged = to_transcript(left_segments, OPERATOR if left_is_operator else CLIENT)
        merged += to_transcript(right_segments, CLIENT if left_is_operator else OPERATOR)
        merged.sort(key=lambda s: (s.start, s.end))
        logger.info("diarization_done", mode="stereo_channels", segments=len(merged))
        return merged

    # --- вспомогательные ---

    @staticmethod
    def _is_degenerate(labels: list[int], segments: list[RawSegment]) -> bool:
        """Кластеризация «развалилась»: один кластер почти пуст по времени."""
        durations = {0: 0.0, 1: 0.0}
        for label, seg in zip(labels, segments, strict=True):
            durations[label] += seg.end - seg.start
        total = sum(durations.values())
        return total <= 0 or min(durations.values()) / total < 0.05

    @staticmethod
    def _single_speaker(raw_segments: list[RawSegment]) -> list[TranscriptSegment]:
        """Один говорящий: роль — по содержанию (по умолчанию Клиент)."""
        role = OPERATOR if _operator_likeness([s.text for s in raw_segments]) > 0 else CLIENT
        return [
            TranscriptSegment(speaker=role, start=s.start, end=s.end, text=s.text)
            for s in raw_segments
        ]

    def _assign_roles_by_label(
        self, labels: list[int], segments: list[RawSegment]
    ) -> dict[int, str]:
        texts: dict[int, list[str]] = {0: [], 1: []}
        for label, seg in zip(labels, segments, strict=True):
            texts[label].append(seg.text)

        score_0 = _operator_likeness(texts[0])
        score_1 = _operator_likeness(texts[1])
        if score_0 != score_1:  # noqa: SIM108 — вложенный тернарник хуже читается
            operator_label = 0 if score_0 > score_1 else 1
        else:
            operator_label = labels[0]  # приор: первым говорит оператор
        return {
            operator_label: OPERATOR,
            1 - operator_label: CLIENT,
        }
