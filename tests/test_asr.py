"""Тесты ASR-модуля: аудио-утилиты, диаризация, сервис (без загрузки whisper)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from tests.conftest import SR, StubTranscriber, make_wav_bytes, tone

from mtbank_analyzer.asr.audio import AudioError, decode_bytes, sniff_format, validate_format
from mtbank_analyzer.asr.diarizer import Diarizer, cluster_two_speakers, mfcc_embedding
from mtbank_analyzer.asr.service import TranscriptionService
from mtbank_analyzer.asr.transcriber import RawSegment, _split_by_word_gap
from mtbank_analyzer.config import Settings
from mtbank_analyzer.schemas import CLIENT, OPERATOR

# -------------------------------------------------------------------- audio


def test_sniff_format_magic_bytes() -> None:
    assert sniff_format(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "wav"
    assert sniff_format(b"OggS" + b"\x00" * 20) == "ogg"
    assert sniff_format(b"ID3\x04\x00" + b"\x00" * 20) == "mp3"
    assert sniff_format(b"\xff\xfb\x90\x00" + b"\x00" * 20) == "mp3"
    assert sniff_format(b"\x1aE\xdf\xa3" + b"\x00" * 20) is None  # webm - не поддержан


def test_validate_format_rejects_unknown() -> None:
    with pytest.raises(AudioError, match="Неподдерживаемый формат"):
        validate_format(b"PK\x03\x04" + b"\x00" * 20)


def test_decode_mono_wav_8k_resamples_to_16k() -> None:
    data = make_wav_bytes(tone(440, 1.0, sample_rate=8000), sample_rate=8000)
    decoded = decode_bytes(data)

    assert decoded.sample_rate == SR
    assert not decoded.is_stereo
    assert decoded.duration_sec == pytest.approx(1.0, abs=0.1)
    assert len(decoded.mono) == pytest.approx(SR, rel=0.05)


def test_decode_detects_true_stereo() -> None:
    left, right = tone(300, 1.0), tone(2000, 1.0)
    data = make_wav_bytes(np.stack([left, right], axis=1), channels=2)
    decoded = decode_bytes(data)

    assert decoded.is_stereo
    assert decoded.channels is not None


def test_decode_upmixed_mono_is_not_stereo() -> None:
    mono = tone(440, 1.0)
    data = make_wav_bytes(np.stack([mono, mono], axis=1), channels=2)
    decoded = decode_bytes(data)

    assert not decoded.is_stereo


# ----------------------------------------------------------------- diarizer


def _two_speaker_waveform_and_segments() -> tuple[np.ndarray, list[RawSegment]]:
    """«Диалог»: чередование низкого (спикер A) и высокого (спикер B) тонов."""
    pieces: list[np.ndarray] = []
    segments: list[RawSegment] = []
    cursor = 0.0
    texts_a = [
        "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?",
        "Уточните, пожалуйста, сумму кредита.",
        "Спасибо за обращение, хорошего дня!",
    ]
    texts_b = [
        "Здравствуйте, хочу узнать про кредит.",
        "Десять тысяч рублей на год.",
        "Спасибо, до свидания.",
    ]
    for text_a, text_b in zip(texts_a, texts_b, strict=True):
        for freq, text in ((250, text_a), (1800, text_b)):
            piece = tone(freq, 1.2)
            segments.append(
                RawSegment(start=round(cursor, 2), end=round(cursor + 1.2, 2), text=text)
            )
            pieces.append(piece)
            cursor += 1.2
    return np.concatenate(pieces), segments


def test_mfcc_embedding_separates_different_tones() -> None:
    low = mfcc_embedding(tone(250, 1.0), SR)
    high = mfcc_embedding(tone(1800, 1.0), SR)
    assert low is not None and high is not None

    same = mfcc_embedding(tone(250, 1.0), SR)
    assert same is not None
    dist_same = float(np.linalg.norm(low - same))
    dist_diff = float(np.linalg.norm(low - high))
    assert dist_diff > dist_same * 3


def test_mfcc_embedding_returns_none_for_tiny_input() -> None:
    assert mfcc_embedding(tone(440, 0.01), SR) is None


def test_cluster_two_speakers_on_synthetic_tones() -> None:
    embs = np.stack([mfcc_embedding(tone(250 if i % 2 == 0 else 1800, 1.0), SR) for i in range(6)])
    labels = cluster_two_speakers(embs)

    assert set(labels) == {0, 1}
    assert len(set(labels[::2])) == 1  # все чётные - один кластер
    assert len(set(labels[1::2])) == 1
    assert labels[0] != labels[1]


def test_diarize_mono_assigns_operator_and_client_roles() -> None:
    waveform, segments = _two_speaker_waveform_and_segments()
    result = Diarizer().diarize_mono(waveform, SR, segments)

    assert [s.speaker for s in result] == [OPERATOR, CLIENT] * 3
    assert result[0].text.startswith("Добрый день, МТБанк")


def test_diarize_mono_few_segments_falls_back_to_single_speaker() -> None:
    waveform = tone(440, 2.0)
    segments = [RawSegment(0.0, 1.0, "Здравствуйте, у меня вопрос по карте.")]
    result = Diarizer().diarize_mono(waveform, SR, segments)

    assert [s.speaker for s in result] == [CLIENT]


def test_single_speaker_operator_marker_detection() -> None:
    waveform = tone(440, 3.0)
    segments = [
        RawSegment(0.0, 1.0, "Вы позвонили в МТБанк."),
        RawSegment(1.0, 2.0, "Оставайтесь на линии."),
    ]
    result = Diarizer().diarize_mono(waveform, SR, segments)
    assert all(s.speaker == OPERATOR for s in result)


def test_label_stereo_by_content_markers() -> None:
    operator_channel = [
        RawSegment(0.0, 2.0, "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?"),
        RawSegment(5.0, 7.0, "Уточните, пожалуйста, сумму."),
    ]
    client_channel = [
        RawSegment(2.5, 4.5, "Здравствуйте, хочу узнать про кредит."),
        RawSegment(7.5, 9.0, "Сто тысяч на два года."),
    ]
    # оператор - во ВТОРОМ (правом) канале: проверяем, что решают маркеры, а не порядок
    merged = Diarizer().label_stereo(client_channel, operator_channel)

    assert [s.speaker for s in merged] == [OPERATOR, CLIENT, OPERATOR, CLIENT]
    assert merged[0].start == 0.0  # отсортировано по времени


def test_label_stereo_tie_breaks_by_first_speaker() -> None:
    left = [RawSegment(0.0, 1.0, "Алло.")]
    right = [RawSegment(1.5, 2.5, "Алло, слушаю.")]
    merged = Diarizer().label_stereo(left, right)

    assert merged[0].speaker == OPERATOR  # первый заговоривший канал
    assert merged[1].speaker == CLIENT


# -------------------------------------------------------------- word gap split


@dataclass(frozen=True)
class FakeWord:
    start: float
    end: float
    word: str


def test_split_by_word_gap_cuts_merged_utterances() -> None:
    words = [
        FakeWord(0.0, 0.3, " Здравствуйте."),
        FakeWord(0.4, 0.7, " У"),
        FakeWord(0.75, 1.1, " меня"),
        FakeWord(1.15, 1.5, " вопрос."),
        # пауза 20 с - говорил собеседник (другой канал)
        FakeWord(21.5, 21.9, " Спасибо,"),
        FakeWord(22.0, 22.4, " понял."),
    ]
    segments = _split_by_word_gap(words, gap_sec=0.6)

    assert len(segments) == 2
    assert segments[0].text == "Здравствуйте. У меня вопрос."
    assert segments[0].start == 0.0 and segments[0].end == 1.5
    assert segments[1].text == "Спасибо, понял."
    assert segments[1].start == 21.5


def test_split_by_word_gap_keeps_continuous_speech_whole() -> None:
    words = [FakeWord(i * 0.4, i * 0.4 + 0.3, f" слово{i}") for i in range(5)]
    segments = _split_by_word_gap(words, gap_sec=0.6)
    assert len(segments) == 1


# ------------------------------------------------------------------ service


async def test_service_mono_pipeline(settings: Settings) -> None:
    service = TranscriptionService(settings, transcriber=StubTranscriber())
    data = make_wav_bytes(tone(440, 1.0))

    result = await service.transcribe_bytes(data)

    assert result.language == "ru"
    assert result.asr_model == "stub-whisper"
    assert result.duration_sec == pytest.approx(1.0, abs=0.1)
    assert len(result.segments) == 1


async def test_service_stereo_transcribes_both_channels(settings: Settings) -> None:
    stub = StubTranscriber()
    service = TranscriptionService(settings, transcriber=stub)
    data = make_wav_bytes(np.stack([tone(300, 1.0), tone(2000, 1.0)], axis=1), channels=2)

    result = await service.transcribe_bytes(data)

    assert len(stub.calls) == 2  # оба канала по отдельности
    assert len(result.segments) == 2


async def test_service_rejects_too_long_audio(settings: Settings) -> None:
    limited = settings.model_copy(update={"max_audio_duration_sec": 0.5})
    service = TranscriptionService(limited, transcriber=StubTranscriber())
    data = make_wav_bytes(tone(440, 2.0))

    with pytest.raises(AudioError, match="длиннее лимита"):
        await service.transcribe_bytes(data)
