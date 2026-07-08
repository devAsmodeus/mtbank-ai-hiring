"""Общие фикстуры и стабы: FakeLLM, стаб-транскрайбер, генераторы аудио."""

from __future__ import annotations

import io
import json
import wave
from typing import ClassVar

import numpy as np
import pytest

from mtbank_analyzer.agents.base import AgentContext
from mtbank_analyzer.asr.transcriber import RawSegment, TranscribeOutcome
from mtbank_analyzer.config import Settings
from mtbank_analyzer.schemas import CLIENT, OPERATOR, TranscriptSegment

SR = 16_000


# ------------------------------------------------------------------ фейк-LLM


class FakeLLM:
    """Детерминированная подмена LLM: отдаёт заготовленные ответы по очереди."""

    model_name = "fake-llm"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if not self._responses:
            raise AssertionError("FakeLLM: закончились заготовленные ответы")
        return self._responses.pop(0)


class FailingLLM:
    """LLM, падающая транспортной ошибкой."""

    model_name = "failing-llm"

    async def complete(self, *, system: str, user: str) -> str:
        raise ConnectionError("LLM недоступна")


class RoutingFakeLLM:
    """Fake-LLM для параллельного графа: выбирает ответ по системному промпту
    (порядок вызова агентов недетерминирован). Ответы переиспользуются."""

    model_name = "routing-fake"

    _MARKERS: ClassVar[dict[str, str]] = {
        "агент-классификатор": "classifier",
        "оценки качества": "quality",
        "compliance-агент": "compliance",
        "агент-суммаризатор": "summarizer",
        "аналитик контакт-центра": "trends",
    }

    def __init__(self, responses: dict[str, dict | str]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    async def complete(self, *, system: str, user: str) -> str:
        agent = next(
            (name for marker, name in self._MARKERS.items() if marker in system),
            None,
        )
        assert agent is not None, f"неизвестный системный промпт: {system[:80]}"
        self.calls.append(agent)
        response = self._responses[agent]
        if isinstance(response, str):
            if response == "__RAISE__":
                raise ConnectionError("LLM недоступна")
            return response
        return json.dumps(response, ensure_ascii=False)


GOOD_RESPONSES: dict[str, dict | str] = {
    "classifier": {"topic": "кредиты", "priority": "medium", "reason": "кредит наличными"},
    "quality": {
        "checklist": {
            "greeting": True,
            "need_detection": True,
            "solution_provided": True,
            "farewell": True,
        },
        "comments": ["Разговор построен корректно"],
    },
    "compliance": {"issues": []},
    "summarizer": {
        "summary": "Клиент уточнил условия кредита наличными и решил подать заявку онлайн.",
        "action_items": ["Оператор: отправить инструкцию на email"],
    },
    "trends": {
        "patterns": ["Клиенты чаще всего спрашивают про кредиты наличными"],
        "recommendations": ["Обновить скрипт по кредитным продуктам"],
    },
}


def fake_llm(*responses: str | dict) -> FakeLLM:
    """Хелпер: dict сериализуется в JSON автоматически."""
    return FakeLLM(
        [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in responses]
    )


# ------------------------------------------------------------------- аудио


def make_wav_bytes(waveform: np.ndarray, sample_rate: int = SR, channels: int = 1) -> bytes:
    """PCM16 WAV из float32-волны (столбцы — каналы при channels=2)."""
    pcm = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def tone(freq: float, seconds: float, sample_rate: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class StubTranscriber:
    """Транскрайбер без модели: мгновенный фиксированный результат."""

    model_name = "stub-whisper"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def warmup(self) -> None:  # интерфейс реального Transcriber
        pass

    def transcribe_waveform(
        self, waveform: np.ndarray, split_on_word_gap: float | None = None
    ) -> TranscribeOutcome:
        self.calls.append(len(waveform))
        return TranscribeOutcome(
            segments=[RawSegment(0.0, 1.0, "Добрый день, МТБанк, меня зовут Анна.")],
            language="ru",
        )


# ----------------------------------------------------------------- фикстуры


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(storage_dir=tmp_path / "data", _env_file=None)


@pytest.fixture
def segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            speaker=OPERATOR,
            start=0.0,
            end=4.2,
            text="Добрый день, МТБанк, меня зовут Анна, чем могу помочь?",
        ),
        TranscriptSegment(
            speaker=CLIENT,
            start=4.5,
            end=8.1,
            text="Здравствуйте, хочу узнать про кредит наличными.",
        ),
        TranscriptSegment(
            speaker=OPERATOR,
            start=8.4,
            end=14.0,
            text="Подскажите, какая сумма и срок вас интересуют?",
        ),
        TranscriptSegment(
            speaker=CLIENT,
            start=14.2,
            end=17.5,
            text="Десять тысяч рублей на год.",
        ),
        TranscriptSegment(
            speaker=OPERATOR,
            start=17.8,
            end=25.0,
            text="Ставка от четырнадцати и девяти процентов годовых, решение за пятнадцать минут.",
        ),
        TranscriptSegment(
            speaker=OPERATOR,
            start=25.2,
            end=28.0,
            text="Спасибо за обращение в МТБанк, хорошего дня!",
        ),
    ]


@pytest.fixture
def ctx(segments: list[TranscriptSegment]) -> AgentContext:
    return AgentContext.from_segments(segments, duration_sec=28.0, language="ru")
