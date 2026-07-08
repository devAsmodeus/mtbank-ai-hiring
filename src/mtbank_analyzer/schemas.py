"""Pydantic-модели: контракт данных всей системы.

JSON-контракт ответа ``/analyze`` зафиксирован в ТЗ — модели ниже воспроизводят
его 1:1 (поле ``meta`` — дополнительное, аддитивное расширение).
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

# Канонические говорящие. Диаризация всегда приводит спикеров к этим ролям.
OPERATOR = "Оператор"
CLIENT = "Клиент"

Topic = Literal["кредиты", "карты", "переводы", "жалобы", "другое"]
Priority = Literal["low", "medium", "high"]
Severity = Literal["low", "medium", "high"]


class TranscriptSegment(BaseModel):
    """Одна реплика транскрипта: спикер + временные метки + текст."""

    speaker: str = Field(description="Роль говорящего: «Оператор» или «Клиент»")
    start: float = Field(ge=0, description="Начало реплики, сек")
    end: float = Field(ge=0, description="Конец реплики, сек")
    text: str

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class Classification(BaseModel):
    """Результат агента-классификатора."""

    topic: Topic
    priority: Priority
    reason: str = Field(default="", description="Краткое объяснение выбора (для аудита)")


class QualityChecklist(BaseModel):
    """Чеклист оператора из ТЗ."""

    greeting: bool = Field(description="Поприветствовал, представился, назвал банк")
    need_detection: bool = Field(description="Выявил потребность уточняющими вопросами")
    solution_provided: bool = Field(description="Дал решение / ответил по существу")
    farewell: bool = Field(description="Корректно завершил разговор")


class QualityScore(BaseModel):
    """Оценка качества: чеклист от LLM, итоговый балл считается кодом.

    LLM отвечает за семантические суждения (было ли приветствие),
    арифметика — детерминированная (веса в ``compute_total``), чтобы балл
    был воспроизводимым и аудируемым.
    """

    total: int = Field(ge=0, le=100)
    checklist: QualityChecklist
    comments: list[str] = Field(default_factory=list, description="Замечания по разговору")

    # Веса пунктов чеклиста (в сумме 100)
    WEIGHTS: ClassVar[dict[str, int]] = {
        "greeting": 20,
        "need_detection": 25,
        "solution_provided": 35,
        "farewell": 20,
    }

    @classmethod
    def compute_total(cls, checklist: QualityChecklist) -> int:
        return sum(weight for item, weight in cls.WEIGHTS.items() if getattr(checklist, item))

    @classmethod
    def from_checklist(cls, checklist: QualityChecklist, comments: list[str]) -> QualityScore:
        return cls(total=cls.compute_total(checklist), checklist=checklist, comments=comments)


class ComplianceIssue(BaseModel):
    """Одно найденное нарушение."""

    rule: str = Field(description="Какое правило нарушено")
    quote: str = Field(default="", description="Цитата из разговора")
    severity: Severity = "medium"
    comment: str = Field(default="", description="Пояснение")


class ComplianceResult(BaseModel):
    """Результат compliance-агента."""

    passed: bool
    issues: list[ComplianceIssue] = Field(default_factory=list)


class CallSummary(BaseModel):
    """Результат суммаризатора."""

    summary: str = Field(description="Резюме разговора, 3–5 предложений")
    action_items: list[str] = Field(default_factory=list)


class AgentFailure(BaseModel):
    """Ошибка отдельного агента — не валит весь анализ (graceful degradation)."""

    agent: str
    error: str


class AnalysisMeta(BaseModel):
    """Служебная информация об анализе (аддитивное расширение контракта)."""

    correlation_id: str = ""
    audio_duration_sec: float | None = None
    language: str | None = None
    asr_model: str | None = None
    llm_model: str | None = None
    processing_ms: int | None = None
    agent_failures: list[AgentFailure] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Полный ответ ``POST /analyze`` — структура из ТЗ + ``meta``."""

    transcript: list[TranscriptSegment]
    classification: Classification
    quality_score: QualityScore
    compliance: ComplianceResult
    summary: str
    action_items: list[str] = Field(default_factory=list)
    meta: AnalysisMeta = Field(default_factory=AnalysisMeta)


class TranscriptionResult(BaseModel):
    """Ответ ``POST /transcribe`` (и вход мультиагентного анализа)."""

    segments: list[TranscriptSegment]
    language: str | None = None
    duration_sec: float = 0.0
    asr_model: str | None = None


def format_dialog(segments: list[TranscriptSegment]) -> str:
    """Транскрипт в текстовый диалог для промптов LLM."""
    lines = [
        f"[{seg.start:07.2f}–{seg.end:07.2f}] {seg.speaker}: {seg.text}"
        for seg in segments
        if seg.text
    ]
    return "\n".join(lines)
