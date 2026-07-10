"""Pydantic-модели: контракт данных всей системы.

JSON-контракт ответа ``/analyze`` зафиксирован в ТЗ — модели ниже воспроизводят
его 1:1 (поле ``meta`` — дополнительное, аддитивное расширение).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Канонические говорящие. Диаризация всегда приводит спикеров к этим ролям.
OPERATOR = "Оператор"
CLIENT = "Клиент"

# Служебное «нарушение», которым помечается звонок, когда LLM-контур комплаенса
# недоступен. В отчёте оно нужно (сигнал ручной проверки), но в агрегаты трендов
# не идёт — иначе завышает долю нарушений и попадает в топ «частых».
COMPLIANCE_NOT_RUN = "Проверка не выполнена"

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

    LLM отвечает за семантические суждения (было ли приветствие), арифметика
    детерминированная, чтобы балл был воспроизводимым и аудируемым. Веса пунктов
    — бизнес-политика, живут в конфиге (``rules/quality_weights.yaml``), а не в
    контракте данных, и передаются извне.
    """

    total: int = Field(ge=0, le=100)
    checklist: QualityChecklist
    comments: list[str] = Field(default_factory=list, description="Замечания по разговору")

    @staticmethod
    def compute_total(checklist: QualityChecklist, weights: dict[str, int]) -> int:
        return sum(w for item, w in weights.items() if getattr(checklist, item, False))

    @classmethod
    def from_checklist(
        cls, checklist: QualityChecklist, comments: list[str], weights: dict[str, int]
    ) -> QualityScore:
        return cls(
            total=cls.compute_total(checklist, weights), checklist=checklist, comments=comments
        )


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
    #: версия промпта каждого агента — основа для A/B и разбора регрессий
    prompt_versions: dict[str, str] = Field(default_factory=dict)
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


class AnalysisRecord(BaseModel):
    """Запись анализа в хранилище — типизированный источник данных для трендов.

    Отделена от ``AnalysisReport``: хранит только агрегируемые поля, а не весь
    транскрипт. Замена бэкенда (Postgres) работает с этой моделью, а не с
    рукописным dict.
    """

    ts: str
    correlation_id: str = ""
    topic: str
    priority: str
    quality_total: int
    checklist: QualityChecklist
    compliance_passed: bool
    issues: list[str] = Field(default_factory=list)
    summary: str = ""
    duration_sec: float | None = None

    @classmethod
    def from_report(cls, report: AnalysisReport, ts: str) -> AnalysisRecord:
        return cls(
            ts=ts,
            correlation_id=report.meta.correlation_id,
            topic=report.classification.topic,
            priority=report.classification.priority,
            quality_total=report.quality_score.total,
            checklist=report.quality_score.checklist,
            compliance_passed=report.compliance.passed,
            # служебное псевдо-нарушение в агрегаты трендов не идёт
            issues=[i.rule for i in report.compliance.issues if i.rule != COMPLIANCE_NOT_RUN],
            summary=report.summary,
            duration_sec=report.meta.audio_duration_sec,
        )


def format_dialog(segments: list[TranscriptSegment]) -> str:
    """Транскрипт в текстовый диалог для промптов LLM."""
    lines = [
        f"[{seg.start:07.2f}–{seg.end:07.2f}] {seg.speaker}: {seg.text}"
        for seg in segments
        if seg.text
    ]
    return "\n".join(lines)
