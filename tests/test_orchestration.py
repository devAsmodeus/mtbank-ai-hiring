"""Интеграционные тесты LangGraph-оркестрации (4 агента + агрегация)."""

from __future__ import annotations

import pytest
from tests.conftest import GOOD_RESPONSES, RoutingFakeLLM

from mtbank_analyzer.orchestration import CallAnalysisOrchestrator
from mtbank_analyzer.schemas import TranscriptionResult, TranscriptSegment


@pytest.fixture
def transcription(segments: list[TranscriptSegment]) -> TranscriptionResult:
    return TranscriptionResult(
        segments=segments, language="ru", duration_sec=28.0, asr_model="large-v3-turbo"
    )


async def test_full_graph_assembles_report(transcription: TranscriptionResult) -> None:
    llm = RoutingFakeLLM(dict(GOOD_RESPONSES))
    orchestrator = CallAnalysisOrchestrator(llm=llm)

    report = await orchestrator.analyze(transcription, correlation_id="test-123")

    # все четыре агента отработали
    assert sorted(llm.calls) == ["classifier", "compliance", "quality", "summarizer"]
    # контракт ТЗ собран полностью
    assert report.classification.topic == "кредиты"
    assert report.quality_score.total == 100
    assert report.compliance.passed is True
    assert "кредита" in report.summary
    assert report.action_items == ["Оператор: отправить инструкцию на email"]
    assert report.transcript == transcription.segments
    # метаданные
    assert report.meta.correlation_id == "test-123"
    assert report.meta.llm_model == "routing-fake"
    assert report.meta.asr_model == "large-v3-turbo"
    assert report.meta.agent_failures == []


async def test_failed_agent_degrades_gracefully(transcription: TranscriptionResult) -> None:
    responses = dict(GOOD_RESPONSES)
    responses["classifier"] = "__RAISE__"
    orchestrator = CallAnalysisOrchestrator(llm=RoutingFakeLLM(responses))

    report = await orchestrator.analyze(transcription)

    # анализ не упал, классификация - безопасный фолбэк
    assert report.classification.topic == "другое"
    assert [f.agent for f in report.meta.agent_failures] == ["classifier"]
    # остальные агенты отработали
    assert report.quality_score.total == 100
    assert report.compliance.passed is True


async def test_compliance_failure_is_fail_closed(transcription: TranscriptionResult) -> None:
    responses = dict(GOOD_RESPONSES)
    responses["compliance"] = "__RAISE__"
    orchestrator = CallAnalysisOrchestrator(llm=RoutingFakeLLM(responses))

    report = await orchestrator.analyze(transcription)

    # комплаенс недоступен → звонок помечается к ручной проверке, а не "passed"
    assert report.compliance.passed is False
    assert any(i.rule == "Проверка не выполнена" for i in report.compliance.issues)
    assert [f.agent for f in report.meta.agent_failures] == ["compliance"]


async def test_empty_transcript_is_rejected() -> None:
    orchestrator = CallAnalysisOrchestrator(llm=RoutingFakeLLM(dict(GOOD_RESPONSES)))
    with pytest.raises(ValueError, match="пуст"):
        await orchestrator.analyze(TranscriptionResult(segments=[]))
