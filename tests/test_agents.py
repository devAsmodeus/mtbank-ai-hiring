"""Юнит-тесты каждого агента (ТЗ: unit-тест на каждого агента)."""

from __future__ import annotations

import pytest
from tests.conftest import FailingLLM, fake_llm

from mtbank_analyzer.agents import (
    AgentError,
    ClassifierAgent,
    ComplianceAgent,
    QualityAgent,
    SummarizerAgent,
)
from mtbank_analyzer.agents.base import AgentContext, extract_json
from mtbank_analyzer.schemas import OPERATOR, ComplianceIssue, TranscriptSegment

# ---------------------------------------------------------------- classifier


async def test_classifier_parses_valid_response(ctx: AgentContext) -> None:
    llm = fake_llm(
        {"topic": "кредиты", "priority": "medium", "reason": "вопрос о кредите наличными"}
    )
    result = await ClassifierAgent(llm=llm).run(ctx)

    assert result.topic == "кредиты"
    assert result.priority == "medium"
    # транскрипт передан в промпт
    assert "кредит наличными" in llm.calls[0]["user"]


async def test_classifier_retries_on_invalid_enum(ctx: AgentContext) -> None:
    llm = fake_llm(
        {"topic": "ипотека", "priority": "medium", "reason": "x"},  # не из таксономии
        {"topic": "кредиты", "priority": "low", "reason": "y"},
    )
    result = await ClassifierAgent(llm=llm).run(ctx)

    assert result.topic == "кредиты"
    assert len(llm.calls) == 2
    assert "не прошёл валидацию" in llm.calls[1]["user"]


async def test_classifier_fails_after_two_bad_attempts(ctx: AgentContext) -> None:
    llm = fake_llm("не json", "совсем не json")
    with pytest.raises(AgentError, match="classifier"):
        await ClassifierAgent(llm=llm).run(ctx)


async def test_agent_wraps_transport_errors(ctx: AgentContext) -> None:
    with pytest.raises(AgentError, match="недоступна"):
        await ClassifierAgent(llm=FailingLLM()).run(ctx)


# ------------------------------------------------------------------ quality


async def test_quality_total_computed_in_code_not_by_llm(ctx: AgentContext) -> None:
    llm = fake_llm(
        {
            "checklist": {
                "greeting": True,
                "need_detection": True,
                "solution_provided": True,
                "farewell": False,
            },
            "comments": ["Оператор не попрощался"],
        }
    )
    result = await QualityAgent(llm=llm).run(ctx)

    # 20 + 25 + 35 + 0 — арифметика детерминированная
    assert result.total == 80
    assert result.checklist.farewell is False
    assert result.comments == ["Оператор не попрощался"]


async def test_quality_full_checklist_is_100(ctx: AgentContext) -> None:
    llm = fake_llm(
        {
            "checklist": {
                "greeting": True,
                "need_detection": True,
                "solution_provided": True,
                "farewell": True,
            },
            "comments": [],
        }
    )
    result = await QualityAgent(llm=llm).run(ctx)
    assert result.total == 100


# --------------------------------------------------------------- compliance


async def test_compliance_clean_call_passes(ctx: AgentContext) -> None:
    llm = fake_llm({"issues": []})
    result = await ComplianceAgent(llm=llm).run(ctx)

    assert result.passed is True
    assert result.issues == []


async def test_compliance_regex_catches_forbidden_phrase() -> None:
    segments = [
        TranscriptSegment(
            speaker=OPERATOR,
            start=0,
            end=5,
            text="Я вам гарантирую, что кредит одобрят стопроцентно.",
        )
    ]
    ctx = AgentContext.from_segments(segments)
    llm = fake_llm({"issues": []})  # LLM ничего не нашла — regex-контур страхует

    result = await ComplianceAgent(llm=llm).run(ctx)

    assert result.passed is False
    assert any("Гарантия одобрения" in issue.rule for issue in result.issues)


async def test_compliance_regex_ignores_client_lines() -> None:
    segments = [
        TranscriptSegment(
            speaker="Клиент",
            start=0,
            end=5,
            text="А вы гарантируете, что одобрят?",
        )
    ]
    ctx = AgentContext.from_segments(segments)
    llm = fake_llm({"issues": []})

    result = await ComplianceAgent(llm=llm).run(ctx)
    assert result.passed is True


async def test_compliance_merges_llm_and_regex_without_duplicates() -> None:
    quote = "Продиктуйте, пожалуйста, полный номер карты и CVV."
    segments = [TranscriptSegment(speaker=OPERATOR, start=0, end=5, text=quote)]
    ctx = AgentContext.from_segments(segments)
    llm = fake_llm(
        {
            "issues": [
                # дубль regex-находки (та же цитата) — должен быть отброшен
                {"rule": "Запрос CVV", "quote": quote, "severity": "high", "comment": ""},
                # уникальная семантическая находка — должна остаться
                {
                    "rule": "Отсутствие дисклеймера",
                    "quote": "Ставка четырнадцать процентов.",
                    "severity": "medium",
                    "comment": "Не упомянуто, что условия индивидуальны",
                },
            ]
        }
    )

    result = await ComplianceAgent(llm=llm).run(ctx)

    assert result.passed is False
    rules = [issue.rule for issue in result.issues]
    assert "Запрос чувствительных данных карты" in rules  # от regex-контура
    assert "Отсутствие дисклеймера" in rules  # от LLM
    assert "Запрос CVV" not in rules  # дубль отброшен
    assert len(result.issues) == 2


def test_compliance_issue_defaults() -> None:
    issue = ComplianceIssue(rule="x")
    assert issue.severity == "medium"


# --------------------------------------------------------------- summarizer


async def test_summarizer_returns_summary_and_action_items(ctx: AgentContext) -> None:
    llm = fake_llm(
        {
            "summary": "Клиент интересовался кредитом наличными на десять тысяч рублей.",
            "action_items": ["Клиент: подать заявку через приложение"],
        }
    )
    result = await SummarizerAgent(llm=llm).run(ctx)

    assert "кредитом" in result.summary
    assert result.action_items == ["Клиент: подать заявку через приложение"]


# --------------------------------------------------------------------- base


def test_extract_json_strips_markdown_fences() -> None:
    raw = '```json\n{"topic": "карты"}\n```'
    assert extract_json(raw) == '{"topic": "карты"}'


def test_extract_json_finds_object_inside_prose() -> None:
    raw = 'Вот результат:\n{"a": 1}\nНадеюсь, помог!'
    assert extract_json(raw) == '{"a": 1}'


def test_extract_json_raises_when_no_object() -> None:
    with pytest.raises(ValueError):
        extract_json("никакого джейсона тут нет")
