"""Офлайн-проверка eval-слоя: парсинг golden-set, детерминированный compliance.

Полный прогон агентов через LLM — в scripts/evaluate_agents.py; здесь без сети
проверяется то, что регрессирует чаще всего: compliance-правила на реалистичных
диалогах и структура эталонного набора.
"""

from __future__ import annotations

from mtbank_analyzer.agents.base import AgentContext
from mtbank_analyzer.agents.compliance import scan_forbidden_phrases
from mtbank_analyzer.eval import EvalMetrics, load_golden_set


def test_golden_set_loads() -> None:
    cases = {c.id: c for c in load_golden_set()}
    assert len(cases) >= 4
    assert "card_cvv_violation" in cases
    assert cases["card_cvv_violation"].expected["compliance_passed"] is False


def test_compliance_rules_on_golden_set() -> None:
    cases = {c.id: c for c in load_golden_set()}

    # нарушение ловится детерминированным контуром
    violation_ctx = AgentContext.from_segments(cases["card_cvv_violation"].segments)
    rules = [issue.rule for issue in scan_forbidden_phrases(violation_ctx)]
    assert "Запрос чувствительных данных карты" in rules

    # чистые звонки не дают ложных срабатываний
    for clean_id in ("credit_ok", "card_block_complaint", "deposit_info"):
        ctx = AgentContext.from_segments(cases[clean_id].segments)
        assert scan_forbidden_phrases(ctx) == [], clean_id


def test_eval_metrics_thresholds() -> None:
    good = EvalMetrics(
        topic_hits=4,
        topic_total=4,
        checklist_hits=8,
        checklist_total=8,
        compliance_hits=4,
        compliance_total=4,
    )
    assert good.meets()

    poor = EvalMetrics(
        topic_hits=1,
        topic_total=4,
        checklist_hits=4,
        checklist_total=8,
        compliance_hits=2,
        compliance_total=4,
    )
    assert not poor.meets()
