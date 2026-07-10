"""Агент трендов (бонус): анализ нескольких звонков, выявление паттернов.

Агрегаты (распределение тем, средний балл, доля нарушений) считает код —
детерминированно; LLM интерпретирует их и резюме звонков в паттерны
и рекомендации для супервайзера.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mtbank_analyzer.agents.base import AgentContext, BaseAgent

_SYSTEM_PROMPT = """\
Ты — аналитик контакт-центра МТБанка. Тебе дана сводка по последним звонкам:
агрегированная статистика и краткие резюме разговоров.

Твоя задача — выявить паттерны и дать рекомендации супервайзеру:
- patterns: 2–5 наблюдений о повторяющихся темах, проблемах клиентов, типичных
  ошибках операторов или compliance-рисках. Только то, что подтверждается данными.
- recommendations: 2–4 конкретных действия для улучшения работы контакт-центра
  (обучение операторов, изменение скриптов, эскалации).

Пиши по-русски, деловым стилем, без воды. Не выдумывай факты, которых нет в сводке.

Ответь СТРОГО одним JSON-объектом без markdown:
{"patterns": ["..."], "recommendations": ["..."]}
"""


class TrendsInsights(BaseModel):
    patterns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class TrendsReport(BaseModel):
    """Ответ GET /trends."""

    calls_analyzed: int
    topics: dict[str, int]
    avg_quality: float
    compliance_violation_rate: float
    frequent_issues: dict[str, int]
    patterns: list[str]
    recommendations: list[str]


@dataclass
class TrendsAgent(BaseAgent[TrendsInsights, TrendsInsights]):
    name: str = field(init=False, default="trends")
    llm_output_model: type[TrendsInsights] = field(init=False, default=TrendsInsights)

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_user_prompt(self, ctx: AgentContext) -> str:
        # ctx.dialog здесь — текстовая сводка по звонкам (см. build_trends_report)
        return ctx.dialog


def aggregate_records(records: list[dict]) -> tuple[dict, str]:
    """Считает агрегаты по записям хранилища и собирает сводку для LLM."""
    topics = Counter(r.get("topic", "другое") for r in records)
    qualities = [r["quality_total"] for r in records if r.get("quality_total") is not None]
    avg_quality = round(sum(qualities) / len(qualities), 1) if qualities else 0.0
    violations = sum(1 for r in records if r.get("compliance_passed") is False)
    violation_rate = round(violations / len(records), 3) if records else 0.0
    issues = Counter(issue for r in records for issue in r.get("issues", []))

    stats = {
        "calls_analyzed": len(records),
        "topics": dict(topics.most_common()),
        "avg_quality": avg_quality,
        "compliance_violation_rate": violation_rate,
        "frequent_issues": dict(issues.most_common(10)),
    }

    lines = [
        f"Всего звонков: {len(records)}",
        f"Распределение тем: {dict(topics.most_common())}",
        f"Средний балл качества: {avg_quality} из 100",
        f"Доля звонков с compliance-нарушениями: {violation_rate:.0%}",
        f"Частые нарушения: {dict(issues.most_common(5)) or 'нет'}",
        "",
        "Резюме звонков (от старых к новым):",
    ]
    lines += [
        f"{i}. [{r.get('topic', '?')}, качество {r.get('quality_total', '?')}] "
        f"{r.get('summary', '')}"
        for i, r in enumerate(records, 1)
    ]
    return stats, "\n".join(lines)


async def build_trends_report(agent: TrendsAgent, records: list[dict]) -> TrendsReport:
    """Агрегаты кода + инсайты LLM → итоговый отчёт по трендам."""
    stats, summary_text = aggregate_records(records)
    ctx = AgentContext(segments=[], dialog=summary_text)
    insights = await agent.run(ctx)
    return TrendsReport(
        **stats,
        patterns=insights.patterns,
        recommendations=insights.recommendations,
    )
