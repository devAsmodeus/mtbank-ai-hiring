"""Compliance-агент: гибрид детерминированных правил и LLM-проверки.

Два контура:
1. Регулярные стоп-фразы из конфига (``rules/compliance.yaml``) - высокоточные,
   дешёвые, не зависят от LLM.
2. LLM - семантические нарушения, которые правилами не поймать
   (отсутствие дисклеймера, давление на клиента, грубость).

Результаты объединяются, ``passed = нет ни одного нарушения``.
Системный промпт - в ``prompts/compliance.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mtbank_analyzer.agents.base import AgentContext, BaseAgent
from mtbank_analyzer.rules import get_compliance_rules
from mtbank_analyzer.schemas import OPERATOR, ComplianceIssue, ComplianceResult


class _ComplianceLLMOutput(BaseModel):
    issues: list[ComplianceIssue] = Field(default_factory=list)


def scan_forbidden_phrases(ctx: AgentContext) -> list[ComplianceIssue]:
    """Детерминированный контур: прогон стоп-фраз по репликам оператора."""
    issues: list[ComplianceIssue] = []
    for seg in ctx.segments:
        if seg.speaker != OPERATOR:
            continue
        for rule in get_compliance_rules():
            if rule.pattern.search(seg.text):
                issues.append(
                    ComplianceIssue(
                        rule=rule.rule,
                        quote=seg.text,
                        severity=rule.severity,
                        comment="Сработало детерминированное правило (regex-контур)",
                    )
                )
    return issues


@dataclass
class ComplianceAgent(BaseAgent[_ComplianceLLMOutput, ComplianceResult]):
    name: str = field(init=False, default="compliance")
    llm_output_model: type[_ComplianceLLMOutput] = field(init=False, default=_ComplianceLLMOutput)

    def postprocess(self, llm_output: _ComplianceLLMOutput, ctx: AgentContext) -> ComplianceResult:
        rule_issues = scan_forbidden_phrases(ctx)
        # Отбрасываем LLM-находку, только если её цитата ТОЧНО совпадает с цитатой
        # regex-контура (то же нарушение). Другое нарушение с цитатой из той же
        # реплики (например, отсутствие дисклеймера) сохраняется.
        rule_quotes = {issue.quote.strip() for issue in rule_issues if issue.quote}
        llm_issues = [
            issue for issue in llm_output.issues if issue.quote.strip() not in rule_quotes
        ]
        issues = rule_issues + llm_issues
        return ComplianceResult(passed=not issues, issues=issues)
