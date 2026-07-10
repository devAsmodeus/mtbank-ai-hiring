"""Compliance-агент: гибрид детерминированных правил и LLM-проверки.

Два контура:
1. Регулярные выражения — высокоточные «стоп-фразы» (гарантии одобрения,
   запрос CVV/ПИН). Дешёвые, детерминированные, не зависят от LLM.
2. LLM — семантические нарушения, которые правилами не поймать
   (отсутствие дисклеймера, давление на клиента, грубость).

Результаты объединяются, ``passed = нет ни одного нарушения``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mtbank_analyzer.agents.base import AgentContext, BaseAgent
from mtbank_analyzer.schemas import OPERATOR, ComplianceIssue, ComplianceResult, Severity

_SYSTEM_PROMPT = """\
Ты — compliance-агент контакт-центра МТБанка. Проверь реплики Оператора на нарушения
регуляторных и внутренних правил банка. Реплики Клиента нарушением не являются.

Типы нарушений:
1. Запрещённые обещания: гарантии одобрения кредита ("гарантирую", "стопроцентно одобрят"),
   обещания доходности, заявления "самые выгодные условия на рынке" без оговорок.
2. Запрос чувствительных данных: оператор просит продиктовать полный номер карты, CVV/CVC,
   ПИН-код, пароль или код из СМС, логин/пароль интернет-банка.
3. Отсутствие обязательного дисклеймера: оператор назвал конкретную ставку или платёж
   по кредитному продукту, но нигде не упомянул, что точные условия рассчитываются
   индивидуально / окончательное решение принимает банк.
4. Некорректное давление: навязывание платных услуг без согласия, манипуляции
   ("предложение сгорит через час"), оформление без явного согласия клиента.
5. Грубость, пренебрежительный тон, переход на личности.

Правила:
- Для каждого нарушения приводи ТОЧНУЮ цитату реплики Оператора из транскрипта.
- Не выдумывай нарушения. Сомневаешься — не включай.
- Если нарушений нет, верни пустой список issues.
- severity: "high" — запрос чувствительных данных, грубость; "medium" — обещания
  и отсутствие дисклеймеров; "low" — стилистические недочёты.

Ответь СТРОГО одним JSON-объектом без markdown и пояснений:
{"issues": [{"rule": "краткое название правила", "quote": "цитата", "severity": "low|medium|high", "comment": "пояснение"}]}
"""

# Высокоточные стоп-фразы: (регэксп, название правила, severity)
FORBIDDEN_PATTERNS: list[tuple[re.Pattern[str], str, Severity]] = [
    (
        re.compile(
            r"гарантир\w*\s+(?:\w+\s+){0,3}?одобр|одобр\w*\s+стопроцентн|стопроцентн\w*\s+одобр",
            re.IGNORECASE,
        ),
        "Гарантия одобрения кредита",
        "medium",
    ),
    (
        re.compile(
            r"(?:продиктуйте|назовите|скажите|сообщите)[^.!?]{0,60}?(?:cvv|cvc|пин[\s-]?код|полный номер карты|пароль)",
            re.IGNORECASE,
        ),
        "Запрос чувствительных данных карты",
        "high",
    ),
    (
        re.compile(
            r"(?:код|пароль)\s+из\s+смс[^.!?]{0,40}?(?:продиктуйте|назовите|скажите|сообщите)|(?:продиктуйте|назовите|скажите|сообщите)[^.!?]{0,40}?(?:код|пароль)\s+из\s+смс",
            re.IGNORECASE,
        ),
        "Запрос кода из СМС",
        "high",
    ),
    (
        re.compile(
            r"предложение\s+(?:сгорит|действует\s+только\s+сегодня|исчезнет)", re.IGNORECASE
        ),
        "Манипулятивное давление на клиента",
        "medium",
    ),
]


class _ComplianceLLMOutput(BaseModel):
    issues: list[ComplianceIssue] = Field(default_factory=list)


def scan_forbidden_phrases(ctx: AgentContext) -> list[ComplianceIssue]:
    """Детерминированный контур: прогон стоп-фраз по репликам оператора."""
    issues: list[ComplianceIssue] = []
    for seg in ctx.segments:
        if seg.speaker != OPERATOR:
            continue
        for pattern, rule, severity in FORBIDDEN_PATTERNS:
            if pattern.search(seg.text):
                issues.append(
                    ComplianceIssue(
                        rule=rule,
                        quote=seg.text,
                        severity=severity,
                        comment="Сработало детерминированное правило (regex-контур)",
                    )
                )
    return issues


@dataclass
class ComplianceAgent(BaseAgent[_ComplianceLLMOutput, ComplianceResult]):
    name: str = field(init=False, default="compliance")
    llm_output_model: type[_ComplianceLLMOutput] = field(init=False, default=_ComplianceLLMOutput)

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def postprocess(self, llm_output: _ComplianceLLMOutput, ctx: AgentContext) -> ComplianceResult:
        rule_issues = scan_forbidden_phrases(ctx)
        # LLM-находки, которые лишь дублируют цитаты regex-контура, отбрасываем
        rule_quotes = {issue.quote for issue in rule_issues}
        llm_issues = [
            issue
            for issue in llm_output.issues
            if not any(issue.quote and issue.quote in quote for quote in rule_quotes)
        ]
        issues = rule_issues + llm_issues
        return ComplianceResult(passed=not issues, issues=issues)
