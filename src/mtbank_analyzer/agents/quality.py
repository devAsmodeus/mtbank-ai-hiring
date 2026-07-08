"""Агент качества: чеклист оператора.

LLM выносит семантические суждения (чеклист + замечания), итоговый балл
считается детерминированно кодом (``QualityScore.compute_total``) — оценка
воспроизводима и аудируема, LLM не «выдумывает» числа.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mtbank_analyzer.agents.base import AgentContext, BaseAgent
from mtbank_analyzer.schemas import QualityChecklist, QualityScore

_SYSTEM_PROMPT = """\
Ты — агент оценки качества работы оператора контакт-центра МТБанка.
Оцени работу оператора по чеклисту. Оценивай ТОЛЬКО реплики Оператора и только по фактам
из транскрипта — ничего не додумывай.

Чеклист (true — выполнено, false — нет):
- greeting — оператор поприветствовал клиента и назвал банк и/или представился по имени
- need_detection — оператор выявил потребность: задал уточняющие вопросы (сумма, срок, цель),
  а не ответил формально на первый же вопрос
- solution_provided — оператор решил вопрос по существу: дал полные ответы, предложил
  конкретный следующий шаг (оформить заявку, отправить инструкцию и т.п.)
- farewell — оператор корректно завершил разговор: поблагодарил за обращение и/или попрощался

Транскрипт получен автоматически (ASR) и может содержать ошибки распознавания —
суди по смыслу, а не по точным словам.

В comments дай 1–4 конкретных замечания или сильные стороны разговора
(например: "Оператор не уточнил срок кредита", "Хорошо отработано возражение по страховке").
Пиши по-русски, ссылайся на факты из разговора.

Ответь СТРОГО одним JSON-объектом без markdown и пояснений:
{"checklist": {"greeting": true, "need_detection": true, "solution_provided": true, "farewell": true}, "comments": ["..."]}
"""


class _QualityLLMOutput(BaseModel):
    """Что возвращает LLM: только суждения, без баллов."""

    checklist: QualityChecklist
    comments: list[str] = Field(default_factory=list)


@dataclass
class QualityAgent(BaseAgent[QualityScore]):
    name: str = field(init=False, default="quality")
    llm_output_model: type[_QualityLLMOutput] = field(init=False, default=_QualityLLMOutput)

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def postprocess(self, llm_output: _QualityLLMOutput, ctx: AgentContext) -> QualityScore:
        return QualityScore.from_checklist(llm_output.checklist, llm_output.comments)
