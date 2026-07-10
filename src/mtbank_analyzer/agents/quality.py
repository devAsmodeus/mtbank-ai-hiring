"""Агент качества: чеклист оператора.

LLM выносит семантические суждения (чеклист + замечания), итоговый балл
считается детерминированно кодом (``QualityScore.compute_total``) — оценка
воспроизводима и аудируема, LLM не «выдумывает» числа.
Системный промпт — в ``prompts/quality.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mtbank_analyzer.agents.base import AgentContext, BaseAgent
from mtbank_analyzer.rules import get_quality_weights
from mtbank_analyzer.schemas import QualityChecklist, QualityScore


class _QualityLLMOutput(BaseModel):
    """Что возвращает LLM: только суждения, без баллов."""

    checklist: QualityChecklist
    comments: list[str] = Field(default_factory=list)


@dataclass
class QualityAgent(BaseAgent[_QualityLLMOutput, QualityScore]):
    name: str = field(init=False, default="quality")
    llm_output_model: type[_QualityLLMOutput] = field(init=False, default=_QualityLLMOutput)

    def postprocess(self, llm_output: _QualityLLMOutput, ctx: AgentContext) -> QualityScore:
        return QualityScore.from_checklist(
            llm_output.checklist, llm_output.comments, get_quality_weights()
        )
