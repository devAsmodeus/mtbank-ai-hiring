"""Агент-суммаризатор: резюме звонка + action items.

Системный промпт — в ``prompts/summarizer.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mtbank_analyzer.agents.base import BaseAgent
from mtbank_analyzer.schemas import CallSummary


@dataclass
class SummarizerAgent(BaseAgent[CallSummary, CallSummary]):
    name: str = field(init=False, default="summarizer")
    llm_output_model: type[CallSummary] = field(init=False, default=CallSummary)
