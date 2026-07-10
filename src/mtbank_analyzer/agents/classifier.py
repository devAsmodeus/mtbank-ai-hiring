"""Агент-классификатор: тематика обращения + приоритет.

Системный промпт — в ``prompts/classifier.yaml`` (версионируется отдельно).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mtbank_analyzer.agents.base import BaseAgent
from mtbank_analyzer.schemas import Classification


@dataclass
class ClassifierAgent(BaseAgent[Classification, Classification]):
    name: str = field(init=False, default="classifier")
    llm_output_model: type[Classification] = field(init=False, default=Classification)
