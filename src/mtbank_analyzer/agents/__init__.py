"""Мультиагентная аналитика звонков.

Каждый агент - узкая роль с собственным промптом и строгой Pydantic-схемой
ответа. Оркестрация агентов - в ``mtbank_analyzer.orchestration``.
"""

from mtbank_analyzer.agents.base import (
    AgentContext,
    AgentError,
    BaseAgent,
    LLMClient,
    OpenAICompatLLM,
)
from mtbank_analyzer.agents.classifier import ClassifierAgent
from mtbank_analyzer.agents.compliance import ComplianceAgent
from mtbank_analyzer.agents.quality import QualityAgent
from mtbank_analyzer.agents.summarizer import SummarizerAgent

__all__ = [
    "AgentContext",
    "AgentError",
    "BaseAgent",
    "ClassifierAgent",
    "ComplianceAgent",
    "LLMClient",
    "OpenAICompatLLM",
    "QualityAgent",
    "SummarizerAgent",
]
