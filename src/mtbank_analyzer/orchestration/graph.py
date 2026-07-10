"""LangGraph-оркестрация: детерминированный граф с параллельным fan-out.

Схема графа::

                    ┌──> classifier ──┐
    START → prepare ├──> quality    ──┼──> aggregate → END
                    ├──> compliance ──┤
                    └──> summarizer ──┘

Решения (обоснование в README):
- Все четыре агента независимы по данным → выполняются параллельно
  (латентность ≈ максимум по агентам, а не сумма).
- Без LLM-супервизора: состав задач фиксирован, LLM-роутинг добавил бы
  латентность, стоимость и недетерминизм без пользы. LangGraph выбран за
  декларативный граф, супершаги с параллелизмом и state-менеджмент.
- Отказ одного агента не валит анализ: ошибка фиксируется в
  ``meta.agent_failures``, для compliance действует fail-closed политика.
"""

from __future__ import annotations

import operator
import uuid
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Annotated, Any, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from mtbank_analyzer.agents import (
    AgentContext,
    AgentError,
    BaseAgent,
    ClassifierAgent,
    ComplianceAgent,
    LLMClient,
    QualityAgent,
    SummarizerAgent,
)
from mtbank_analyzer.agents.compliance import scan_forbidden_phrases
from mtbank_analyzer.logging_setup import get_logger
from mtbank_analyzer.schemas import (
    COMPLIANCE_NOT_RUN,
    AgentFailure,
    AnalysisMeta,
    AnalysisReport,
    CallSummary,
    Classification,
    ComplianceIssue,
    ComplianceResult,
    QualityChecklist,
    QualityScore,
    TranscriptionResult,
)

logger = get_logger(__name__)

_AGENT_NODES = ("classifier", "quality", "compliance", "summarizer")


class GraphState(TypedDict, total=False):
    """Состояние графа. Каждый агент пишет только свой ключ — конфликтов нет,
    ``failures`` объединяется reducer-ом (параллельные записи складываются)."""

    transcription: TranscriptionResult
    ctx: AgentContext
    classification: Classification
    quality: QualityScore
    compliance: ComplianceResult
    call_summary: CallSummary
    failures: Annotated[list[AgentFailure], operator.add]
    report: AnalysisReport


class CallAnalysisOrchestrator:
    """Собирает агентов в граф и предоставляет единую точку входа ``analyze``.

    Используется двумя рантаймами: OpenWebUI Pipeline (чат) и FastAPI (REST).
    """

    def __init__(self, llm: LLMClient, agent_timeout_sec: float = 120.0) -> None:
        self.llm = llm
        self.classifier = ClassifierAgent(llm=llm, timeout_sec=agent_timeout_sec)
        self.quality = QualityAgent(llm=llm, timeout_sec=agent_timeout_sec)
        self.compliance = ComplianceAgent(llm=llm, timeout_sec=agent_timeout_sec)
        self.summarizer = SummarizerAgent(llm=llm, timeout_sec=agent_timeout_sec)
        self._graph = self._build_graph()

    # ------------------------------------------------------------ public API

    async def analyze(
        self,
        transcription: TranscriptionResult,
        correlation_id: str | None = None,
    ) -> AnalysisReport:
        """Прогоняет транскрипт через мультиагентный граф."""
        if not transcription.segments:
            raise ValueError("транскрипт пуст — нечего анализировать")

        correlation_id = correlation_id or uuid.uuid4().hex
        # reset-token, а не unbind: если correlation_id уже привязан вызывающим
        # (HTTP-middleware), по выходе восстановим его значение, а не удалим ключ.
        tokens = structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        started = perf_counter()
        try:
            logger.info(
                "analysis_started",
                segments=len(transcription.segments),
                duration_sec=transcription.duration_sec,
            )
            state: GraphState = await self._graph.ainvoke({"transcription": transcription})
            report: AnalysisReport = state["report"]
            report.meta.correlation_id = correlation_id
            report.meta.llm_model = self.llm.model_name
            logger.info(
                "analysis_finished",
                agents_ms=int((perf_counter() - started) * 1000),
                failures=[f.agent for f in report.meta.agent_failures],
                topic=report.classification.topic,
                quality_total=report.quality_score.total,
                compliance_passed=report.compliance.passed,
            )
            return report
        finally:
            structlog.contextvars.reset_contextvars(**tokens)

    # ------------------------------------------------------------ построение

    def _build_graph(self) -> Any:
        # mypy не выводит generics StateGraph для TypedDict-схемы — работаем как с Any
        graph: Any = StateGraph(GraphState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("classifier", self._agent_node(self.classifier, "classification"))
        graph.add_node("quality", self._agent_node(self.quality, "quality"))
        graph.add_node("compliance", self._agent_node(self.compliance, "compliance"))
        graph.add_node("summarizer", self._agent_node(self.summarizer, "call_summary"))
        graph.add_node("aggregate", self._aggregate)

        graph.add_edge(START, "prepare")
        for node in _AGENT_NODES:
            graph.add_edge("prepare", node)  # fan-out: параллельный супершаг
        graph.add_edge(list(_AGENT_NODES), "aggregate")  # fan-in: ждёт всех
        graph.add_edge("aggregate", END)
        return graph.compile()

    @staticmethod
    def _prepare(state: GraphState) -> dict:
        """Готовит общий контекст для всех агентов."""
        t = state["transcription"]
        ctx = AgentContext.from_segments(
            t.segments, duration_sec=t.duration_sec, language=t.language
        )
        return {"ctx": ctx}

    @staticmethod
    def _agent_node(agent: BaseAgent, result_key: str) -> Callable[[GraphState], Awaitable[dict]]:
        """Узел графа: запуск агента с изоляцией ошибок."""

        async def node(state: GraphState) -> dict:
            try:
                return {result_key: await agent.run(state["ctx"])}
            except AgentError as exc:
                return {"failures": [AgentFailure(agent=agent.name, error=str(exc))]}

        return node

    @staticmethod
    def _aggregate(state: GraphState) -> dict:
        """Собирает итоговый отчёт; для упавших агентов — безопасные фолбэки."""
        ctx = state["ctx"]
        transcription = state["transcription"]
        failures = state.get("failures", [])

        classification = state.get("classification") or Classification(
            topic="другое", priority="medium", reason="классификатор недоступен"
        )
        quality = state.get("quality") or QualityScore(
            total=0,
            checklist=QualityChecklist(
                greeting=False,
                need_detection=False,
                solution_provided=False,
                farewell=False,
            ),
            comments=["Оценка качества недоступна: агент завершился с ошибкой"],
        )
        # Fail-closed: если LLM-контур комплаенса упал, прогоняем хотя бы
        # детерминированные правила и помечаем звонок к ручной проверке.
        compliance = state.get("compliance")
        if compliance is None:
            rule_issues = scan_forbidden_phrases(ctx)
            compliance = ComplianceResult(
                passed=False,
                issues=[
                    *rule_issues,
                    ComplianceIssue(
                        rule=COMPLIANCE_NOT_RUN,
                        severity="high",
                        comment="LLM-контур комплаенса недоступен — требуется ручная проверка",
                    ),
                ],
            )
        call_summary = state.get("call_summary") or CallSummary(
            summary="Автоматическое резюме недоступно: агент завершился с ошибкой.",
            action_items=[],
        )

        report = AnalysisReport(
            transcript=transcription.segments,
            classification=classification,
            quality_score=quality,
            compliance=compliance,
            summary=call_summary.summary,
            action_items=call_summary.action_items,
            meta=AnalysisMeta(
                audio_duration_sec=transcription.duration_sec,
                language=transcription.language,
                asr_model=transcription.asr_model,
                agent_failures=failures,
            ),
        )
        return {"report": report}
