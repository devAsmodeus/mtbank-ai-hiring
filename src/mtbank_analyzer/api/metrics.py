"""Prometheus-метрики для дашборда аналитиков контакт-центра."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

ANALYZE_REQUESTS = Counter(
    "analyze_requests_total",
    "Запросы POST /analyze по статусу ответа",
    ["status"],
)
ANALYZE_DURATION = Histogram(
    "analyze_duration_seconds",
    "Полное время анализа звонка (ASR + агенты)",
    buckets=(1, 5, 10, 20, 30, 45, 60, 90, 120, 180),
)
ASR_DURATION = Histogram(
    "asr_duration_seconds",
    "Время транскрибации",
    buckets=(1, 5, 10, 20, 30, 45, 60, 90, 120),
)
AGENTS_DURATION = Histogram(
    "agents_duration_seconds",
    "Время мультиагентного анализа",
    buckets=(1, 2, 5, 10, 20, 30, 60, 120),
)
CALLS_ANALYZED = Counter("calls_analyzed_total", "Проанализированные звонки")
CALL_TOPICS = Counter("calls_by_topic_total", "Звонки по тематикам", ["topic"])
QUALITY_SCORE = Histogram(
    "call_quality_score",
    "Распределение quality_score",
    buckets=(0, 20, 40, 55, 65, 80, 90, 100),
)
COMPLIANCE_FAILED = Counter("compliance_failed_total", "Звонки с compliance-нарушениями")


def observe_report(topic: str, quality_total: int, compliance_passed: bool) -> None:
    """Единая точка записи бизнес-метрик по завершённому анализу."""
    CALLS_ANALYZED.inc()
    CALL_TOPICS.labels(topic=topic).inc()
    QUALITY_SCORE.observe(quality_total)
    if not compliance_passed:
        COMPLIANCE_FAILED.inc()
