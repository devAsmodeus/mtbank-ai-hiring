"""JSONL-хранилище результатов анализа — источник данных для агента трендов.

Для прототипа осознанно выбран append-only JSONL вместо PostgreSQL:
нулевая инфраструктура, атомарные дозаписи, простая выгрузка. Интерфейс
позволяет заменить бэкенд на БД без изменения вызывающего кода.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from mtbank_analyzer.logging_setup import get_logger
from mtbank_analyzer.schemas import COMPLIANCE_NOT_RUN, AnalysisReport

logger = get_logger(__name__)


class AnalysisStore:
    def __init__(self, directory: Path) -> None:
        self._path = directory / "analyses.jsonl"
        self._lock = asyncio.Lock()

    async def append(self, report: AnalysisReport) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "correlation_id": report.meta.correlation_id,
            "topic": report.classification.topic,
            "priority": report.classification.priority,
            "quality_total": report.quality_score.total,
            "checklist": report.quality_score.checklist.model_dump(),
            "compliance_passed": report.compliance.passed,
            "issues": [
                issue.rule for issue in report.compliance.issues if issue.rule != COMPLIANCE_NOT_RUN
            ],
            "summary": report.summary,
            "duration_sec": report.meta.audio_duration_sec,
        }
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    async def load_recent(self, limit: int = 50) -> list[dict]:
        async with self._lock:
            return await asyncio.to_thread(self._read_tail, limit)

    def _read_tail(self, limit: int) -> list[dict]:
        if not self._path.exists():
            return []
        records: list[dict] = []
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("storage_bad_line_skipped")
        # limit<=0 не должен означать «вся история» (records[-0:] == records)
        return records[-limit:] if limit > 0 else []
