"""Хранилище результатов анализа - источник данных для агента трендов.

Интерфейс ``AnalysisStorage`` (Protocol) отделяет вызывающий код от бэкенда:
JSONL-реализация ниже хватает для прототипа, а на её место можно поставить
Postgres/MinIO, не трогая routes и агент трендов. Записи типизированы
(``AnalysisRecord``), а не рукописный dict.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from mtbank_analyzer.logging_setup import get_logger
from mtbank_analyzer.schemas import AnalysisRecord, AnalysisReport

logger = get_logger(__name__)


class AnalysisStorage(Protocol):
    """Контракт хранилища анализов (для DI и подмены бэкенда)."""

    async def append(self, report: AnalysisReport) -> None: ...

    async def load_recent(self, limit: int = 50) -> list[AnalysisRecord]: ...


class JsonlAnalysisStore:
    """Append-only JSONL: нулевая инфраструктура, атомарные дозаписи."""

    def __init__(self, directory: Path) -> None:
        self._path = directory / "analyses.jsonl"

    async def append(self, report: AnalysisReport) -> None:
        record = AnalysisRecord.from_report(report, ts=datetime.now(UTC).isoformat())
        await asyncio.to_thread(self._append_line, record.model_dump_json())

    def _append_line(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    async def load_recent(self, limit: int = 50) -> list[AnalysisRecord]:
        return await asyncio.to_thread(self._read_tail, limit)

    def _read_tail(self, limit: int) -> list[AnalysisRecord]:
        if limit <= 0 or not self._path.exists():
            return []
        # deque(maxlen) держит в памяти только хвост, не весь файл
        with self._path.open(encoding="utf-8") as fh:
            tail = deque(fh, maxlen=limit)
        records: list[AnalysisRecord] = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(AnalysisRecord.model_validate_json(line))
            except ValidationError:
                logger.warning("storage_bad_line_skipped")
        return records


# Обратная совместимость имени, использованного в остальном коде.
AnalysisStore = JsonlAnalysisStore
