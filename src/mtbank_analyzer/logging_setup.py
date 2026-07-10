"""Структурированное JSON-логирование (structlog).

Требование ТЗ: JSON-логи с входом/выходом каждого агента.
Настраивает structlog и перехватывает stdlib-логи (uvicorn, httpx и т.д.),
чтобы весь вывод сервиса был единым JSON-потоком.
"""

from __future__ import annotations

import logging
import sys

import structlog


def _resolve_level(level: str) -> int:
    """Имя уровня → числовой код; неизвестное имя не роняет процесс, а даёт INFO.

    Settings уже валидирует LOG_LEVEL, но configure_logging может вызываться и
    напрямую (pipeline.py), поэтому подстраховываемся здесь.
    """
    return logging.getLevelNamesMapping().get(level.strip().upper(), logging.INFO)


def configure_logging(level: str = "INFO") -> None:
    """Идемпотентная настройка JSON-логирования для всего процесса."""
    numeric_level = _resolve_level(level)
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # stdlib-логи (uvicorn.access и пр.) - в тот же JSON-формат
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger
