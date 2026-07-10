"""Тесты валидации конфигурации и устойчивости логирования."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mtbank_analyzer.config import Settings
from mtbank_analyzer.logging_setup import configure_logging


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_blank_whisper_language_means_auto() -> None:
    # пустая строка из env → автоопределение (None), а не невалидный код языка
    assert _settings(whisper_language="").whisper_language is None
    assert _settings(whisper_language="   ").whisper_language is None


def test_explicit_whisper_language_kept() -> None:
    assert _settings(whisper_language="en").whisper_language == "en"


def test_log_level_is_normalized() -> None:
    assert _settings(log_level="debug").log_level == "DEBUG"


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="LOG_LEVEL"):
        _settings(log_level="trace")


def test_configure_logging_survives_unknown_level() -> None:
    # прямой вызов с мусорным уровнем не должен ронять процесс (фолбэк на INFO)
    configure_logging("trace")
    configure_logging("INFO")  # вернуть штатный уровень для остальных тестов
