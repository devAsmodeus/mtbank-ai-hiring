"""Конфигурация через переменные окружения / .env (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Settings(BaseSettings):
    """Все настройки системы. Значения по умолчанию - для docker compose."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (любой OpenAI-совместимый эндпоинт) ---
    llm_base_url: str = Field(
        default="http://ollama:11434/v1",
        description="OpenAI-совместимый эндпоинт: Ollama / vLLM / Groq / OpenRouter",
    )
    llm_api_key: str = Field(default="ollama", description="API-ключ (для Ollama - любой)")
    llm_model: str = Field(default="qwen2.5:7b", description="Имя модели у провайдера")
    llm_temperature: float = 0.1
    llm_timeout_sec: float = 90.0
    llm_json_mode: bool = Field(
        default=True,
        description="Просить response_format=json_object (поддержан Ollama/Groq/OpenRouter)",
    )
    agent_timeout_sec: float = Field(default=120.0, description="Жёсткий таймаут одного агента")

    # --- ASR ---
    whisper_model: str = Field(
        default="large-v3-turbo",
        description="Модель faster-whisper: large-v3-turbo быстрее и точнее medium",
    )
    whisper_device: str = Field(default="auto", description="auto | cpu | cuda")
    whisper_compute_type: str = Field(
        default="auto", description="auto → int8 на CPU, float16 на GPU"
    )
    whisper_beam_size: int = 2
    whisper_language: str | None = Field(default="ru", description="None - автоопределение языка")

    # --- Диаризация ---
    diarization_enabled: bool = True

    # --- Analysis Engine (FastAPI) ---
    max_upload_mb: int = 50
    max_audio_duration_sec: float = 1800.0
    storage_dir: Path = Field(
        default=Path("data"), description="Каталог JSONL-хранилища анализов (для трендов)"
    )

    # --- Логирование ---
    log_level: str = "INFO"

    @field_validator("whisper_language", mode="before")
    @classmethod
    def _blank_language_means_auto(cls, value: str | None) -> str | None:
        # Через env нельзя передать None: пустая строка WHISPER_LANGUAGE= означает
        # автоопределение языка (иначе '' уходит в whisper как невалидный код).
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"LOG_LEVEL={value!r} не поддерживается; допустимо: "
                f"{', '.join(sorted(_VALID_LOG_LEVELS))}"
            )
        return level
