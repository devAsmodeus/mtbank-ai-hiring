"""Базовый агент: вызов LLM, строгая валидация JSON, ретрай, JSON-логирование.

Контракт агента:
- вход — ``AgentContext`` (диалог + сегменты + метаданные);
- выход — Pydantic-модель (``llm_output_model`` → ``postprocess`` → результат);
- при невалидном JSON — один повтор с текстом ошибки валидации в промпте;
- вход и выход каждого запуска логируются в JSON (требование ТЗ).
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC
from dataclasses import dataclass, field
from time import perf_counter
from typing import Generic, Protocol, TypeVar, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from mtbank_analyzer.config import Settings
from mtbank_analyzer.logging_setup import get_logger
from mtbank_analyzer.prompts import Prompt, get_prompt_registry
from mtbank_analyzer.schemas import TranscriptSegment, format_dialog

logger = get_logger(__name__)

#: схема, которую возвращает LLM
TLLMOut = TypeVar("TLLMOut", bound=BaseModel)
#: итоговый результат агента (может отличаться после postprocess)
TResult = TypeVar("TResult", bound=BaseModel)

_MAX_ATTEMPTS = 2


class AgentError(RuntimeError):
    """Агент не смог получить валидный результат."""

    def __init__(self, agent: str, message: str) -> None:
        self.agent = agent
        super().__init__(f"[{agent}] {message}")


class LLMClient(Protocol):
    """Минимальный интерфейс LLM — позволяет подменять клиента в тестах."""

    model_name: str

    async def complete(self, *, system: str, user: str) -> str: ...


class OpenAICompatLLM:
    """Клиент любого OpenAI-совместимого эндпоинта (Ollama/vLLM/Groq/OpenRouter).

    JSON-mode (``response_format={"type": "json_object"}``) повышает долю валидных
    ответов, но поддержан не всеми провайдерами — при ошибке клиент один раз
    повторяет запрос без JSON-mode и запоминает это.
    """

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.llm_model
        self._plain = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=SecretStr(settings.llm_api_key),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_sec,
            max_retries=2,
        )
        self._json = (
            self._plain.bind(response_format={"type": "json_object"})
            if settings.llm_json_mode
            else None
        )
        self._json_mode_broken = False

    async def complete(self, *, system: str, user: str) -> str:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        client = (
            self._json if self._json is not None and not self._json_mode_broken else self._plain
        )
        try:
            response = await client.ainvoke(messages)
        except Exception as exc:
            # JSON-mode отключаем НАВСЕГДА только если провайдер явно его не принял
            # (клиентская ошибка 400/404/422). Транзиентную ошибку (429, 5xx, сеть)
            # пробрасываем — иначе один rate-limit калечит JSON-mode на весь процесс.
            if client is self._json and _looks_like_json_mode_unsupported(exc):
                logger.warning("llm_json_mode_unsupported", model=self.model_name)
                self._json_mode_broken = True
                response = await self._plain.ainvoke(messages)
            else:
                raise
        return _content_to_text(response.content)


def _looks_like_json_mode_unsupported(exc: Exception) -> bool:
    """Похоже ли исключение на «провайдер не поддержал response_format».

    Клиентские статусы 400/404/422 означают отказ обработать сам запрос; 429,
    5xx и сетевые ошибки транзиентны и НЕ должны отключать JSON-mode.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (400, 404, 422)


def _content_to_text(content: str | list) -> str:
    """LangChain может вернуть список блоков — склеиваем текстовые части."""
    if isinstance(content, str):
        return content
    parts = [block if isinstance(block, str) else block.get("text", "") for block in content]
    return "".join(parts)


def extract_json(text: str) -> str:
    """Достаёт JSON-объект из ответа LLM (сносит markdown-ограждения и преамбулы)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("в ответе LLM не найден JSON-объект")
    return text[start : end + 1]


@dataclass(frozen=True)
class AgentContext:
    """Вход всех агентов: подготовленный транскрипт и метаданные звонка."""

    segments: list[TranscriptSegment]
    dialog: str
    duration_sec: float = 0.0
    language: str | None = None

    @classmethod
    def from_segments(
        cls,
        segments: list[TranscriptSegment],
        duration_sec: float = 0.0,
        language: str | None = None,
    ) -> AgentContext:
        return cls(
            segments=segments,
            dialog=format_dialog(segments),
            duration_sec=duration_sec,
            language=language,
        )


@dataclass
class BaseAgent(ABC, Generic[TLLMOut, TResult]):
    """Скелет агента: prompt → LLM → JSON → Pydantic → postprocess."""

    llm: LLMClient
    timeout_sec: float = 120.0
    prompt: Prompt = field(init=False)
    name: str = field(init=False, default="agent")

    #: Схема, которую обязан вернуть LLM (задаётся в наследнике)
    llm_output_model: type[TLLMOut] = field(init=False)

    def __post_init__(self) -> None:
        # системный промпт и его версия берутся из реестра по имени агента
        self.prompt = get_prompt_registry().get(self.name)

    @property
    def system_prompt(self) -> str:
        return self.prompt.system

    def build_user_prompt(self, ctx: AgentContext) -> str:
        return (
            f"Транскрипт телефонного звонка "
            f"(длительность {ctx.duration_sec:.0f} сек, "
            f"формат реплик: [начало–конец] Роль: текст):\n\n{ctx.dialog}"
        )

    def postprocess(self, llm_output: TLLMOut, ctx: AgentContext) -> TResult:
        """По умолчанию ответ LLM и есть результат агента (TLLMOut == TResult)."""
        return cast(TResult, llm_output)

    async def run(self, ctx: AgentContext) -> TResult:
        log = logger.bind(
            agent=self.name,
            prompt_version=self.prompt.version,
            llm_model=self.llm.model_name,
        )
        user_prompt = self.build_user_prompt(ctx)
        log.info("agent_input", input=user_prompt)

        started = perf_counter()
        prompt = user_prompt
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                async with asyncio.timeout(self.timeout_sec):
                    raw = await self.llm.complete(system=self.system_prompt, user=prompt)
                llm_output = self.llm_output_model.model_validate_json(extract_json(raw))
                result: TResult = self.postprocess(llm_output, ctx)
                log.info(
                    "agent_output",
                    output=result.model_dump(),
                    attempt=attempt,
                    latency_ms=int((perf_counter() - started) * 1000),
                )
                return result
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                # Невалидный JSON — даём LLM одну попытку исправиться
                last_error = exc
                log.warning("agent_invalid_output", error=str(exc), attempt=attempt)
                prompt = (
                    f"{user_prompt}\n\n"
                    f"ВНИМАНИЕ: твой предыдущий ответ не прошёл валидацию схемы "
                    f"({exc}). Верни СТРОГО один валидный JSON-объект по схеме "
                    f"из инструкции, без markdown и пояснений."
                )
            except TimeoutError as exc:
                last_error = exc
                log.error("agent_timeout", timeout_sec=self.timeout_sec)
                break
            except Exception as exc:  # транспорт/провайдер LLM
                last_error = exc
                log.error("agent_llm_error", error=str(exc))
                break

        raise AgentError(self.name, str(last_error))
