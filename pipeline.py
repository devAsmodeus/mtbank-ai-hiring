"""
title: MTBank Call Analysis
author: Pavel Kruglikovskii
version: 0.1.0
license: MIT
description: Анализ звонков контакт-центра — ASR, диаризация и мультиагентная аналитика (LangGraph) прямо в чате OpenWebUI.
"""

# OpenWebUI Pipeline — оркестрация анализа звонка для чат-интерфейса.
#
# Роль в архитектуре (подробное обоснование — README, «Архитектурные решения»):
# - тяжёлый ASR (faster-whisper + диаризация) делегируется Analysis Engine
#   (POST /transcribe): модель загружена один раз в одном процессе;
# - мультиагентная оркестрация (LangGraph: классификатор ∥ качество ∥
#   комплаенс ∥ суммаризатор) выполняется ЗДЕСЬ, в процессе pipelines —
#   как в скелете из ТЗ; тот же граф из общего пакета использует REST /analyze;
# - готовый отчёт отправляется в engine (POST /reports), чтобы Prometheus-метрики
#   и хранилище трендов были едиными для чата и REST.
#
# Тип pipeline — «pipe» (собственная модель в списке моделей OpenWebUI):
# анализ звонка — самостоятельный сценарий с собственным выводом, а не
# модификация чужого запроса (filter) и не кнопка на сообщении (action).

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Generator, Iterator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from mtbank_analyzer.agents import LLMClient, OpenAICompatLLM
from mtbank_analyzer.config import Settings
from mtbank_analyzer.logging_setup import configure_logging, get_logger
from mtbank_analyzer.orchestration import CallAnalysisOrchestrator
from mtbank_analyzer.schemas import (
    OPERATOR,
    AnalysisReport,
    TranscriptionResult,
)

logger = get_logger(__name__)

_URL_RE = re.compile(r"https?://\S+")
_TRENDS_COMMANDS = {"тренды", "trends", "/тренды", "/trends"}
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg")

_HELP_MESSAGE = """\
👋 Я анализирую звонки контакт-центра МТБанка.

**Как отправить звонок на анализ:**
1. 📎 Прикрепите аудиофайл к сообщению (WAV / MP3 / OGG), или
2. 🔗 Пришлите прямую ссылку на аудио (`https://...`), или
3. 📈 Напишите `тренды` — сводка и паттерны по последним звонкам.

**Что вы получите:** транскрипт с диаризацией (Оператор/Клиент), тематику
и приоритет, чеклист качества, compliance-проверку, резюме и action items.
"""


class Pipeline:
    """OpenWebUI Pipeline «pipe»-типа: одна модель = один сценарий анализа."""

    class Valves(BaseModel):
        """Конфигурация, редактируемая в OpenWebUI (Admin → Pipelines)."""

        ENGINE_URL: str = Field(default="http://api:8000", description="URL Analysis Engine (ASR)")
        OPENWEBUI_BASE_URL: str = Field(
            default="http://open-webui:8080",
            description="URL OpenWebUI для скачивания приложенных файлов",
        )
        OPENWEBUI_API_KEY: str = Field(
            default="",
            description="API-ключ OpenWebUI (Settings → Account) для доступа к файлам",
        )
        LLM_BASE_URL: str = "http://ollama:11434/v1"
        LLM_API_KEY: str = "ollama"
        LLM_MODEL: str = "qwen2.5:7b"
        LLM_TEMPERATURE: float = 0.1
        LLM_JSON_MODE: bool = True
        AGENT_TIMEOUT_SEC: float = 120.0
        ASR_TIMEOUT_SEC: float = 600.0

    def __init__(self) -> None:
        self.name = "МТБанк: Анализ звонка"
        # начальные значения валв — из переменных окружения контейнера,
        # дальше редактируются через UI (Admin → Pipelines)
        self.valves = self.Valves(
            **{
                field: os.environ[field]
                for field in self.Valves.model_fields
                if field in os.environ
            }
        )
        self.orchestrator: CallAnalysisOrchestrator | None = None
        configure_logging()

    # ------------------------------------------------------- lifecycle hooks

    async def on_startup(self) -> None:
        self._build_orchestrator()
        logger.info("pipeline_started", engine_url=self.valves.ENGINE_URL)

    async def on_shutdown(self) -> None:
        logger.info("pipeline_stopped")

    async def on_valves_updated(self) -> None:
        self._build_orchestrator()
        logger.info("pipeline_valves_updated", llm_model=self.valves.LLM_MODEL)

    def _build_orchestrator(self, llm: LLMClient | None = None) -> None:
        """Пересобирает граф агентов; в тестах принимает fake-LLM."""
        if llm is None:
            llm = OpenAICompatLLM(
                Settings(
                    llm_base_url=self.valves.LLM_BASE_URL,
                    llm_api_key=self.valves.LLM_API_KEY,
                    llm_model=self.valves.LLM_MODEL,
                    llm_temperature=self.valves.LLM_TEMPERATURE,
                    llm_json_mode=self.valves.LLM_JSON_MODE,
                    _env_file=None,
                )
            )
        self.orchestrator = CallAnalysisOrchestrator(
            llm, agent_timeout_sec=self.valves.AGENT_TIMEOUT_SEC
        )

    # ------------------------------------------------------------------ pipe

    def pipe(
        self, user_message: str, model_id: str, messages: list[dict], body: dict
    ) -> str | Generator | Iterator:
        """Точка входа OpenWebUI. Синхронный генератор — стримим прогресс."""
        return self._run(user_message, body)

    def _run(self, user_message: str, body: dict) -> Generator[str, None, None]:
        try:
            if user_message.strip().lower() in _TRENDS_COMMANDS:
                yield from self._handle_trends()
                return

            audio = self._extract_audio(user_message, body)
            if audio is None:
                yield _HELP_MESSAGE
                return

            source_label, data = audio
            yield f"🎙️ Транскрибирую `{source_label}`…\n"
            transcription = self._transcribe(data)
            if not transcription.segments:
                yield "\n⚠️ Речь в аудио не распознана. Проверьте файл."
                return
            yield (
                f"✅ Готово: {len(transcription.segments)} реплик, "
                f"{transcription.duration_sec:.0f} сек, язык: "
                f"{transcription.language or '—'}\n\n"
            )

            yield "🤖 Агенты анализируют (классификация ∥ качество ∥ комплаенс ∥ резюме)…\n\n"
            assert self.orchestrator is not None, "pipeline не инициализирован"
            report = _run_async(self.orchestrator.analyze(transcription))

            self._push_report_to_engine(report)
            yield "---\n\n"
            yield render_report(report)
        except UserFacingError as exc:
            yield f"\n⚠️ {exc}"
        except Exception:
            logger.exception("pipeline_unhandled_error")
            yield (
                "\n❌ Внутренняя ошибка анализа. Подробности в логах pipelines "
                "(docker compose logs pipelines)."
            )

    # ------------------------------------------------------ получение аудио

    def _extract_audio(self, user_message: str, body: dict) -> tuple[str, bytes] | None:
        """Аудио из приложенного файла OpenWebUI или по URL из текста."""
        file_ref = self._find_attached_file(body)
        if file_ref is not None:
            file_id, filename = file_ref
            return filename, self._download_openwebui_file(file_id, filename)

        match = _URL_RE.search(user_message or "")
        if match:
            url = match.group(0).rstrip(").,;»'\"")
            return url.rsplit("/", 1)[-1] or "audio", self._download_url(url)
        return None

    @staticmethod
    def _find_attached_file(body: dict) -> tuple[str, str] | None:
        """Ищет аудиофайл в body["files"] / body["metadata"]["files"].

        Формат элементов различается между версиями OpenWebUI — разбираем
        защитно: берём последний файл с аудио-расширением или audio/* типом.
        """
        candidates: list[Any] = []
        for container in (body or {}, (body or {}).get("metadata") or {}):
            files = container.get("files")
            if isinstance(files, list):
                candidates.extend(files)

        for item in reversed(candidates):
            if not isinstance(item, dict):
                continue
            file_info = item.get("file") if isinstance(item.get("file"), dict) else item
            file_id = item.get("id") or file_info.get("id")
            filename = (
                file_info.get("filename")
                or file_info.get("name")
                or (file_info.get("meta") or {}).get("name")
                or ""
            )
            content_type = (
                (file_info.get("meta") or {}).get("content_type") or item.get("content_type") or ""
            )
            is_audio = filename.lower().endswith(_AUDIO_EXTENSIONS) or content_type.startswith(
                "audio/"
            )
            if file_id and is_audio:
                return str(file_id), filename or str(file_id)
        return None

    def _download_openwebui_file(self, file_id: str, filename: str) -> bytes:
        url = f"{self.valves.OPENWEBUI_BASE_URL.rstrip('/')}/api/v1/files/{file_id}/content"
        headers = {}
        if self.valves.OPENWEBUI_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.OPENWEBUI_API_KEY}"
        try:
            response = httpx.get(url, headers=headers, timeout=60.0, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("openwebui_file_download_failed", file_id=file_id, error=str(exc))
            raise UserFacingError(
                f"Не удалось скачать файл «{filename}» из OpenWebUI. "
                "Проверьте валву OPENWEBUI_API_KEY (Admin → Pipelines) "
                "или пришлите прямую ссылку на аудио."
            ) from exc
        return response.content

    @staticmethod
    def _download_url(url: str) -> bytes:
        try:
            response = httpx.get(url, timeout=120.0, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise UserFacingError(f"Не удалось скачать аудио по ссылке: {exc}") from exc
        return response.content

    # ------------------------------------------------------- вызовы engine

    def _transcribe(self, data: bytes) -> TranscriptionResult:
        try:
            response = httpx.post(
                f"{self.valves.ENGINE_URL.rstrip('/')}/transcribe",
                files={"file": ("audio", data)},
                timeout=self.valves.ASR_TIMEOUT_SEC,
            )
        except httpx.HTTPError as exc:
            raise UserFacingError(
                f"ASR-сервис недоступен ({exc}). Проверьте контейнер api."
            ) from exc
        if response.status_code == 400:
            raise UserFacingError(response.json().get("detail", "Неподдерживаемое аудио"))
        if response.status_code != 200:
            raise UserFacingError(f"ASR-сервис вернул ошибку HTTP {response.status_code}")
        return TranscriptionResult.model_validate(response.json())

    def _push_report_to_engine(self, report: AnalysisReport) -> None:
        """Отчёт → engine: единые метрики и данные трендов. Не критично."""
        try:
            httpx.post(
                f"{self.valves.ENGINE_URL.rstrip('/')}/reports",
                json=report.model_dump(),
                timeout=10.0,
            ).raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("report_push_failed", error=str(exc))

    def _handle_trends(self) -> Generator[str, None, None]:
        yield "📈 Собираю тренды по проанализированным звонкам…\n\n"
        try:
            response = httpx.get(f"{self.valves.ENGINE_URL.rstrip('/')}/trends", timeout=180.0)
        except httpx.HTTPError as exc:
            raise UserFacingError(f"Сервис аналитики недоступен: {exc}") from exc
        if response.status_code == 409:
            yield "ℹ️ Пока мало данных: проанализируйте хотя бы два звонка."
            return
        if response.status_code != 200:
            raise UserFacingError(f"Ошибка сервиса трендов: HTTP {response.status_code}")
        yield render_trends(response.json())


class UserFacingError(RuntimeError):
    """Ошибка, которую показываем пользователю в чате как есть."""


def _run_async(coro):
    """Выполняет корутину из синхронного pipe().

    Сервер pipelines обычно зовёт sync-pipe в тредпуле (там event loop нет —
    достаточно asyncio.run), но отдельные версии вызывают его из потока с
    запущенным loop — тогда выполняем в отдельном потоке со своим loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


# ------------------------------------------------------------------ рендеринг


def _fmt_ts(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def render_report(report: AnalysisReport) -> str:
    """Финальный markdown-отчёт для чата OpenWebUI."""
    cls = report.classification
    quality = report.quality_score
    compliance = report.compliance

    compliance_badge = (
        "✅ нарушений нет" if compliance.passed else f"🚨 нарушений: {len(compliance.issues)}"
    )
    lines = [
        "## 📞 Анализ звонка",
        "",
        f"**Тематика:** {cls.topic} · **Приоритет:** {cls.priority}",
        f"**Качество:** {quality.total}/100 · **Комплаенс:** {compliance_badge}",
        "",
        "### 📝 Резюме",
        report.summary,
        "",
    ]

    if report.action_items:
        lines += ["**Action items:**"]
        lines += [f"- [ ] {item}" for item in report.action_items]
        lines += [""]

    checklist_names = {
        "greeting": "Приветствие",
        "need_detection": "Выявление потребности",
        "solution_provided": "Решение вопроса",
        "farewell": "Завершение разговора",
    }
    lines += ["### ⭐ Чеклист оператора", "", "| Пункт | Оценка |", "|---|---|"]
    for key, label in checklist_names.items():
        passed = getattr(quality.checklist, key)
        lines.append(f"| {label} | {'✅' if passed else '❌'} |")
    lines += [""]
    if quality.comments:
        lines += ["**Замечания:**"]
        lines += [f"- {comment}" for comment in quality.comments]
        lines += [""]

    lines += ["### 🛡️ Комплаенс", ""]
    if compliance.passed:
        lines += ["Нарушений не обнаружено.", ""]
    else:
        lines += ["| Правило | Серьёзность | Цитата |", "|---|---|---|"]
        for issue in compliance.issues:
            quote = issue.quote.replace("|", "\\|") if issue.quote else "—"
            lines.append(f"| {issue.rule} | {issue.severity} | {quote} |")
        lines += [""]

    operator_count = sum(1 for s in report.transcript if s.speaker == OPERATOR)
    lines += [
        "### 🎙️ Транскрипт",
        "",
        "<details>",
        f"<summary>Показать ({len(report.transcript)} реплик, "
        f"{_fmt_ts(report.meta.audio_duration_sec or 0)}; "
        f"оператор: {operator_count})</summary>",
        "",
    ]
    lines += [f"**[{_fmt_ts(seg.start)}] {seg.speaker}:** {seg.text}" for seg in report.transcript]
    lines += ["", "</details>", ""]

    meta = report.meta
    footer = (
        f"_ASR: {meta.asr_model or '—'} · LLM: {meta.llm_model or '—'}"
        f" · id: {meta.correlation_id or '—'}_"
    )
    if meta.agent_failures:
        failed = ", ".join(f.agent for f in meta.agent_failures)
        footer += f"\n\n⚠️ _Часть агентов недоступна ({failed}) — результат неполный._"
    lines.append(footer)
    return "\n".join(lines)


def render_trends(trends: dict) -> str:
    lines = [
        "## 📈 Тренды контакт-центра",
        "",
        f"Проанализировано звонков: **{trends.get('calls_analyzed', 0)}**",
        f"Средний балл качества: **{trends.get('avg_quality', 0)}/100**",
        f"Доля звонков с нарушениями: **{trends.get('compliance_violation_rate', 0) * 100:.0f}%**",
        "",
        "**Темы обращений:**",
    ]
    lines += [f"- {topic}: {count}" for topic, count in (trends.get("topics") or {}).items()]
    issues = trends.get("frequent_issues") or {}
    if issues:
        lines += ["", "**Частые нарушения:**"]
        lines += [f"- {rule}: {count}" for rule, count in issues.items()]
    patterns = trends.get("patterns") or []
    if patterns:
        lines += ["", "**Паттерны:**"]
        lines += [f"- {pattern}" for pattern in patterns]
    recommendations = trends.get("recommendations") or []
    if recommendations:
        lines += ["", "**Рекомендации:**"]
        lines += [f"- {rec}" for rec in recommendations]
    return "\n".join(lines)
