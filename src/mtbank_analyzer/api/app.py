"""Фабрика FastAPI-приложения Analysis Engine.

Запуск: ``uvicorn --factory mtbank_analyzer.api.app:create_app``.
Все зависимости (ASR, оркестратор, хранилище) внедряются через фабрику —
в тестах подменяются стабами без monkeypatching.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from mtbank_analyzer import __version__
from mtbank_analyzer.agents import OpenAICompatLLM
from mtbank_analyzer.agents.trends import TrendsAgent
from mtbank_analyzer.api.routes import router
from mtbank_analyzer.api.ws import ws_router
from mtbank_analyzer.asr import TranscriptionService
from mtbank_analyzer.asr.audio import AudioError
from mtbank_analyzer.config import Settings
from mtbank_analyzer.logging_setup import configure_logging, get_logger
from mtbank_analyzer.orchestration import CallAnalysisOrchestrator
from mtbank_analyzer.storage import AnalysisStorage, JsonlAnalysisStore

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    transcription_service: TranscriptionService | None = None,
    orchestrator: CallAnalysisOrchestrator | None = None,
    trends_agent: TrendsAgent | None = None,
    store: AnalysisStorage | None = None,
    warmup_asr: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    transcription_service = transcription_service or TranscriptionService(settings)
    if orchestrator is None or trends_agent is None:
        llm = OpenAICompatLLM(settings)
        orchestrator = orchestrator or CallAnalysisOrchestrator(
            llm, agent_timeout_sec=settings.agent_timeout_sec
        )
        trends_agent = trends_agent or TrendsAgent(llm=llm, timeout_sec=settings.agent_timeout_sec)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "engine_starting",
            whisper_model=settings.whisper_model,
            llm_model=settings.llm_model,
            llm_base_url=settings.llm_base_url,
        )
        warmup_task = None
        if warmup_asr:
            # прогрев whisper в фоне: первый запрос не ждёт загрузку модели
            warmup_task = asyncio.create_task(
                asyncio.to_thread(transcription_service.transcriber.warmup)
            )
        yield
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
        logger.info("engine_stopped")

    app = FastAPI(
        title="MTBank Call Analysis Engine",
        description="ASR + Multi-Agent аналитика звонков контакт-центра",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.transcription = transcription_service
    app.state.orchestrator = orchestrator
    app.state.trends_agent = trends_agent
    app.state.store = store or JsonlAnalysisStore(settings.storage_dir)

    @app.middleware("http")
    async def request_logging(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        correlation_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        request.state.correlation_id = correlation_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("correlation_id")
        if request.url.path not in ("/metrics", "/healthz"):
            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=int((time.perf_counter() - started) * 1000),
                correlation_id=correlation_id,
            )
        response.headers["x-request-id"] = correlation_id
        return response

    @app.exception_handler(AudioError)
    async def audio_error_handler(request: Request, exc: AudioError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.include_router(router)
    app.include_router(ws_router)
    return app
