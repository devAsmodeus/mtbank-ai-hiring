"""REST-эндпоинты Analysis Engine."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from mtbank_analyzer.agents.trends import TrendsReport, build_trends_report
from mtbank_analyzer.api import metrics
from mtbank_analyzer.asr.audio import AudioError, fetch_audio_from_url
from mtbank_analyzer.logging_setup import get_logger
from mtbank_analyzer.schemas import AnalysisReport, TranscriptionResult

logger = get_logger(__name__)

router = APIRouter()


async def _read_audio_input(request: Request, file: UploadFile | None, url: str | None) -> bytes:
    """Единый приём аудио: multipart-файл, form-поле url или JSON {"url": ...}."""
    max_mb = request.app.state.settings.max_upload_mb
    max_bytes = max_mb * 1024 * 1024
    if file is None and url is None:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            body = await request.json()
            url = body.get("url") if isinstance(body, dict) else None
    if file is not None:
        # size известен из multipart-парсера до чтения тела — отсекаем большой
        # аплоад, не материализуя его в памяти целиком
        if file.size is not None and file.size > max_bytes:
            raise AudioError(f"Файл больше лимита {max_mb} МБ")
        data = await file.read()
        if len(data) > max_bytes:
            raise AudioError(f"Файл больше лимита {max_mb} МБ")
        if not data:
            raise AudioError("Пустой файл")
        return data
    if url:
        return await fetch_audio_from_url(url, max_bytes=max_bytes)
    raise HTTPException(
        status_code=422,
        detail='Передайте аудио: multipart-поле "file" или {"url": "https://..."}',
    )


@router.post("/analyze", response_model=AnalysisReport)
async def analyze(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
) -> AnalysisReport:
    """Полный анализ звонка: ASR → диаризация → 4 агента → отчёт (контракт ТЗ)."""
    started = time.perf_counter()
    status = "error"
    try:
        data = await _read_audio_input(request, file, url)

        asr_started = time.perf_counter()
        transcription: TranscriptionResult = await request.app.state.transcription.transcribe_bytes(
            data
        )
        metrics.ASR_DURATION.observe(time.perf_counter() - asr_started)
        if not transcription.segments:
            raise HTTPException(status_code=422, detail="Речь в аудио не распознана")

        agents_started = time.perf_counter()
        report: AnalysisReport = await request.app.state.orchestrator.analyze(
            transcription, correlation_id=request.state.correlation_id
        )
        metrics.AGENTS_DURATION.observe(time.perf_counter() - agents_started)

        report.meta.processing_ms = int((time.perf_counter() - started) * 1000)
        metrics.observe_report(
            topic=report.classification.topic,
            quality_total=report.quality_score.total,
            compliance_passed=report.compliance.passed,
        )
        await request.app.state.store.append(report)
        status = "ok"
        return report
    except AudioError:
        status = "bad_input"
        raise
    except HTTPException as exc:
        status = "bad_input" if exc.status_code < 500 else "error"
        raise
    finally:
        metrics.ANALYZE_REQUESTS.labels(status=status).inc()
        metrics.ANALYZE_DURATION.observe(time.perf_counter() - started)


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
) -> TranscriptionResult:
    """Только ASR + диаризация (используется OpenWebUI Pipeline)."""
    data = await _read_audio_input(request, file, url)
    asr_started = time.perf_counter()
    result: TranscriptionResult = await request.app.state.transcription.transcribe_bytes(data)
    metrics.ASR_DURATION.observe(time.perf_counter() - asr_started)
    return result


@router.post("/reports", status_code=204)
async def ingest_report(request: Request, report: AnalysisReport) -> Response:
    """Приём готового отчёта от OpenWebUI Pipeline.

    Pipeline выполняет мультиагентный анализ у себя (см. pipeline.py), а сюда
    отдаёт результат, чтобы метрики Prometheus и хранилище трендов оставались
    едиными для чата и REST.
    """
    metrics.observe_report(
        topic=report.classification.topic,
        quality_total=report.quality_score.total,
        compliance_passed=report.compliance.passed,
    )
    await request.app.state.store.append(report)
    return Response(status_code=204)


@router.get("/trends", response_model=TrendsReport)
async def trends(request: Request, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> TrendsReport:
    """Бонус: агент трендов по последним проанализированным звонкам."""
    records = await request.app.state.store.load_recent(limit=limit)
    if len(records) < 2:
        raise HTTPException(
            status_code=409,
            detail="Недостаточно данных: проанализируйте хотя бы 2 звонка",
        )
    return await build_trends_report(request.app.state.trends_agent, records)


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    transcriber = request.app.state.transcription.transcriber
    return {
        "status": "ok",
        "asr_model": getattr(transcriber, "model_name", "unknown"),
        "asr_loaded": getattr(transcriber, "_model", None) is not None,
        "llm_model": request.app.state.settings.llm_model,
    }


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
