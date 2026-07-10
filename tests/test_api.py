"""Интеграционные тесты FastAPI Analysis Engine (стаб-ASR + fake-LLM)."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from tests.conftest import (
    GOOD_RESPONSES,
    RoutingFakeLLM,
    StubTranscriber,
    make_wav_bytes,
    tone,
)

from mtbank_analyzer.agents.trends import TrendsAgent
from mtbank_analyzer.api.app import create_app
from mtbank_analyzer.asr.service import TranscriptionService
from mtbank_analyzer.config import Settings
from mtbank_analyzer.orchestration import CallAnalysisOrchestrator


@pytest.fixture
def client(settings: Settings) -> TestClient:
    llm = RoutingFakeLLM(dict(GOOD_RESPONSES))
    app = create_app(
        settings=settings,
        transcription_service=TranscriptionService(settings, transcriber=StubTranscriber()),
        orchestrator=CallAnalysisOrchestrator(llm),
        trends_agent=TrendsAgent(llm=llm),
        warmup_asr=False,
    )
    with TestClient(app) as test_client:
        yield test_client


WAV = None  # ленивая генерация — один раз на сессию


def wav_bytes() -> bytes:
    global WAV
    if WAV is None:
        WAV = make_wav_bytes(tone(440, 1.0))
    return WAV


# ------------------------------------------------------------------ analyze


def test_analyze_multipart_returns_full_contract(client: TestClient) -> None:
    response = client.post("/analyze", files={"file": ("call.wav", wav_bytes(), "audio/wav")})

    assert response.status_code == 200, response.text
    body = response.json()
    # контракт ТЗ
    assert body["classification"] == {
        "topic": "кредиты",
        "priority": "medium",
        "reason": "кредит наличными",
    }
    assert body["quality_score"]["total"] == 100
    assert body["quality_score"]["checklist"]["greeting"] is True
    assert body["compliance"]["passed"] is True
    assert body["summary"].startswith("Клиент уточнил")
    assert body["action_items"] == ["Оператор: отправить инструкцию на email"]
    assert body["transcript"][0]["speaker"] in ("Оператор", "Клиент")
    assert {"start", "end", "text"} <= set(body["transcript"][0])
    # метаданные и трассировка
    assert body["meta"]["processing_ms"] >= 0
    assert body["meta"]["asr_model"] == "stub-whisper"
    assert response.headers["x-request-id"] == body["meta"]["correlation_id"]


@respx.mock
def test_analyze_by_json_url(client: TestClient) -> None:
    respx.get("https://example.com/call.wav").mock(
        return_value=httpx.Response(200, content=wav_bytes())
    )

    response = client.post("/analyze", json={"url": "https://example.com/call.wav"})

    assert response.status_code == 200, response.text
    assert response.json()["classification"]["topic"] == "кредиты"


def test_analyze_without_input_is_422(client: TestClient) -> None:
    response = client.post("/analyze")
    assert response.status_code == 422


def test_analyze_unsupported_format_is_400(client: TestClient) -> None:
    response = client.post(
        "/analyze", files={"file": ("x.zip", b"PK\x03\x04" + b"\x00" * 100, "application/zip")}
    )
    assert response.status_code == 400
    assert "Неподдерживаемый формат" in response.json()["detail"]


def test_analyze_empty_file_is_400(client: TestClient) -> None:
    response = client.post("/analyze", files={"file": ("x.wav", b"", "audio/wav")})
    assert response.status_code == 400


# --------------------------------------------------------------- transcribe


def test_transcribe_returns_segments(client: TestClient) -> None:
    response = client.post("/transcribe", files={"file": ("call.wav", wav_bytes(), "audio/wav")})

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "ru"
    assert body["asr_model"] == "stub-whisper"
    assert len(body["segments"]) == 1
    assert body["segments"][0]["speaker"] in ("Оператор", "Клиент")


# ------------------------------------------------------------ service endpoints


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["asr_model"] == "stub-whisper"


def test_metrics_exposes_business_counters(client: TestClient) -> None:
    client.post("/analyze", files={"file": ("call.wav", wav_bytes(), "audio/wav")})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "calls_analyzed_total" in response.text
    assert 'calls_by_topic_total{topic="кредиты"}' in response.text


# -------------------------------------------------------------------- trends


def test_trends_requires_at_least_two_calls(client: TestClient) -> None:
    response = client.get("/trends")
    assert response.status_code == 409


def test_trends_rejects_nonpositive_limit(client: TestClient) -> None:
    # limit=0 больше не означает «вся история» — валидатор отвергает
    assert client.get("/trends?limit=0").status_code == 422
    assert client.get("/trends?limit=-5").status_code == 422


def test_trends_aggregates_and_insights(client: TestClient) -> None:
    for _ in range(2):
        assert (
            client.post(
                "/analyze", files={"file": ("call.wav", wav_bytes(), "audio/wav")}
            ).status_code
            == 200
        )

    response = client.get("/trends")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["calls_analyzed"] == 2
    assert body["topics"] == {"кредиты": 2}
    assert body["avg_quality"] == 100.0
    assert body["compliance_violation_rate"] == 0.0
    assert body["patterns"]  # инсайты LLM
    assert body["recommendations"]


# ------------------------------------------------------------------- reports


def test_reports_ingest_feeds_trends_and_metrics(client: TestClient) -> None:
    report = {
        "transcript": [{"speaker": "Клиент", "start": 0, "end": 1, "text": "Алло"}],
        "classification": {"topic": "карты", "priority": "high", "reason": "блокировка"},
        "quality_score": {
            "total": 45,
            "checklist": {
                "greeting": True,
                "need_detection": False,
                "solution_provided": False,
                "farewell": True,
            },
        },
        "compliance": {"passed": False, "issues": [{"rule": "Грубость"}]},
        "summary": "Клиент пожаловался на блокировку карты.",
        "action_items": [],
        "meta": {"correlation_id": "from-pipeline"},
    }

    assert client.post("/reports", json=report).status_code == 204
    assert client.post("/reports", json=report).status_code == 204

    trends = client.get("/trends").json()
    assert trends["calls_analyzed"] == 2
    assert trends["topics"] == {"карты": 2}
    assert trends["compliance_violation_rate"] == 1.0
    assert trends["frequent_issues"] == {"Грубость": 2}

    metrics_text = client.get("/metrics").text
    assert 'calls_by_topic_total{topic="карты"}' in metrics_text


# ----------------------------------------------------------------- websocket


def test_ws_realtime_transcription(client: TestClient) -> None:
    pcm = (tone(440, 2.5) * 32767).astype("<i2").tobytes()

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text('{"sample_rate": 16000}')
        assert ws.receive_json()["type"] == "ready"

        ws.send_bytes(pcm)  # 2.5 c > блок 2 c → сегмент приходит сразу
        message = ws.receive_json()
        assert message["type"] == "segment"
        assert "МТБанк" in message["text"]

        ws.send_text('{"type": "flush"}')
        # остаток 0.5 c дорабатывается отдельным сегментом
        tail = ws.receive_json()
        assert tail["type"] == "segment"
        assert tail["start"] >= 2.0

        done = ws.receive_json()
        assert done["type"] == "done"
        assert done["duration_sec"] == pytest.approx(2.5, abs=0.05)
        assert done["segments"], "финал должен содержать диаризованные сегменты"
        assert done["segments"][0]["speaker"] in ("Оператор", "Клиент")


def test_ws_rejects_invalid_sample_rate(client: TestClient) -> None:
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text('{"sample_rate": 0}')
        message = ws.receive_json()
        assert message["type"] == "error"


def test_ws_odd_frame_does_not_crash(client: TestClient) -> None:
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text('{"sample_rate": 16000}')
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(b"\x00\x01\x02")  # нечётное число байт — не должно ронять сокет
        ws.send_text('{"type": "flush"}')
        assert ws.receive_json()["type"] == "done"
