"""Интеграционный тест OpenWebUI Pipeline (ТЗ: интеграционный тест pipeline).

Pipeline тестируется как чёрный ящик: engine замокан respx-ом, LLM — fake.
Проверяем весь путь: сообщение пользователя → скачивание аудио →
/transcribe → LangGraph-агенты → markdown-отчёт → push в /reports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest
import respx
from tests.conftest import GOOD_RESPONSES, RoutingFakeLLM

ROOT = Path(__file__).resolve().parent.parent
ENGINE = "http://engine.test"


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("owui_pipeline", ROOT / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["owui_pipeline"] = module
    spec.loader.exec_module(module)
    return module


pipeline_module = _load_pipeline_module()

TRANSCRIPTION_JSON = {
    "segments": [
        {
            "speaker": "Оператор",
            "start": 0.0,
            "end": 4.2,
            "text": "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?",
        },
        {
            "speaker": "Клиент",
            "start": 4.5,
            "end": 8.1,
            "text": "Здравствуйте, хочу узнать про кредит.",
        },
    ],
    "language": "ru",
    "duration_sec": 8.1,
    "asr_model": "large-v3-turbo",
}


@pytest.fixture
def pipeline():
    p = pipeline_module.Pipeline()
    p.valves = p.Valves(ENGINE_URL=ENGINE, OPENWEBUI_BASE_URL="http://owui.test")
    p._build_orchestrator(llm=RoutingFakeLLM(dict(GOOD_RESPONSES)))
    return p


def _collect(generator) -> str:
    return "".join(generator)


@respx.mock
def test_pipe_url_message_full_flow(pipeline) -> None:
    respx.get("https://files.test/call.wav").mock(
        return_value=httpx.Response(200, content=b"RIFF....WAVE-fake")
    )
    transcribe_route = respx.post(f"{ENGINE}/transcribe").mock(
        return_value=httpx.Response(200, json=TRANSCRIPTION_JSON)
    )
    reports_route = respx.post(f"{ENGINE}/reports").mock(return_value=httpx.Response(204))

    output = _collect(
        pipeline.pipe(
            "Проанализируй звонок https://files.test/call.wav",
            "mtbank-call-analysis",
            [],
            {},
        )
    )

    # прогресс + все секции отчёта
    assert "Транскрибирую" in output
    assert "## 📞 Анализ звонка" in output
    assert "**Тематика:** кредиты" in output
    assert "**Качество:** 100/100" in output
    assert "### 🛡️ Комплаенс" in output
    assert "Нарушений не обнаружено" in output
    assert "### 📝 Резюме" in output
    assert "- [ ] Оператор: отправить инструкцию на email" in output
    assert "[00:00] Оператор:" in output
    # ASR вызван, отчёт отправлен в engine
    assert transcribe_route.called
    assert reports_route.called


@respx.mock
def test_pipe_openwebui_attached_file(pipeline) -> None:
    file_route = respx.get("http://owui.test/api/v1/files/f-123/content").mock(
        return_value=httpx.Response(200, content=b"ID3-fake-mp3")
    )
    respx.post(f"{ENGINE}/transcribe").mock(
        return_value=httpx.Response(200, json=TRANSCRIPTION_JSON)
    )
    respx.post(f"{ENGINE}/reports").mock(return_value=httpx.Response(204))

    body = {
        "files": [
            {
                "type": "file",
                "id": "f-123",
                "file": {"id": "f-123", "filename": "звонок.mp3", "meta": {}},
            }
        ]
    }
    output = _collect(pipeline.pipe("проанализируй", "m", [], body))

    assert file_route.called
    assert "## 📞 Анализ звонка" in output


def test_pipe_without_audio_shows_help(pipeline) -> None:
    output = _collect(pipeline.pipe("привет, что ты умеешь?", "m", [], {}))
    assert "Прикрепите аудиофайл" in output


@respx.mock
def test_pipe_engine_down_is_user_friendly(pipeline) -> None:
    respx.get("https://files.test/call.wav").mock(
        return_value=httpx.Response(200, content=b"RIFF....WAVE")
    )
    respx.post(f"{ENGINE}/transcribe").mock(side_effect=httpx.ConnectError)

    output = _collect(pipeline.pipe("https://files.test/call.wav", "m", [], {}))

    assert "⚠️" in output
    assert "ASR-сервис недоступен" in output


@respx.mock
def test_pipe_bad_audio_shows_engine_detail(pipeline) -> None:
    respx.get("https://files.test/doc.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4")
    )
    respx.post(f"{ENGINE}/transcribe").mock(
        return_value=httpx.Response(400, json={"detail": "Неподдерживаемый формат аудио."})
    )

    output = _collect(pipeline.pipe("https://files.test/doc.pdf", "m", [], {}))

    assert "Неподдерживаемый формат аудио" in output


@respx.mock
def test_pipe_trends_command(pipeline) -> None:
    respx.get(f"{ENGINE}/trends").mock(
        return_value=httpx.Response(
            200,
            json={
                "calls_analyzed": 5,
                "topics": {"кредиты": 3, "карты": 2},
                "avg_quality": 82.0,
                "compliance_violation_rate": 0.2,
                "frequent_issues": {"Гарантия одобрения кредита": 1},
                "patterns": ["Рост обращений по кредитам"],
                "recommendations": ["Обновить скрипт приветствия"],
            },
        )
    )

    output = _collect(pipeline.pipe("тренды", "m", [], {}))

    assert "## 📈 Тренды контакт-центра" in output
    assert "кредиты: 3" in output
    assert "Рост обращений по кредитам" in output


@respx.mock
def test_pipe_degrades_when_llm_fails(pipeline) -> None:
    """LLM полностью недоступна → отчёт всё равно приходит, с пометкой."""
    responses = dict(GOOD_RESPONSES)
    for agent in ("classifier", "quality", "compliance", "summarizer"):
        responses[agent] = "__RAISE__"
    pipeline._build_orchestrator(llm=RoutingFakeLLM(responses))

    respx.get("https://files.test/call.wav").mock(
        return_value=httpx.Response(200, content=b"RIFF....WAVE")
    )
    respx.post(f"{ENGINE}/transcribe").mock(
        return_value=httpx.Response(200, json=TRANSCRIPTION_JSON)
    )
    respx.post(f"{ENGINE}/reports").mock(return_value=httpx.Response(204))

    output = _collect(pipeline.pipe("https://files.test/call.wav", "m", [], {}))

    assert "## 📞 Анализ звонка" in output
    assert "Часть агентов недоступна" in output
    # fail-closed комплаенс
    assert "Проверка не выполнена" in output
