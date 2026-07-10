# Changelog

Значимые изменения проекта. Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версии — [SemVer](https://semver.org/lang/ru/).

## [Unreleased]

## [0.1.0] — 2026-07-10

### Added
- ASR-пайплайн на faster-whisper (`large-v3-turbo`) с диаризацией Оператор/Клиент:
  раздельная транскрибация каналов для стерео, MFCC-кластеризация для моно.
- Мультиагентная аналитика на LangGraph (параллельный fan-out): классификатор,
  агент качества, compliance-агент, суммаризатор.
- Два рантайма на общем ядре: OpenWebUI Pipeline (чат) и FastAPI Analysis Engine.
- REST `POST /analyze` и `/transcribe`, WebSocket `/ws/transcribe` (real-time),
  `GET /trends` (агент трендов), Prometheus `/metrics`, автопровижен дашборд Grafana.
- Реестр версионированных промптов (`prompts/`), compliance-правила и веса
  скоринга в конфиге (`rules/`); версии промптов пишутся в `meta` и логи.
- Eval-слой агентов на эталонном наборе (`eval/golden_set.yaml`).
- Docker Compose (open-webui + pipelines + api + prometheus + grafana), CI, тесты.

[Unreleased]: https://github.com/devAsmodeus/mtbank-ai-hiring/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/devAsmodeus/mtbank-ai-hiring/releases/tag/v0.1.0
