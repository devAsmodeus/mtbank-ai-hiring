# Как вносить изменения

## Окружение

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[api,dev]"
pre-commit install
```

## Проверки перед коммитом

Их же гоняет CI и pre-commit-хук:

```bash
ruff check .          # линт
ruff format .         # форматирование
mypy                  # типизация (src + pipeline.py)
pytest -q             # тесты
```

## Соглашения

- Ветка от `main`, PR с описанием причины изменения.
- Сообщения коммитов — Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`).
- Промпты правятся в `prompts/*.yaml`, регуляторные правила — в `rules/*.yaml`,
  без изменения кода.
- Новый код — с тестами; тесты не требуют GPU, сети и загрузки моделей.
