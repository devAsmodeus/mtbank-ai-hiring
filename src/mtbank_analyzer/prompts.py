"""Реестр версионированных промптов агентов.

Промпты вынесены из кода в YAML-файлы (``prompts/<agent>.yaml``), потому что
для банка это данные, а не код: их правят и версионируют отдельно от релиза,
а версия промпта пишется в лог и в ``meta`` отчёта - это основа для A/B и
корреляции «качество отчёта ↔ версия промпта».

Каждый файл: ``name``, ``version``, ``system``. Загрузка ленивая и кэшируется.
Каталог можно переопределить через ``MTBANK_PROMPTS_DIR`` (A/B: подсунуть
альтернативный набор промптов, не пересобирая образ).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    """Один промпт: имя агента, версия, системный текст."""

    name: str
    version: str
    system: str


class PromptRegistry:
    """Загружает и кэширует промпты из каталога YAML."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or Path(os.environ.get("MTBANK_PROMPTS_DIR", _DEFAULT_DIR))
        self._cache: dict[str, Prompt] = {}

    def get(self, name: str) -> Prompt:
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name: str) -> Prompt:
        path = self.directory / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Промпт не найден: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            return Prompt(name=data["name"], version=str(data["version"]), system=data["system"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Некорректный формат промпта {path}: {exc}") from exc


@lru_cache
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()
