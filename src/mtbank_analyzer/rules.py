"""Реестр compliance-правил (regex-стоп-фразы) из YAML-конфига.

Регуляторные стоп-фразы — данные, которыми владеет compliance-офицер, а не
код. Они вынесены в ``rules/compliance.yaml`` и правятся без релиза. Каталог
переопределяется через ``MTBANK_RULES_DIR``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from mtbank_analyzer.schemas import Severity

_DEFAULT_DIR = Path(__file__).parent / "rules"


@dataclass(frozen=True)
class ComplianceRule:
    """Скомпилированное детерминированное правило."""

    rule: str
    severity: Severity
    pattern: re.Pattern[str]


def _load_rules(directory: Path) -> list[ComplianceRule]:
    path = directory / "compliance.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules: list[ComplianceRule] = []
    for item in data.get("patterns", []):
        rules.append(
            ComplianceRule(
                rule=item["rule"],
                severity=item.get("severity", "medium"),
                pattern=re.compile(item["regex"], re.IGNORECASE),
            )
        )
    return rules


@lru_cache
def get_compliance_rules() -> list[ComplianceRule]:
    directory = Path(os.environ.get("MTBANK_RULES_DIR", _DEFAULT_DIR))
    return _load_rules(directory)


@lru_cache
def get_quality_weights() -> dict[str, int]:
    """Веса пунктов чеклиста качества (бизнес-политика скоринга) из конфига."""
    directory = Path(os.environ.get("MTBANK_RULES_DIR", _DEFAULT_DIR))
    data = yaml.safe_load((directory / "quality_weights.yaml").read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}
