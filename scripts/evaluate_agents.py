"""Оценка качества агентов на эталонном наборе через реальную LLM (ML-flow).

Меряет точность классификации, чеклиста и детекции compliance против
``eval/golden_set.yaml``. Exit code != 0, если метрики ниже порогов — можно
подключать в CI против прод-модели или для сравнения версий промптов (A/B).

Запуск: python scripts/evaluate_agents.py   (нужна рабочая LLM в .env)
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

# Корпоративные сети с TLS-инспекцией
with contextlib.suppress(ImportError):
    import truststore

    truststore.inject_into_ssl()

from mtbank_analyzer.agents import OpenAICompatLLM
from mtbank_analyzer.config import Settings
from mtbank_analyzer.eval import evaluate, load_golden_set
from mtbank_analyzer.orchestration import CallAnalysisOrchestrator


async def main() -> int:
    settings = Settings()
    orchestrator = CallAnalysisOrchestrator(
        OpenAICompatLLM(settings), agent_timeout_sec=settings.agent_timeout_sec
    )
    cases = load_golden_set()
    print(f"Оценка агентов на {len(cases)} эталонных звонках (модель {settings.llm_model})…\n")
    metrics = await evaluate(orchestrator, cases)

    print(
        f"Topic accuracy:      {metrics.topic_accuracy:.0%}  ({metrics.topic_hits}/{metrics.topic_total})"
    )
    print(
        f"Checklist accuracy:  {metrics.checklist_accuracy:.0%}  "
        f"({metrics.checklist_hits}/{metrics.checklist_total})"
    )
    print(
        f"Compliance accuracy: {metrics.compliance_accuracy:.0%}  "
        f"({metrics.compliance_hits}/{metrics.compliance_total})"
    )
    if metrics.failures:
        print("\nРасхождения:")
        for failure in metrics.failures:
            print(f"  - {failure}")

    ok = metrics.meets()
    print(f"\n{'ПОРОГИ ПРОЙДЕНЫ' if ok else 'НИЖЕ ПОРОГА'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
