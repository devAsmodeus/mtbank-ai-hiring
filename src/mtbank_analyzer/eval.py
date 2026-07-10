"""Оценка качества агентов на эталонном наборе (ML-flow / регрессии промптов).

WER меряет ASR; этот слой меряет суждения агентов: точность классификации,
чеклиста качества и детекции compliance-нарушений. Baseline-метрики нужны,
чтобы смена модели или правка промпта не роняли качество незаметно (при A/B
двух версий промпта сравниваются именно эти числа).

Логика вынесена сюда, чтобы её переиспользовали и CLI-скрипт
(``scripts/evaluate_agents.py`` — полный прогон через LLM), и офлайн-тест
(``tests/test_eval.py`` — детерминированный compliance-контур без LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mtbank_analyzer.orchestration import CallAnalysisOrchestrator
from mtbank_analyzer.schemas import TranscriptionResult, TranscriptSegment

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "eval" / "golden_set.yaml"


@dataclass(frozen=True)
class GoldenCase:
    """Один эталонный звонок: транскрипт + ожидаемые результаты агентов."""

    id: str
    segments: list[TranscriptSegment]
    expected: dict


def load_golden_set(path: Path | None = None) -> list[GoldenCase]:
    data = yaml.safe_load((path or _DEFAULT_PATH).read_text(encoding="utf-8"))
    cases: list[GoldenCase] = []
    for raw in data.get("cases", []):
        segments = [
            TranscriptSegment(
                speaker=turn["speaker"],
                start=float(i),
                end=float(i) + 1.0,
                text=turn["text"],
            )
            for i, turn in enumerate(raw["dialog"])
        ]
        cases.append(GoldenCase(id=raw["id"], segments=segments, expected=raw["expected"]))
    return cases


@dataclass
class EvalMetrics:
    """Агрегированные метрики точности по эталонному набору."""

    cases: int = 0
    topic_hits: int = 0
    topic_total: int = 0
    checklist_hits: int = 0
    checklist_total: int = 0
    compliance_hits: int = 0
    compliance_total: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def topic_accuracy(self) -> float:
        return self.topic_hits / self.topic_total if self.topic_total else 1.0

    @property
    def checklist_accuracy(self) -> float:
        return self.checklist_hits / self.checklist_total if self.checklist_total else 1.0

    @property
    def compliance_accuracy(self) -> float:
        return self.compliance_hits / self.compliance_total if self.compliance_total else 1.0

    def meets(
        self, *, topic: float = 0.75, checklist: float = 0.75, compliance: float = 0.9
    ) -> bool:
        return (
            self.topic_accuracy >= topic
            and self.checklist_accuracy >= checklist
            and self.compliance_accuracy >= compliance
        )


async def evaluate(orchestrator: CallAnalysisOrchestrator, cases: list[GoldenCase]) -> EvalMetrics:
    """Прогоняет граф на эталонных кейсах и сравнивает с ожиданиями."""
    metrics = EvalMetrics()
    for case in cases:
        metrics.cases += 1
        transcription = TranscriptionResult(segments=case.segments, language="ru")
        report = await orchestrator.analyze(transcription)
        exp = case.expected

        if "topic" in exp:
            metrics.topic_total += 1
            if report.classification.topic == exp["topic"]:
                metrics.topic_hits += 1
            else:
                metrics.failures.append(
                    f"{case.id}: topic {report.classification.topic} != {exp['topic']}"
                )

        for item, want in (exp.get("checklist") or {}).items():
            metrics.checklist_total += 1
            if getattr(report.quality_score.checklist, item) == want:
                metrics.checklist_hits += 1

        if "compliance_passed" in exp:
            metrics.compliance_total += 1
            if report.compliance.passed == exp["compliance_passed"]:
                metrics.compliance_hits += 1
            else:
                metrics.failures.append(
                    f"{case.id}: compliance_passed {report.compliance.passed} "
                    f"!= {exp['compliance_passed']}"
                )

    return metrics
