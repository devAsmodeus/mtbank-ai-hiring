"""Оценка качества ASR: WER/CER по эталонным транскриптам (jiwer).

Прогоняет все аудио из ``test_data/`` через полный ASR-пайплайн проекта
(декодирование → faster-whisper → диаризация) и сравнивает с эталонами
``<имя>.txt``. Результат — markdown-таблица для README.

Нормализация перед сравнением (стандартная для русского ASR-эваля):
нижний регистр, ё→е, удаление пунктуации, схлопывание пробелов.
Числительные НЕ нормализуются (если модель напишет «10» вместо «десяти» —
это честно засчитывается как ошибка), поэтому цифры в таблице слегка
пессимистичны.

Запуск: python scripts/evaluate_wer.py [--data test_data] [--model large-v3-turbo]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import re
import sys
import time
from pathlib import Path

# Корпоративные сети с TLS-инспекцией (загрузка модели с HuggingFace)
with contextlib.suppress(ImportError):
    import truststore

    truststore.inject_into_ssl()

import jiwer

from mtbank_analyzer.config import Settings
from mtbank_analyzer.schemas import OPERATOR

AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_reference(path: Path) -> str:
    """Эталон: строки «Роль: текст» → сплошной текст."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(re.sub(r"^(Оператор|Клиент):\s*", "", line))
    return " ".join(lines)


async def evaluate(data_dir: Path, model: str | None, out_path: Path) -> None:
    settings = Settings(_env_file=None)
    if model:
        settings = settings.model_copy(update={"whisper_model": model})

    from mtbank_analyzer.asr import TranscriptionService

    service = TranscriptionService(settings)
    print("Прогрев модели (загрузка при первом запуске)…", flush=True)
    service.transcriber.warmup()  # загрузка модели — вне замеров времени

    rows: list[dict] = []
    for audio_path in sorted(data_dir.iterdir()):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        ref_path = audio_path.with_suffix(".txt")
        if not ref_path.exists():
            print(f"!! нет эталона для {audio_path.name}, пропуск")
            continue

        print(f"== {audio_path.name}", flush=True)
        data = audio_path.read_bytes()
        started = time.perf_counter()
        result = await service.transcribe_bytes(data)
        elapsed = time.perf_counter() - started

        hypothesis = normalize(" ".join(seg.text for seg in result.segments))
        reference = normalize(read_reference(ref_path))
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        operator_segments = sum(1 for s in result.segments if s.speaker == OPERATOR)

        rows.append(
            {
                "file": audio_path.name,
                "duration": result.duration_sec,
                "asr_sec": elapsed,
                "rtf": elapsed / result.duration_sec if result.duration_sec else 0,
                "wer": wer,
                "cer": cer,
                "segments": len(result.segments),
                "operator_segments": operator_segments,
            }
        )
        print(
            f"   WER {wer:6.1%}  CER {cer:6.1%}  {elapsed:5.1f} c "
            f"(RTF {rows[-1]['rtf']:.2f}), сегментов: {len(result.segments)}"
        )

    if not rows:
        print("Аудиофайлы не найдены", file=sys.stderr)
        sys.exit(1)

    table = render_markdown(rows, settings.whisper_model)
    out_path.write_text(table, encoding="utf-8")
    print(f"\n{table}\nСохранено в {out_path}")


def render_markdown(rows: list[dict], model: str) -> str:
    total_audio = sum(r["duration"] for r in rows)
    total_asr = sum(r["asr_sec"] for r in rows)
    lines = [
        f"### WER-таблица (модель: `{model}`, CPU int8)",
        "",
        "| Файл | Длит., с | WER | CER | ASR, с | RTF | Реплик (оператор) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['file']}` | {r['duration']:.0f} | **{r['wer']:.1%}** "
            f"| {r['cer']:.1%} | {r['asr_sec']:.1f} | {r['rtf']:.2f} "
            f"| {r['segments']} ({r['operator_segments']}) |"
        )
    lines += [
        "",
        f"Суммарно: {total_audio / 60:.1f} мин аудио за {total_asr:.0f} с "
        f"(средний RTF {total_asr / total_audio:.2f}).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("test_data"))
    parser.add_argument("--model", default=None, help="переопределить WHISPER_MODEL")
    parser.add_argument("--out", type=Path, default=Path("test_data") / "wer_report.md")
    args = parser.parse_args()
    asyncio.run(evaluate(args.data, args.model, args.out))
