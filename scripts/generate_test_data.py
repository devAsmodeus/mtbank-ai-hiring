"""Генерация тестовых аудиоданных (ТЗ: подготовка данных — часть задания).

Синтезирует русские диалоги/монологи через edge-tts (Светлана — оператор,
Дмитрий — клиент), склеивает реплики с паузами и раскладывает по форматам:

- dialog_credit_16k.wav      — диалог о кредите (сценарий из ТЗ), WAV 16 кГц
- dialog_credit_8k_phone.wav — тот же диалог через телефонный кодек (8 кГц µ-law)
- dialog_card_block.mp3      — диалог с compliance-нарушениями, MP3
- dialog_card_block_stereo.wav — тот же диалог, оператор/клиент по каналам (стерео)
- monologue_deposit.ogg      — IVR-монолог о вкладах, OGG Vorbis
- monologue_transfer_8k.wav  — голосовое обращение клиента, 8 кГц µ-law

Рядом с каждым аудио пишется эталонный транскрипт ``<имя>.txt``
(строки «Роль: текст» — ровно то, что подавалось в TTS).

Запуск:  python scripts/generate_test_data.py  [--out test_data]
Требует: интернет (edge-tts), ffmpeg в PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

# Корпоративные сети с TLS-инспекцией: берём доверие из системного хранилища
with contextlib.suppress(ImportError):
    import truststore

    truststore.inject_into_ssl()

SR = 16_000
PAUSE_SEC = 0.7

VOICES = {
    "Оператор": "ru-RU-SvetlanaNeural",
    "Клиент": "ru-RU-DmitryNeural",
}

Line = tuple[str, str]  # (роль, текст)

# Сценарий из docs/sample-dialog.md исходного ТЗ (кредит наличными)
DIALOG_CREDIT: list[Line] = [
    ("Оператор", "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?"),
    ("Клиент", "Здравствуйте. Хочу узнать про условия по кредиту наличными."),
    ("Оператор", "Конечно, подскажите, пожалуйста, какая сумма вас интересует и на какой срок?"),
    ("Клиент", "Примерно десять тысяч рублей, на год."),
    (
        "Оператор",
        "Отлично. На данный момент ставка от четырнадцати и девяти процентов годовых, решение за пятнадцать минут. Вы уже являетесь клиентом МТБанка?",
    ),
    ("Клиент", "Да, у меня есть карточка ваша."),
    (
        "Оператор",
        "Прекрасно, тогда для вас действуют специальные условия. Ежемесячный платёж составит около девятисот рублей. Вам удобно подать заявку онлайн через приложение или предпочитаете приехать в отделение?",
    ),
    ("Клиент", "Лучше онлайн. Но у меня вопрос — если я захочу досрочно погасить, есть штрафы?"),
    (
        "Оператор",
        "Нет, досрочное погашение без штрафов и комиссий, в любое время и в любом объёме.",
    ),
    ("Клиент", "Хорошо, а страховка обязательна?"),
    (
        "Оператор",
        "Страхование жизни подключается по вашему желанию, это не обязательное условие получения кредита. Однако при подключении страховки ставка может быть немного снижена.",
    ),
    ("Клиент", "Понятно. Тогда я попробую подать через приложение."),
    (
        "Оператор",
        "Отлично. Если возникнут вопросы в процессе заполнения — звоните, мы поможем. Также могу отправить вам краткую инструкцию на почту, если хотите.",
    ),
    ("Клиент", "Да, пожалуйста, отправьте."),
    (
        "Оператор",
        "Хорошо, письмо с инструкцией и ссылкой на заявку придёт на адрес из вашего профиля в течение нескольких минут. Есть ещё вопросы?",
    ),
    ("Клиент", "Нет, всё понятно, спасибо."),
    ("Оператор", "Спасибо за обращение в МТБанк, хорошего дня!"),
    ("Клиент", "И вам, до свидания."),
]

# Диалог с намеренными compliance-нарушениями (для демонстрации агентов)
DIALOG_CARD: list[Line] = [
    ("Оператор", "Добрый день, МТБанк, меня зовут Ольга, слушаю вас."),
    (
        "Клиент",
        "Здравствуйте. У меня заблокировалась карта, не могу расплатиться в магазине. Я очень недоволен, уже второй раз за месяц такая история.",
    ),
    (
        "Оператор",
        "Понимаю ваше недовольство, давайте разберёмся. Подскажите, пожалуйста, вашу фамилию и имя.",
    ),
    ("Клиент", "Сидоров Пётр."),
    (
        "Оператор",
        "Спасибо. Для проверки продиктуйте, пожалуйста, полный номер карты и код CVV с обратной стороны.",
    ),
    ("Клиент", "Разве можно называть CVV по телефону? Мне казалось, это запрещено."),
    ("Оператор", "Не переживайте, это стандартная процедура, у нас всё защищено."),
    ("Клиент", "Нет, код я диктовать не буду."),
    (
        "Оператор",
        "Хорошо, проверила по номеру телефона. Карта заблокирована из-за подозрительной операции. Могу оформить перевыпуск, новую карту одобрят стопроцентно, я вам гарантирую.",
    ),
    ("Клиент", "А сколько стоит перевыпуск?"),
    (
        "Оператор",
        "Для вас бесплатно, но только если оформим прямо сейчас — завтра это предложение сгорит.",
    ),
    ("Клиент", "Ну хорошо, давайте оформим."),
    (
        "Оператор",
        "Оформила. Новая карта будет готова через три рабочих дня, забрать можно в вашем отделении. Всего доброго.",
    ),
    ("Клиент", "До свидания."),
]

# IVR-монолог о вкладах (один голос, оператор)
MONOLOGUE_DEPOSIT: list[Line] = [
    (
        "Оператор",
        "Здравствуйте! Вы позвонили в МТБанк. Информируем вас об актуальных условиях по вкладам. "
        "Вклад «Накопительный» — ставка до одиннадцати процентов годовых при размещении от трёх месяцев, "
        "пополнение и частичное снятие доступны в мобильном приложении без комиссии. "
        "Вклад «Стабильный» — фиксированная ставка девять и пять десятых процента годовых на весь срок. "
        "Проценты выплачиваются ежемесячно на карту или капитализируются по вашему выбору. "
        "Точные условия рассчитываются индивидуально и зависят от суммы и срока размещения. "
        "Для оформления вклада посетите ближайшее отделение или воспользуйтесь мобильным приложением. "
        "Спасибо за обращение в МТБанк!",
    ),
]

# Голосовое обращение клиента (один голос, телефонное качество)
MONOLOGUE_TRANSFER: list[Line] = [
    (
        "Клиент",
        "Здравствуйте. Я звоню по поводу перевода, который отправил вчера вечером на карту другого банка. "
        "Деньги до сих пор не пришли, хотя обычно зачисление занимает несколько минут. "
        "Сумма перевода — двести пятьдесят рублей. "
        "Прошу проверить статус операции и перезвонить мне по номеру, с которого я звоню. Спасибо.",
    ),
]


def _decode_mp3(path: Path) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return np.asarray(decode_audio(str(path), sampling_rate=SR), dtype=np.float32)


async def _synthesize_lines(lines: list[Line], workdir: Path) -> list[np.ndarray]:
    import edge_tts

    workdir.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = []
    for i, (role, text) in enumerate(lines):
        piece_path = workdir / f"line_{i:03d}.mp3"
        await edge_tts.Communicate(text, VOICES[role]).save(str(piece_path))
        pieces.append(_decode_mp3(piece_path))
        print(f"  synthesized [{role}] {text[:50]}...", flush=True)
    return pieces


def _write_wav(path: Path, waveform: np.ndarray, channels: int = 1) -> None:
    pcm = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


def _assemble_mono(pieces: list[np.ndarray]) -> np.ndarray:
    pause = np.zeros(int(PAUSE_SEC * SR), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for piece in pieces:
        chunks += [piece, pause]
    return np.concatenate(chunks[:-1])


def _assemble_stereo(lines: list[Line], pieces: list[np.ndarray]) -> np.ndarray:
    """Оператор — левый канал, клиент — правый (как в проде колл-центров)."""
    pause = int(PAUSE_SEC * SR)
    total = sum(len(p) for p in pieces) + pause * (len(pieces) - 1)
    left = np.zeros(total, dtype=np.float32)
    right = np.zeros(total, dtype=np.float32)
    cursor = 0
    for (role, _), piece in zip(lines, pieces, strict=True):
        channel = left if role == "Оператор" else right
        channel[cursor : cursor + len(piece)] = piece
        cursor += len(piece) + pause
    return np.stack([left, right], axis=1)


def _ffmpeg(*args: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def _write_reference(path: Path, lines: list[Line]) -> None:
    path.write_text("\n".join(f"{role}: {text}" for role, text in lines) + "\n", encoding="utf-8")


async def generate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        print("== dialog_credit ==")
        credit_pieces = await _synthesize_lines(DIALOG_CREDIT, workdir / "credit")
        credit_mono = _assemble_mono(credit_pieces)
        wav16 = out_dir / "dialog_credit_16k.wav"
        _write_wav(wav16, credit_mono)
        _write_reference(out_dir / "dialog_credit_16k.txt", DIALOG_CREDIT)
        # телефонный кодек: 8 кГц µ-law (требование ТЗ)
        _ffmpeg(
            "-i",
            str(wav16),
            "-ar",
            "8000",
            "-ac",
            "1",
            "-c:a",
            "pcm_mulaw",
            str(out_dir / "dialog_credit_8k_phone.wav"),
        )
        _write_reference(out_dir / "dialog_credit_8k_phone.txt", DIALOG_CREDIT)

        print("== dialog_card ==")
        card_pieces = await _synthesize_lines(DIALOG_CARD, workdir / "card")
        card_mono = _assemble_mono(card_pieces)
        card_wav = workdir / "dialog_card.wav"
        _write_wav(card_wav, card_mono)
        _ffmpeg(
            "-i",
            str(card_wav),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(out_dir / "dialog_card_block.mp3"),
        )
        _write_reference(out_dir / "dialog_card_block.txt", DIALOG_CARD)
        stereo = _assemble_stereo(DIALOG_CARD, card_pieces)
        _write_wav(out_dir / "dialog_card_block_stereo.wav", stereo, channels=2)
        _write_reference(out_dir / "dialog_card_block_stereo.txt", DIALOG_CARD)

        print("== monologue_deposit ==")
        deposit_pieces = await _synthesize_lines(MONOLOGUE_DEPOSIT, workdir / "dep")
        deposit_wav = workdir / "deposit.wav"
        _write_wav(deposit_wav, _assemble_mono(deposit_pieces))
        _ffmpeg(
            "-i",
            str(deposit_wav),
            "-c:a",
            "libvorbis",
            "-q:a",
            "4",
            str(out_dir / "monologue_deposit.ogg"),
        )
        _write_reference(out_dir / "monologue_deposit.txt", MONOLOGUE_DEPOSIT)

        print("== monologue_transfer ==")
        transfer_pieces = await _synthesize_lines(MONOLOGUE_TRANSFER, workdir / "tr")
        transfer_wav = workdir / "transfer.wav"
        _write_wav(transfer_wav, _assemble_mono(transfer_pieces))
        _ffmpeg(
            "-i",
            str(transfer_wav),
            "-ar",
            "8000",
            "-ac",
            "1",
            "-c:a",
            "pcm_mulaw",
            str(out_dir / "monologue_transfer_8k.wav"),
        )
        _write_reference(out_dir / "monologue_transfer_8k.txt", MONOLOGUE_TRANSFER)

    total_sec = 0.0
    print("\nГотово:")
    for audio in sorted(out_dir.glob("*")):
        if audio.suffix in (".wav", ".mp3", ".ogg"):
            duration = _probe_duration(audio)
            total_sec += duration
            print(f"  {audio.name:36s} {duration:6.1f} c  {audio.stat().st_size // 1024} КБ")
    print(f"Суммарная длительность: {total_sec / 60:.1f} мин")


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("test_data"))
    args = parser.parse_args()
    try:
        asyncio.run(generate(args.out))
    except KeyboardInterrupt:
        sys.exit(130)
