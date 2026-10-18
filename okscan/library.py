"""Список треков, поиск аудиофайлов в папках и экспорт в CSV — общее для окна и консоли."""
import csv
import os
from dataclasses import dataclass
from pathlib import Path

from okscan.analyzer import Result, SUPPORTED_FORMATS, fold_bpm, format_bpm

ST_QUEUED, ST_RUNNING, ST_DONE, ST_ERROR, ST_STOPPED = "в очереди", "анализ", "готово", "ошибка", "остановлено"


@dataclass(eq=False)
class Track:
    path: str
    order: int
    status: str = ST_QUEUED
    result: Result | None = None
    gen: int = 0


def collect_audio(paths):
    """Раскрывает папки (со всеми подпапками) в список поддерживаемых аудиофайлов."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if Path(f).suffix.lower() in SUPPORTED_FORMATS:
                        out.append(str(Path(root) / f))
        elif p.suffix.lower() in SUPPORTED_FORMATS:
            out.append(str(p))
    return out


def write_csv(path, tracks, bpm_low):
    # Точка с запятой и BOM — чтобы русский Excel открыл файл без вопросов.
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["Файл", "Папка", "BPM", "BPM без приведения", "Уверенность BPM",
                    "Camelot", "Open Key", "Тональность", "Ошибка"])
        for t in tracks:
            r = t.result or Result(path=t.path)
            w.writerow([Path(t.path).name, str(Path(t.path).parent),
                        format_bpm(fold_bpm(r.bpm_raw, bpm_low)),
                        "" if r.bpm_raw is None else f"{r.bpm_raw:.2f}",
                        "" if r.bpm_confidence is None else f"{r.bpm_confidence:.2f}",
                        r.camelot, r.openkey, r.key_name, r.error or ("" if t.result else t.status)])
