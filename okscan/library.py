"""Результат анализа, поиск аудиофайлов в папках, экспорт в CSV и кэш результатов.

Модуль лёгкий (без torch и librosa): консоль сначала смотрит в кэш и грузит
тяжёлый анализатор, только если что-то действительно нужно посчитать.
"""
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_FORMATS = {'.mp3', '.mp4', '.wav', '.flac', '.ogg', '.m4a', '.aac',
                     '.aiff', '.aif', '.au', '.opus', '.wma', '.alac'}

# Номер версии алгоритма в кэше: менять при любой правке, от которой меняется результат,
# иначе останутся старые ответы.
CACHE_VERSION = 1
CACHE_LIMIT = 5000   # столько последних файлов помнит кэш


@dataclass
class Result:
    path: str
    camelot: str = ''
    openkey: str = ''
    key_name: str = ''
    bpm: float | None = None        # после приведения к диапазону
    bpm_raw: float | None = None    # как определил алгоритм
    bpm_confidence: float | None = None
    downbeat: float | None = None   # первая сильная доля, секунды от начала файла
    error: str = ''


@dataclass(eq=False)
class Track:
    path: str
    order: int
    result: Result | None = None


def fold_bpm(bpm, low):
    """Приводит темп к диапазону [low, 2*low] умножением/делением на 2.
    low = 0 — не менять. Лечит самую частую ошибку: темп «вдвое» больше или меньше.
    Обе границы входят: при low=50 темп 100 остаётся 100, а 133 становится 66,5."""
    if bpm is None or not low:
        return bpm
    while bpm < low:
        bpm *= 2
    while bpm > 2 * low:
        bpm /= 2
    return bpm


def format_bpm(bpm):
    """128.0 -> «128», 127.46 -> «127.5». Десятые — потолок разумной точности."""
    if bpm is None:
        return ''
    text = f"{bpm:.1f}"
    return text[:-2] if text.endswith('.0') else text


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
                    "Camelot", "Open Key", "Тональность", "Первая сильная доля, с", "Ошибка"])
        for t in tracks:
            r = t.result or Result(path=t.path)
            w.writerow([Path(t.path).name, str(Path(t.path).parent),
                        format_bpm(fold_bpm(r.bpm_raw, bpm_low)),
                        "" if r.bpm_raw is None else f"{r.bpm_raw:.2f}",
                        "" if r.bpm_confidence is None else f"{r.bpm_confidence:.2f}",
                        r.camelot, r.openkey, r.key_name,
                        "" if r.downbeat is None else f"{r.downbeat:.4f}", r.error])


class ResultCache:
    """Кэш результатов в JSON-файле. Ключ — полный путь, отметка — размер и время
    изменения: перезаписанный (например, заново отрендеренный) файл считается заново.
    Ошибки не кэшируются: файл могут починить или он ещё дописывается."""

    def __init__(self, path):
        self.path = Path(path)
        self.items = {}
        self.dirty = False
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('version') == CACHE_VERSION:
                self.items = dict(data['items'])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass   # нет файла или он испорчен — начинаем с пустого кэша

    @staticmethod
    def _key(path):
        st = os.stat(path)
        return os.path.normcase(os.path.abspath(path)), f"{st.st_size}:{st.st_mtime_ns}"

    def get(self, path, bpm_low):
        try:
            key, stamp = self._key(path)
        except OSError:
            return None
        entry = self.items.get(key)
        if not entry or entry.get('stamp') != stamp:
            return None
        try:
            r = Result(path=str(path), **entry['result'])
        except (TypeError, KeyError):
            return None
        r.bpm = fold_bpm(r.bpm_raw, bpm_low)
        return r

    def put(self, path, result):
        if result.error:
            return
        try:
            key, stamp = self._key(path)
        except OSError:
            return
        data = asdict(result)
        for field in ('path', 'bpm', 'error'):   # путь — в ключе, bpm зависит от диапазона
            data.pop(field)
        self.items.pop(key, None)   # свежие записи — в конец, старые вытесняются первыми
        self.items[key] = {'stamp': stamp, 'result': data}
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        items = dict(list(self.items.items())[-CACHE_LIMIT:])
        tmp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Через временный файл: прерванная запись не портит старый кэш.
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix='.part')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump({'version': CACHE_VERSION, 'items': items}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            # кэш — только ускорение, без него всё работает
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
