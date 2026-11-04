"""Analysis result, finding audio files in folders, CSV export and the results cache.

A light module (no onnxruntime or networks): the console checks the cache first and loads
the analyzer only if something really has to be computed.
"""
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_FORMATS = {'.mp3', '.mp4', '.wav', '.flac', '.ogg', '.m4a', '.aac',
                     '.aiff', '.aif', '.au', '.opus', '.wma', '.alac'}

# Algorithm version in the cache: change it with any edit that changes results,
# otherwise old answers stay.
CACHE_VERSION = 4   # 4: key from the middle 10 minutes; 3: first downbeat fix for syncopated kicks; 2: added the start of sound (lead)
CACHE_LIMIT = 5000   # how many latest files the cache remembers


@dataclass
class Result:
    path: str
    camelot: str = ''
    openkey: str = ''
    key_name: str = ''
    bpm: float | None = None        # after folding into the range
    bpm_raw: float | None = None    # as detected by the algorithm
    bpm_confidence: float | None = None
    downbeat: float | None = None   # first downbeat, seconds from the start of the file
    lead: float | None = None       # where sound starts in the file, s: to match REAPER's timeline
    error: str = ''


@dataclass(eq=False)
class Track:
    path: str
    order: int
    result: Result | None = None


def fold_bpm(bpm, low):
    """Folds tempo into the range [low, 2*low] by multiplying or dividing by 2.
    low = 0 means no change. Fixes the most common error: tempo twice too high or too low.
    Both bounds are included: with low=50, 100 stays 100 and 133 becomes 66.5."""
    if bpm is None or not low:
        return bpm
    while bpm < low:
        bpm *= 2
    while bpm > 2 * low:
        bpm /= 2
    return bpm


def format_bpm(bpm):
    """128.0 -> "128", 127.46 -> "127.5". Tenths are the limit of reasonable precision."""
    if bpm is None:
        return ''
    text = f"{bpm:.1f}"
    return text[:-2] if text.endswith('.0') else text


def collect_audio(paths):
    """Expands folders (with all subfolders) into a list of supported audio files."""
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
    # Semicolon and BOM: Excel with a decimal comma locale opens the file without questions.
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["File", "Folder", "BPM", "BPM unfolded", "BPM confidence",
                    "Camelot", "Open Key", "Key", "First downbeat, s", "Error"])
        for t in tracks:
            r = t.result or Result(path=t.path)
            w.writerow([Path(t.path).name, str(Path(t.path).parent),
                        format_bpm(fold_bpm(r.bpm_raw, bpm_low)),
                        "" if r.bpm_raw is None else f"{r.bpm_raw:.2f}",
                        "" if r.bpm_confidence is None else f"{r.bpm_confidence:.2f}",
                        r.camelot, r.openkey, r.key_name,
                        "" if r.downbeat is None else f"{r.downbeat:.4f}", r.error])


class ResultCache:
    """Results cache in a JSON file. The key is the full path, the stamp is size and
    modification time: an overwritten (say, re-rendered) file is analyzed again.
    Errors are not cached: the file may get fixed or may still be being written."""

    def __init__(self, path):
        self.path = Path(path)
        self.items = {}
        self.dirty = False
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('version') == CACHE_VERSION:
                self.items = dict(data['items'])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass   # no file or a damaged one: start with an empty cache

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
        for field in ('path', 'bpm', 'error'):   # the path is in the key, bpm depends on the range
            data.pop(field)
        self.items.pop(key, None)   # fresh entries go last, old ones are evicted first
        self.items[key] = {'stamp': stamp, 'result': data}
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        items = dict(list(self.items.items())[-CACHE_LIMIT:])
        tmp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Through a temporary file: an interrupted write does not damage the old cache.
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix='.part')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump({'version': CACHE_VERSION, 'items': items}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            # the cache is only a speedup, everything works without it
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
