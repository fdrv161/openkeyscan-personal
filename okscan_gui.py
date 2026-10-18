"""
OpenKeyScan (личная сборка) — окно для определения тональности и BPM.

Сделано под скринридер: стандартные элементы Windows (NVDA и Экранный диктор
читают их сами), всё доступно с клавиатуры, итоги озвучиваются через NVDA.
В аудиофайлы программа ничего не пишет.
"""
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

# Кэш numba — в профиль пользователя: папка программы может быть недоступна для записи.
_APPDIR = Path(os.environ.get('APPDATA', Path.home())) / 'OpenKeyScan-Personal'
os.environ.setdefault('NUMBA_CACHE_DIR', str(_APPDIR / 'numba-cache'))

import wx  # noqa: E402

from okscan.analyzer import Analyzer, SUPPORTED_FORMATS, fold_bpm, format_bpm  # noqa: E402
from okscan.library import (Track, collect_audio, write_csv,  # noqa: E402
                            ST_QUEUED, ST_RUNNING, ST_DONE, ST_ERROR, ST_STOPPED)
from okscan import speech  # noqa: E402

APP_NAME = "OpenKeyScan — личная сборка"
VERSION = "1.0.0"
SETTINGS_FILE = _APPDIR / 'settings.json'

# Нижняя граница диапазона BPM -> подпись. Диапазон — [low, 2*low].
# По умолчанию 50–100: хип-хоп и трэп, всё быстрее 100 делится пополам (133 -> 66,5).
BPM_RANGES = [(0, "Без приведения"), (50, "50–100"), (60, "60–120"), (70, "70–140"), (78, "78–156"),
              (88, "88–176"), (98, "98–196")]
DEFAULT_SETTINGS = {'bpm_low': 50, 'workers': 2, 'last_dir': ''}

COLUMNS = [("Файл", 320), ("BPM", 70), ("Camelot", 80), ("Тональность", 120),
           ("Open Key", 80), ("Статус", 160), ("Папка", 300)]


def load_settings():
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
        return {**DEFAULT_SETTINGS, **{k: v for k, v in data.items() if k in DEFAULT_SETTINGS}}
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)


def save_settings(s):
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError:
        pass  # не критично: просто не запомним настройки


class TrackList(wx.ListCtrl):
    """Виртуальный список: строки берутся из frame.tracks, поэтому тысячи треков не тормозят."""

    def __init__(self, parent, frame):
        super().__init__(parent, style=wx.LC_REPORT | wx.LC_VIRTUAL | wx.LC_HRULES)
        self.frame = frame
        for i, (title, width) in enumerate(COLUMNS):
            self.InsertColumn(i, title, width=width)

    def OnGetItemText(self, item, col):
        t = self.frame.tracks[item]
        r = t.result
        if col == 0:
            return Path(t.path).name
        if col == 6:
            return str(Path(t.path).parent)
        if col == 5:
            return f"{ST_ERROR}: {r.error}" if t.status == ST_ERROR and r else t.status
        if not r or r.error:
            return ""
        if col == 1:
            return format_bpm(fold_bpm(r.bpm_raw, self.frame.settings['bpm_low']))
        return {2: r.camelot, 3: r.key_name, 4: r.openkey}[col]


class FileDrop(wx.FileDropTarget):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def OnDropFiles(self, x, y, filenames):
        self.frame.add_paths(filenames)
        return True


class MainFrame(wx.Frame):
    def __init__(self, initial_paths):
        super().__init__(None, title=APP_NAME, size=(1100, 650))
        self.settings = load_settings()
        self.tracks: list[Track] = []
        self.known_paths = set()
        self.order_counter = 0
        self.gen = 0                  # поколение: «Стоп» делает старые задания недействительными
        self.jobs = queue.Queue()
        self.analyzer = None
        self.batch_total = self.batch_done = self.batch_errors = 0
        self.batch_last: Track | None = None

        self._build_menu()
        panel = wx.Panel(self)
        # Подпись стоит прямо перед списком: по ней скринридер называет список.
        label = wx.StaticText(panel, label="&Треки:")
        self.list = TrackList(panel, self)
        # Текст окна списка не виден на экране, но это его имя для MSAA/скринридера.
        self.list.SetLabel("Треки")
        self.list.SetDropTarget(FileDrop(self))
        self.btn_files = wx.Button(panel, label="Добавить &файлы…")
        self.btn_folder = wx.Button(panel, label="Добавить &папку…")
        self.btn_export = wx.Button(panel, label="&Экспорт CSV…")
        self.btn_stop = wx.Button(panel, label="&Остановить анализ")

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        for b in (self.btn_files, self.btn_folder, self.btn_export, self.btn_stop):
            buttons.Add(b, 0, wx.RIGHT, 8)
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(label, 0, wx.ALL, 6)
        box.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)
        box.Add(buttons, 0, wx.ALL, 6)
        panel.SetSizer(box)
        self.CreateStatusBar()

        self.btn_files.Bind(wx.EVT_BUTTON, self.on_add_files)
        self.btn_folder.Bind(wx.EVT_BUTTON, self.on_add_folder)
        self.btn_export.Bind(wx.EVT_BUTTON, self.on_export)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop)
        self.list.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.set_status("Загрузка моделей… Файлы уже можно добавлять.")
        threading.Thread(target=self._load_analyzer, daemon=True).start()
        if initial_paths:
            self.add_paths(initial_paths)
        else:
            self.btn_files.SetFocus()

    # ---------- меню ----------

    def _build_menu(self):
        mb = wx.MenuBar()

        def item(menu, label, handler, kind=wx.ITEM_NORMAL):
            it = menu.Append(wx.ID_ANY, label, kind=kind)
            self.Bind(wx.EVT_MENU, handler, it)
            return it

        m = wx.Menu()
        item(m, "Добавить &файлы…\tCtrl+O", self.on_add_files)
        item(m, "Добавить &папку…\tCtrl+Shift+O", self.on_add_folder)
        item(m, "&Экспорт в CSV…\tCtrl+S", self.on_export)
        m.AppendSeparator()
        item(m, "В&ыход\tAlt+F4", lambda e: self.Close())
        mb.Append(m, "&Файл")

        m = wx.Menu()
        item(m, "&Копировать выбранные\tCtrl+C", self.on_copy)
        item(m, "Копировать &все\tCtrl+Shift+C", self.on_copy_all)
        item(m, "Показать в &папке\tCtrl+E", self.on_show_in_folder)
        m.AppendSeparator()
        item(m, "&Удалить из списка\tDel", self.on_remove)
        item(m, "&Очистить список\tCtrl+Shift+Del", self.on_clear)
        item(m, "Выделить в&се\tCtrl+A", self.on_select_all)
        mb.Append(m, "&Правка")

        m = wx.Menu()
        item(m, "&Остановить анализ\tEsc", self.on_stop)
        item(m, "Сказать &прогресс\tCtrl+I", self.on_say_progress)
        item(m, "Проанализировать &заново выбранные\tCtrl+R", self.on_reanalyze)
        mb.Append(m, "&Анализ")

        m = wx.Menu()
        item(m, "Порядок &добавления\tCtrl+0", lambda e: self.sort_by('order'))
        item(m, "По &имени\tCtrl+1", lambda e: self.sort_by('name'))
        item(m, "По &BPM\tCtrl+2", lambda e: self.sort_by('bpm'))
        item(m, "По &Camelot\tCtrl+3", lambda e: self.sort_by('camelot'))
        mb.Append(m, "&Сортировка")

        m = wx.Menu()
        rng = wx.Menu()
        for low, title in BPM_RANGES:
            it = item(rng, title, lambda e, low=low: self.set_bpm_range(low), wx.ITEM_RADIO)
            it.Check(low == self.settings['bpm_low'])
        m.AppendSubMenu(rng, "&Диапазон BPM")
        wk = wx.Menu()
        for n in (1, 2, 3, 4):
            it = item(wk, f"{n}", lambda e, n=n: self.set_workers(n), wx.ITEM_RADIO)
            it.Check(n == self.settings['workers'])
        m.AppendSubMenu(wk, "&Потоков анализа (после перезапуска)")
        mb.Append(m, "&Настройки")

        m = wx.Menu()
        item(m, "&О программе", self.on_about)
        mb.Append(m, "Спр&авка")
        self.SetMenuBar(mb)

    # ---------- загрузка и рабочие потоки ----------

    def _load_analyzer(self):
        workers = self.settings['workers']
        try:
            analyzer = Analyzer(workers=workers)
        except Exception as e:
            wx.CallAfter(self.error_box, f"Не удалось загрузить модели: {e}\n"
                                         "Переустановите программу.")
            return
        self.analyzer = analyzer
        for _ in range(workers):
            threading.Thread(target=self._worker, daemon=True).start()
        wx.CallAfter(self._on_loaded)

    def _on_loaded(self):
        if self.batch_total:
            self.update_status()
        else:
            self.set_status("Модели загружены. Добавьте файлы или папку: Ctrl+O или Ctrl+Shift+O.")

    def _worker(self):
        while True:
            track = self.jobs.get()
            if track.gen != self.gen:
                continue  # задание отменено кнопкой «Стоп»
            wx.CallAfter(self._set_track_status, track, ST_RUNNING)
            # Темп считаем без приведения к диапазону: диапазон применяется при показе,
            # поэтому его смена в настройках сразу видна без повторного анализа.
            res = self.analyzer.analyze(track.path, bpm_low=0)
            wx.CallAfter(self._on_result, track, res)

    def _set_track_status(self, track, status):
        if track.gen == self.gen:
            track.status = status
            self.refresh_track(track)

    def _on_result(self, track, res):
        track.result = res
        track.status = ST_ERROR if res.error else ST_DONE
        self.refresh_track(track)
        if track.gen != self.gen:
            return
        self.batch_done += 1
        self.batch_errors += bool(res.error)
        self.batch_last = track
        self.update_status()
        if self.batch_done >= self.batch_total:
            self.announce_batch_end()

    # ---------- добавление ----------

    def add_paths(self, paths):
        self.set_status("Ищу аудиофайлы…")
        threading.Thread(target=lambda: wx.CallAfter(self._add_files, collect_audio(paths)),
                         daemon=True).start()

    def _add_files(self, files):
        new = [f for f in files if os.path.normcase(f) not in self.known_paths]
        if not new:
            self.set_status("Новых аудиофайлов не найдено.")
            speech.speak("Новых аудиофайлов не найдено")
            return
        if self.batch_done >= self.batch_total:   # прошлая пачка закончена — начинаем счёт заново
            self.batch_total = self.batch_done = self.batch_errors = 0
        first_index = len(self.tracks)
        for f in new:
            self.known_paths.add(os.path.normcase(f))
            t = Track(path=f, order=self.order_counter, gen=self.gen)
            self.order_counter += 1
            self.tracks.append(t)
            self.jobs.put(t)
        self.batch_total += len(new)
        self.list.SetItemCount(len(self.tracks))
        self.list.SetFocus()
        self.select_only(first_index)
        self.update_status()
        speech.speak(f"Добавлено файлов: {len(new)}")

    def on_add_files(self, event):
        wildcard = "Аудиофайлы|" + ";".join(f"*{e}" for e in sorted(SUPPORTED_FORMATS)) + "|Все файлы|*.*"
        with wx.FileDialog(self, "Выберите аудиофайлы", defaultDir=self.settings['last_dir'],
                           wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.settings['last_dir'] = dlg.GetDirectory()
                save_settings(self.settings)
                self.add_paths(dlg.GetPaths())

    def on_add_folder(self, event):
        with wx.DirDialog(self, "Выберите папку с музыкой (подпапки тоже будут добавлены)",
                          defaultPath=self.settings['last_dir']) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.settings['last_dir'] = dlg.GetPath()
                save_settings(self.settings)
                self.add_paths([dlg.GetPath()])

    # ---------- анализ ----------

    def on_stop(self, event):
        pending = [t for t in self.tracks if t.status == ST_QUEUED and t.gen == self.gen]
        if not pending and self.batch_done >= self.batch_total:
            speech.speak("Анализ не идёт")
            return
        self.gen += 1
        for t in pending:
            t.status = ST_STOPPED
        self.batch_total = self.batch_done = self.batch_errors = 0
        self.list.RefreshItems(0, max(0, len(self.tracks) - 1))
        msg = f"Анализ остановлен. Не проанализировано: {len(pending)}. Повторить — Ctrl+R."
        self.set_status(msg)
        speech.speak(msg, interrupt=True)

    def on_reanalyze(self, event):
        tracks = self.selected_tracks()
        if not tracks:
            speech.speak("Ничего не выбрано")
            return
        if self.batch_done >= self.batch_total:
            self.batch_total = self.batch_done = self.batch_errors = 0
        for t in tracks:
            t.status, t.result, t.gen = ST_QUEUED, None, self.gen
            self.jobs.put(t)
        self.batch_total += len(tracks)
        self.list.RefreshItems(0, len(self.tracks) - 1)
        self.update_status()
        speech.speak(f"Поставлено в очередь: {len(tracks)}")

    def progress_text(self):
        if not self.batch_total:
            return f"Треков в списке: {len(self.tracks)}."
        text = f"Обработано {self.batch_done} из {self.batch_total}"
        if self.batch_errors:
            text += f", ошибок: {self.batch_errors}"
        if self.analyzer is None:
            text += ". Модели ещё загружаются"
        return text + "."

    def update_status(self):
        self.set_status(self.progress_text())

    def on_say_progress(self, event):
        text = self.progress_text()
        if not speech.speak(text, interrupt=True):
            wx.MessageBox(text, APP_NAME, wx.OK | wx.ICON_INFORMATION, self)

    def announce_batch_end(self):
        if self.batch_total == 1 and self.batch_last:
            msg = self.describe(self.batch_last)
        else:
            msg = f"Анализ завершён: {self.batch_total} треков"
            if self.batch_errors:
                msg += f", ошибок: {self.batch_errors}. Ошибки видны в столбце «Статус»"
            msg += "."
        self.set_status(msg)
        # Без NVDA сообщаем окном: его прочитает любой скринридер.
        if not speech.speak(msg):
            wx.MessageBox(msg, APP_NAME, wx.OK | wx.ICON_INFORMATION, self)

    # ---------- список ----------

    def refresh_track(self, track):
        try:
            i = self.tracks.index(track)
        except ValueError:
            return  # трек уже удалён из списка
        self.list.RefreshItem(i)

    def selected_indices(self):
        out, i = [], self.list.GetFirstSelected()
        while i != -1:
            out.append(i)
            i = self.list.GetNextSelected(i)
        return out

    def selected_tracks(self):
        return [self.tracks[i] for i in self.selected_indices()]

    def select_only(self, index):
        for i in self.selected_indices():
            self.list.Select(i, False)
        if 0 <= index < len(self.tracks):
            self.list.Select(index)
            self.list.Focus(index)

    def on_select_all(self, event):
        for i in range(len(self.tracks)):
            self.list.Select(i)

    def describe(self, t):
        """Строка трека для буфера обмена и озвучки."""
        name = Path(t.path).name
        r = t.result
        if r is None:
            return f"{name}: {t.status}"
        if r.error:
            return f"{name}: ошибка — {r.error}"
        bpm = format_bpm(fold_bpm(r.bpm_raw, self.settings['bpm_low'])) or "нет"
        return f"{name}: BPM {bpm}, {r.camelot}, {r.key_name}, Open Key {r.openkey}"

    def copy_text(self, tracks):
        if not tracks:
            speech.speak("Нечего копировать")
            return
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject("\r\n".join(self.describe(t) for t in tracks)))
            wx.TheClipboard.Close()
            speech.speak(f"Скопировано строк: {len(tracks)}")

    def on_copy(self, event):
        self.copy_text(self.selected_tracks())

    def on_copy_all(self, event):
        self.copy_text(self.tracks)

    def on_show_in_folder(self, event):
        tracks = self.selected_tracks()
        if tracks:
            subprocess.Popen(['explorer', '/select,', tracks[0].path])

    def on_remove(self, event):
        idx = self.selected_indices()
        if not idx:
            speech.speak("Ничего не выбрано")
            return
        for i in reversed(idx):
            t = self.tracks.pop(i)
            self.known_paths.discard(os.path.normcase(t.path))
            if t.status in (ST_QUEUED, ST_RUNNING) and t.gen == self.gen:
                # трек больше не ждём в счётчике пачки
                t.gen = -1
                self.batch_total -= 1
        self.list.SetItemCount(len(self.tracks))
        self.list.RefreshItems(0, max(0, len(self.tracks) - 1))
        self.select_only(min(idx[0], len(self.tracks) - 1))
        speech.speak(f"Удалено из списка: {len(idx)}. Файлы на диске не тронуты.")
        self.update_status()

    def on_clear(self, event):
        if not self.tracks:
            return
        if wx.MessageBox("Очистить весь список? Файлы на диске не будут затронуты.", APP_NAME,
                         wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION, self) != wx.YES:
            return
        self.gen += 1
        self.tracks.clear()
        self.known_paths.clear()
        self.batch_total = self.batch_done = self.batch_errors = 0
        self.list.SetItemCount(0)
        self.set_status("Список очищен.")
        self.btn_files.SetFocus()

    def sort_by(self, key):
        focused = self.list.GetFocusedItem()
        focused_track = self.tracks[focused] if 0 <= focused < len(self.tracks) else None
        low = self.settings['bpm_low']

        def bpm_key(t):
            b = fold_bpm(t.result.bpm_raw, low) if t.result and t.result.bpm_raw else None
            return (b is None, b or 0)

        def camelot_key(t):
            c = t.result.camelot if t.result and t.result.camelot else ''
            return (not c, int(c[:-1]) if c else 0, c[-1:])

        keys = {'order': lambda t: t.order, 'name': lambda t: Path(t.path).name.lower(),
                'bpm': bpm_key, 'camelot': camelot_key}
        self.tracks.sort(key=keys[key])
        self.list.RefreshItems(0, max(0, len(self.tracks) - 1))
        if focused_track:
            self.select_only(self.tracks.index(focused_track))
        names = {'order': "в порядке добавления", 'name': "по имени", 'bpm': "по BPM", 'camelot': "по Camelot"}
        speech.speak(f"Отсортировано {names[key]}")

    def on_context_menu(self, event):
        m = wx.Menu()
        for label, handler in (("Копировать\tCtrl+C", self.on_copy),
                               ("Показать в папке\tCtrl+E", self.on_show_in_folder),
                               ("Проанализировать заново\tCtrl+R", self.on_reanalyze),
                               ("Удалить из списка\tDel", self.on_remove)):
            it = m.Append(wx.ID_ANY, label)
            self.Bind(wx.EVT_MENU, handler, it)
        self.list.PopupMenu(m)
        m.Destroy()

    # ---------- экспорт и настройки ----------

    def on_export(self, event):
        if not self.tracks:
            wx.MessageBox("Список пуст — нечего экспортировать.", APP_NAME, wx.OK | wx.ICON_INFORMATION, self)
            return
        with wx.FileDialog(self, "Сохранить результаты в CSV", defaultDir=self.settings['last_dir'],
                           defaultFile="openkeyscan.csv", wildcard="CSV (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        try:
            write_csv(path, self.tracks, self.settings['bpm_low'])
        except OSError as e:
            self.error_box(f"Не удалось сохранить файл: {e}")
            return
        msg = f"Сохранено строк: {len(self.tracks)}."
        self.set_status(msg)
        if not speech.speak(msg):
            wx.MessageBox(msg, APP_NAME, wx.OK | wx.ICON_INFORMATION, self)

    def set_bpm_range(self, low):
        self.settings['bpm_low'] = low
        save_settings(self.settings)
        if self.tracks:
            self.list.RefreshItems(0, len(self.tracks) - 1)
        title = dict(BPM_RANGES)[low]
        speech.speak(f"Диапазон BPM: {title}")

    def set_workers(self, n):
        self.settings['workers'] = n
        save_settings(self.settings)
        speech.speak(f"Потоков: {n}. Применится после перезапуска программы.")

    def on_about(self, event):
        wx.MessageBox(
            f"{APP_NAME}, версия {VERSION}.\n\n"
            "Тональность: модель OpenKeyScan (MIT, rekordcloud, на основе MusicalKeyCNN).\n"
            "Темп: DeepRhythm 0.7 (AGPL-3.0, bleugreen) с уточнением до десятых.\n"
            "Озвучка: NVDA Controller Client (LGPL-2.1, NV Access).\n\n"
            "Программа только читает аудиофайлы и ничего в них не записывает.",
            "О программе", wx.OK | wx.ICON_INFORMATION, self)

    def set_status(self, text):
        self.SetStatusText(text)

    def error_box(self, text):
        self.set_status(text)
        wx.MessageBox(text, APP_NAME, wx.OK | wx.ICON_ERROR, self)

    def on_close(self, event):
        busy = self.batch_done < self.batch_total
        if busy and event.CanVeto():
            if wx.MessageBox("Анализ ещё идёт. Всё равно выйти?", APP_NAME,
                             wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION, self) != wx.YES:
                event.Veto()
                return
        self.Destroy()


def main():
    app = wx.App()
    frame = MainFrame(sys.argv[1:])
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
