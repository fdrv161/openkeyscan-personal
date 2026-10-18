"""
Озвучка сообщений через NVDA (nvdaControllerClient.dll, LGPL-2.1).
Если NVDA не запущена или библиотеки нет — speak() возвращает False,
и программа сообщает о результате другим способом (окно-сообщение).
"""
import ctypes
import threading

from okscan.analyzer import resource_path

_lib = None
_lock = threading.Lock()


def _load():
    global _lib
    if _lib is None:
        try:
            _lib = ctypes.windll.LoadLibrary(str(resource_path('assets/nvdaControllerClient.dll')))
        except OSError:
            _lib = False
    return _lib


def nvda_running():
    lib = _load()
    return bool(lib) and lib.nvdaController_testIfRunning() == 0


def speak(text, interrupt=False):
    """Произносит текст через NVDA. Возвращает True, если NVDA его приняла."""
    with _lock:
        if not nvda_running():
            return False
        if interrupt:
            _lib.nvdaController_cancelSpeech()
        return _lib.nvdaController_speakText(ctypes.c_wchar_p(text)) == 0
