# -*- mode: python ; coding: utf-8 -*-
# Сборка движка для скриптов REAPER: консольная okscan-cli.exe со всеми библиотеками
# в одной папке. Запуск: pyinstaller --noconfirm okscan.spec
from pathlib import Path

root = Path.cwd()

datas = [
    (str(root / 'models' / 'keynet.onnx'), 'models'),
    (str(root / 'models' / 'deeprhythm.onnx'), 'models'),
    (str(root / 'okscan' / 'deeprhythm' / 'LICENSE-deeprhythm'), 'licenses'),
    (str(root / 'LICENSE'), 'licenses'),
]

# Ненужное для анализа: графика, тесты, оболочки — только раздувают сборку.
# torch, librosa и прочее для разработчика стоят в той же .venv — не даём им попасть внутрь.
excludes = ['tkinter', 'matplotlib', 'IPython', 'jupyter', 'pytest', 'pandas', 'wx',
            'torch', 'torchaudio', 'torchvision', 'tensorboard', 'nnAudio', 'onnx',
            'librosa', 'numba', 'llvmlite', 'scipy', 'sklearn', 'soundfile']

a = Analysis(
    ['okscan_cli.py'],
    pathex=[str(root)],
    datas=datas,
    hiddenimports=['okscan.selftest', 'soxr'],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='okscan-cli',
    console=True,
    upx=False,
)
coll = COLLECT(cli, a.binaries, a.datas, name='OpenKeyScan', upx=False)
