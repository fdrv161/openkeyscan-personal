# -*- mode: python ; coding: utf-8 -*-
# Сборка движка для скриптов REAPER: консольная okscan-cli.exe со всеми библиотеками
# в одной папке. Запуск: pyinstaller --noconfirm okscan.spec
from pathlib import Path

root = Path.cwd()

datas = [
    (str(root / 'checkpoints' / 'openkeyscan3.pt'), 'checkpoints'),
    (str(root / 'assets' / 'deeprhythm-0.7.pth'), 'assets'),
    (str(root / 'okscan' / 'deeprhythm' / 'LICENSE-deeprhythm'), 'licenses'),
    (str(root / 'LICENSE'), 'licenses'),
]

# Ненужное для анализа: графика, тесты, оболочки — только раздувают сборку.
excludes = ['tkinter', 'matplotlib', 'IPython', 'jupyter', 'pytest', 'torchaudio',
            'torchvision', 'tensorboard', 'pandas', 'wx']

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
