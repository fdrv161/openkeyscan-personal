# -*- mode: python ; coding: utf-8 -*-
# Сборка личной версии под Windows: окно OpenKeyScan.exe и консольная okscan-cli.exe
# в одной папке с общими библиотеками. Запуск: pyinstaller --noconfirm okscan.spec
from pathlib import Path

root = Path.cwd()

datas = [
    (str(root / 'checkpoints' / 'openkeyscan3.pt'), 'checkpoints'),
    (str(root / 'assets' / 'deeprhythm-0.7.pth'), 'assets'),
    (str(root / 'assets' / 'nvdaControllerClient.dll'), 'assets'),
    (str(root / 'assets' / 'nvdaControllerClient-license.txt'), 'licenses'),
    (str(root / 'okscan' / 'deeprhythm' / 'LICENSE-deeprhythm'), 'licenses'),
    (str(root / 'LICENSE'), 'licenses'),
]

# Ненужное для анализа: графика, тесты, оболочки — только раздувают сборку.
excludes = ['tkinter', 'matplotlib', 'IPython', 'jupyter', 'pytest', 'torchaudio',
            'torchvision', 'tensorboard', 'pandas']

a = Analysis(
    ['okscan_gui.py', 'okscan_cli.py'],
    pathex=[str(root)],
    datas=datas,
    hiddenimports=['okscan.selftest', 'soxr'],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

gui = EXE(
    pyz,
    [s for s in a.scripts if s[0] != 'okscan_cli'],
    [],
    exclude_binaries=True,
    name='OpenKeyScan',
    console=False,
    upx=False,
)
cli = EXE(
    pyz,
    [s for s in a.scripts if s[0] != 'okscan_gui'],
    [],
    exclude_binaries=True,
    name='okscan-cli',
    console=True,
    upx=False,
)
coll = COLLECT(gui, cli, a.binaries, a.datas, name='OpenKeyScan', upx=False)
