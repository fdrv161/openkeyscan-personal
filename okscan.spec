# -*- mode: python ; coding: utf-8 -*-
# Builds the engine for the REAPER scripts: console okscan-cli.exe with all libraries
# in one folder. Run: pyinstaller --noconfirm okscan.spec
from pathlib import Path

root = Path.cwd()

datas = [
    (str(root / 'models' / 'keynet.onnx'), 'models'),
    (str(root / 'models' / 'deeprhythm.onnx'), 'models'),
    (str(root / 'okscan' / 'deeprhythm' / 'LICENSE-deeprhythm'), 'licenses'),
    (str(root / 'LICENSE'), 'licenses'),
]

# Not needed for analysis: graphics, tests, shells only bloat the build.
# torch, librosa and other developer tools live in the same .venv: keep them out of the build.
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
