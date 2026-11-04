# OpenKeyScan for REAPER

Detects the key and tempo (BPM) of selected items right in REAPER. There is no window: select an item, press a shortcut, get the answer, for example "beat: 88 BPM, 7A, D minor". Audio files are only read, nothing is ever written to them.

The project is based on [OpenKeyScan](https://github.com/rekordcloud/openkeyscan-analyzer) by rekordcloud: the key model comes from the original, tempo detection, REAPER scripts and a console engine for Windows are added on top.

## Download and install

The `OpenKeyScan-REAPER-<version>.zip` archive is in [Releases](https://github.com/fdrv161/openkeyscan-reaper/releases). Unpack the OpenKeyScan folder into REAPER's Scripts folder, load "OpenKeyScan - Install actions.lua" through the action list and run it once. Step-by-step instructions and all the actions are described in [reaper/README.md](reaper/README.md), the same file is inside the archive.

Requirements: Windows 10 or 11 (64-bit), REAPER 6 or newer, about 200 MB of free disk space.

## Features

- Key in three notations: regular name (D minor), Camelot (7A) and Open Key (12m).
- Tempo to two decimal places, folded into a convenient range: by default 50 to 100 BPM, so 133 BPM is shown as 66.5. The range is switched by a separate action.
- Sets the project tempo from the beat, appends the key to the track name and moves items so that the first downbeat lands on a bar line. All of it is undone with a single Ctrl+Z.
- Remembers results: analyzing the same file again takes a fraction of a second.
- Screen reader friendly: with OSARA installed in REAPER, all messages are spoken by NVDA, JAWS or Narrator, no windows appear and the focus stays where it was.

## How it works

The archive has two layers: Lua scripts for REAPER and the `okscan-cli.exe` engine in the engine folder. A script passes a list of files to the engine, waits for the results file and reports the answer.

For each file the engine:

1. Decodes the audio through PyAV (FFmpeg inside) to mono 44.1 kHz.
2. Key: a CQT spectrum and the OpenKeyScan convolutional network (24 classes: 12 tonics, major and minor).
3. Tempo: HCQM features and the [DeepRhythm](https://github.com/bleugreen/deeprhythm) network give an integer BPM, which is then refined to hundredths by onset autocorrelation.
4. First downbeat: found from kick and bass attacks, then a grid is fitted through the exact start of each hit so that it passes through as many hits as possible.

Both networks are converted to ONNX and run through onnxruntime, audio processing is written in numpy. PyTorch and librosa are not needed, so the program takes about 180 MB and analyzes a track in a couple of seconds.

## Console engine

The engine also works without REAPER:

```
okscan-cli "D:\Music\track.mp3"
okscan-cli "D:\Music" --csv result.csv
okscan-cli "D:\Music" --range 0
okscan-cli --selftest
```

Folders are scanned with all subfolders. `--range LOW` folds the tempo into the range from LOW to 2*LOW (default 50, `0` means no folding), `--csv` saves a table, `--no-cache` analyzes again. All options with examples: `okscan-cli --help`. Exit code 0: everything analyzed, 1: errors in some files, 2: wrong parameters or nothing found.

## Building from source

You need Python 3.14 on Windows x64.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-win.lock.txt
.venv\Scripts\python -m okscan.selftest
.venv\Scripts\python -m PyInstaller --noconfirm okscan.spec
```

The engine appears in `dist\OpenKeyScan`. The REAPER scripts in the `reaper` folder find it there by themselves, so for development they can be loaded straight from the repository.

Working with the networks takes a separate set of dependencies, `requirements-dev.txt` (PyTorch, librosa, nnAudio):

- `tools/export_onnx.py` converts the networks to `models/*.onnx` again if the weights change.
- `tools/check_parity.py` compares the numpy audio processing with librosa and nnAudio, and the final answers with the result of an older build.
- `tools/check_lua.py` checks the syntax of the REAPER scripts.

## Releasing

The GitHub Actions build runs on a `vX.Y.Z` tag: it tests the engine, builds the archive for REAPER and publishes it in Releases. Without a tag the build can be started by hand (Actions, then Run workflow), the archive is then kept for 7 days.

## What is where

- `reaper`: REAPER scripts and their help.
- `okscan`: the engine: analysis, audio processing, cache, self-test.
- `okscan_cli.py`: the console program, `okscan.spec`: its PyInstaller build.
- `models`: the key and tempo networks in ONNX format.
- `tools`: the helper scripts described above.
- `model.py`, `dataset.py`, `train.py`, `eval.py`, `preprocess_data.py`, `checkpoints` and `test`: from the original OpenKeyScan, for training and evaluating the key model. The engine does not use them.

## Licenses and credits

- Key model: [OpenKeyScan](https://github.com/rekordcloud/openkeyscan-analyzer) (rekordcloud), based on [MusicalKeyCNN](https://github.com/a1ex90/MusicalKeyCNN) (a1ex90), MIT license. The architecture is described in F. Korzeniowski, G. Widmer, "Genre-Agnostic Key Classification With Convolutional Neural Networks", ISMIR 2018 ([arXiv](https://arxiv.org/abs/1808.05340)).
- Tempo detection: [DeepRhythm](https://github.com/bleugreen/deeprhythm) 0.7 (bleugreen), AGPL-3.0 license.
- Audio processing is ported from [librosa](https://github.com/librosa/librosa) (ISC) and [nnAudio](https://github.com/KinWaiCheuk/nnAudio) (MIT).

Because of DeepRhythm, the built program as a whole is distributed under AGPL-3.0: if you pass it on, pass on the source code too. The license texts are in `LICENSE` and `okscan/deeprhythm/LICENSE-deeprhythm`, and in the built program in the `engine\_internal\licenses` folder.
