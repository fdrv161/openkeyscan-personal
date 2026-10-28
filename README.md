# OpenKeyScan для REAPER

Определяет тональность и темп (BPM) выбранных элементов прямо в REAPER. Своего окна нет: выделили элемент, нажали горячую клавишу, получили ответ, например «beat: 88 BPM, 7A, D minor». Аудиофайлы только читаются, в них ничего не записывается.

Это личная сборка на основе [OpenKeyScan](https://github.com/rekordcloud/openkeyscan-analyzer) от rekordcloud: от оригинала взята модель тональности, к ней добавлены определение темпа, скрипты REAPER и консольный движок под Windows.

## Скачать и установить

Архив `OpenKeyScan-REAPER-<версия>.zip` лежит в разделе [Releases](https://github.com/fdrv161/openkeyscan-personal/releases). Распакуйте папку OpenKeyScan из архива в папку Scripts REAPER и загрузите три действия через список действий. Подробная инструкция и описание всех действий — в [reaper/README.md](reaper/README.md), этот же файл лежит в архиве.

Требования: Windows 10 или 11 (64 бита), REAPER 6 или новее, около 200 МБ свободного места.

## Что умеет

- Тональность в трёх записях: обычное название (D minor), Camelot (7A) и Open Key (12m).
- Темп с точностью до сотых, приведённый к удобному диапазону: по умолчанию 50–100 BPM, то есть 133 BPM показывается как 66,5. Диапазон переключается отдельным действием.
- Выставляет темп проекта по биту, дописывает тональность в имя дорожки и сдвигает элементы так, чтобы первая сильная доля легла на линию такта. Всё отменяется одним Ctrl+Z.
- Запоминает результаты: повторный анализ того же файла занимает доли секунды.
- Со скринридером: если в REAPER установлена OSARA, все сообщения произносятся NVDA, JAWS или Экранным диктором, окна не появляются и фокус остаётся на месте.

## Как это устроено

В архиве два слоя: скрипты на Lua для REAPER и движок `okscan-cli.exe` в папке engine. Скрипт передаёт движку список файлов, ждёт файл с результатами и озвучивает ответ.

Движок для каждого файла:

1. Декодирует звук через PyAV (внутри FFmpeg) в моно 44,1 кГц.
2. Тональность: спектр CQT и свёрточная сеть OpenKeyScan (24 класса: 12 тоник, мажор и минор).
3. Темп: признаки HCQM и сеть [DeepRhythm](https://github.com/bleugreen/deeprhythm) дают целый BPM, затем он уточняется до сотых по автокорреляции атак.
4. Первая сильная доля: ищется по атакам бочки и баса, затем по точному началу каждого удара подбирается сетка, через которую проходит больше всего ударов.

Обе сети переведены в формат ONNX и запускаются через onnxruntime, обработка звука написана на numpy. PyTorch и librosa в программе не нужны, поэтому она весит около 180 МБ и анализирует трек за пару секунд.

## Консольный движок

Движок можно запускать и без REAPER:

```
okscan-cli "D:\Music\track.mp3"
okscan-cli "D:\Music" --csv result.csv
okscan-cli "D:\Music" --range 0
okscan-cli --selftest
```

Папки обходятся вместе с подпапками. `--range LOW` приводит темп к диапазону от LOW до 2×LOW (по умолчанию 50, `0` — не приводить), `--csv` сохраняет таблицу, `--no-cache` анализирует заново. Все параметры с примерами: `okscan-cli --help`. Код выхода 0 — всё проанализировано, 1 — были ошибки в отдельных файлах, 2 — неверные параметры или ничего не найдено.

## Сборка из исходников

Нужен Python 3.14 под Windows x64.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-win.lock.txt
.venv\Scripts\python -m okscan.selftest
.venv\Scripts\python -m PyInstaller --noconfirm okscan.spec
```

Готовый движок появится в `dist\OpenKeyScan`. Скрипты REAPER из папки `reaper` находят его там сами, так что для разработки их можно загружать прямо из репозитория.

Для работы с сетями есть отдельный набор зависимостей `requirements-dev.txt` (PyTorch, librosa, nnAudio):

- `tools/export_onnx.py` — заново переводит сети в `models/*.onnx`, если поменялись веса.
- `tools/check_parity.py` — сверяет обработку звука на numpy с librosa и nnAudio, а итоговые ответы — с результатом прошлой сборки.
- `tools/check_lua.py` — проверяет синтаксис скриптов REAPER.

## Выпуск версии

Сборка на GitHub Actions запускается по тегу `vX.Y.Z`: она проверяет движок, собирает архив для REAPER и выкладывает его в Releases. Без тега сборку можно запустить вручную (Actions, затем Run workflow), архив тогда хранится 7 дней.

## Что где лежит

- `reaper` — скрипты для REAPER и справка к ним.
- `okscan` — движок: анализ, обработка звука, кэш, самопроверка.
- `okscan_cli.py` — консольная программа, `okscan.spec` — её сборка в PyInstaller.
- `models` — сети тональности и темпа в формате ONNX.
- `tools` — служебные скрипты, описанные выше.
- `model.py`, `dataset.py`, `train.py`, `eval.py`, `preprocess_data.py`, `checkpoints` и `test` — из оригинального OpenKeyScan: обучение и проверка модели тональности. Движком не используются.

## Лицензии и благодарности

- Модель тональности: [OpenKeyScan](https://github.com/rekordcloud/openkeyscan-analyzer) (rekordcloud) на основе [MusicalKeyCNN](https://github.com/a1ex90/MusicalKeyCNN) (a1ex90), лицензия MIT. Архитектура описана в статье F. Korzeniowski, G. Widmer, «Genre-Agnostic Key Classification With Convolutional Neural Networks», ISMIR 2018 ([arXiv](https://arxiv.org/abs/1808.05340)).
- Определение темпа: [DeepRhythm](https://github.com/bleugreen/deeprhythm) 0.7 (bleugreen), лицензия AGPL-3.0.
- Обработка звука перенесена из [librosa](https://github.com/librosa/librosa) (ISC) и [nnAudio](https://github.com/KinWaiCheuk/nnAudio) (MIT).

Из-за DeepRhythm собранная программа целиком распространяется по AGPL-3.0: если передаёте её другим, передавайте и исходники. Тексты лицензий лежат в `LICENSE` и `okscan/deeprhythm/LICENSE-deeprhythm`, в собранной программе — в папке `engine\_internal\licenses`.
