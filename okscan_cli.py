"""
Консольная версия: тональность и BPM для файлов и папок.

Примеры:
  okscan-cli "D:\\Music\\track.mp3"
  okscan-cli "D:\\Music" --csv result.csv
  okscan-cli "D:\\Music" --range 70
  okscan-cli --selftest
  okscan-cli --list files.txt --out result.tsv   (для скрипта REAPER)

Коды выхода: 0 — всё проанализировано, 1 — были ошибки в отдельных файлах,
2 — неверные параметры или ничего не найдено.
"""
import argparse
import os
import sys
from pathlib import Path

_APPDIR = Path(os.environ.get('APPDATA', Path.home())) / 'OpenKeyScan-Personal'
os.environ.setdefault('NUMBA_CACHE_DIR', str(_APPDIR / 'numba-cache'))


def main():
    # Вывод в UTF-8, иначе русские имена файлов ломаются при перенаправлении в файл.
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(
        prog='okscan-cli',
        description="Определяет тональность (Camelot, Open Key) и темп (BPM) аудиофайлов. "
                    "Файлы только читаются, теги не пишутся.",
        epilog="Примеры:\n  okscan-cli track.mp3\n  okscan-cli D:\\Music --csv result.csv\n"
               "  okscan-cli D:\\Music --range 0   (без приведения BPM)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='*', help="файлы или папки (папки — со всеми подпапками)")
    p.add_argument('--range', type=int, default=50, metavar='LOW',
                   help="приводить BPM к диапазону от LOW до 2×LOW, по умолчанию 50 (50–100, быстрее 100 делится пополам); 0 — не приводить")
    p.add_argument('--csv', metavar='FILE', help="сохранить результаты в CSV (разделитель «;»)")
    p.add_argument('--workers', type=int, default=1, help="сколько файлов анализировать одновременно (1–8)")
    p.add_argument('--selftest', action='store_true', help="проверить движок на синтетических треках")
    p.add_argument('--list', metavar='FILE', help="список файлов в UTF-8, по одному пути в строке")
    p.add_argument('--out', metavar='FILE',
                   help="записать результаты в TSV (UTF-8): путь, BPM, Camelot, тональность, Open Key, ошибка, первая сильная доля (с). "
                        "Файл появляется целиком в самом конце — по нему скрипт понимает, что анализ закончен")
    args = p.parse_args()
    if args.out:
        # Что бы ни случилось, файл результата должен появиться — иначе скрипт REAPER ждал бы впустую.
        try:
            return run(p, args)
        except Exception as e:
            write_tsv(args.out, [['ERROR', '', '', '', '', f"сбой анализатора: {e}"]])
            return 2
    return run(p, args)


def write_tsv(path, rows):
    """Пишет во временный файл и переименовывает: читатель никогда не увидит половину результата."""
    tmp = f"{path}.part"
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write('\t'.join(str(c).replace('\t', ' ').replace('\n', ' ') for c in row) + '\n')
    os.replace(tmp, path)


def run(p, args):
    if args.selftest:
        from okscan.selftest import main as selftest
        return selftest()
    listed = []
    if args.list:
        listed = [ln.strip() for ln in Path(args.list).read_text(encoding='utf-8-sig').splitlines() if ln.strip()]
    if not args.paths and not listed:
        if args.out:
            write_tsv(args.out, [['ERROR', '', '', '', '', "не передано ни одного файла"]])
        else:
            p.print_help()
        return 2

    from concurrent.futures import ThreadPoolExecutor
    from okscan.analyzer import Analyzer, format_bpm
    from okscan.library import collect_audio, write_csv, Track

    # Файлы из --list берём как есть: о неподдерживаемом формате анализатор скажет сам.
    files = listed + collect_audio(args.paths)
    if not files:
        print("Ошибка: не найдено ни одного поддерживаемого аудиофайла. Проверьте путь.", file=sys.stderr)
        if args.out:
            write_tsv(args.out, [['ERROR', '', '', '', '', "не найдено ни одного аудиофайла"]])
        return 2
    workers = max(1, min(8, args.workers))
    print(f"Найдено файлов: {len(files)}. Загружаю модели…", file=sys.stderr)
    analyzer = Analyzer(workers=workers)

    tracks, errors = [], 0
    with ThreadPoolExecutor(workers) as pool:
        for n, r in enumerate(pool.map(lambda f: analyzer.analyze(f, bpm_low=args.range), files), 1):
            name = Path(r.path).name
            if r.error:
                errors += 1
                print(f"Файл: {name}, ошибка: {r.error}")
            else:
                print(f"Файл: {name}, BPM: {format_bpm(r.bpm) or 'нет'}, Camelot: {r.camelot}, "
                      f"Тональность: {r.key_name}, Open Key: {r.openkey}")
            tracks.append(Track(path=r.path, order=n, result=r))
            # редкие строки прогресса вместо перерисовки — так удобнее скринридеру
            if n % 25 == 0 and n < len(files):
                print(f"Обработано {n} из {len(files)}", file=sys.stderr)

    if args.out:
        write_tsv(args.out, [[t.result.path, format_bpm(t.result.bpm), t.result.camelot, t.result.key_name,
                              t.result.openkey, t.result.error,
                              "" if t.result.downbeat is None else f"{t.result.downbeat:.4f}"] for t in tracks])
    if args.csv:
        write_csv(args.csv, tracks, args.range)
        print(f"Результаты сохранены в {args.csv}", file=sys.stderr)
    print(f"Готово. Файлов: {len(files)}, ошибок: {errors}.", file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
