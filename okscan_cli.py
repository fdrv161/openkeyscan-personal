"""
Console version: key and BPM for files and folders.

Examples:
  okscan-cli "D:\\Music\\track.mp3"
  okscan-cli "D:\\Music" --csv result.csv
  okscan-cli "D:\\Music" --range 70
  okscan-cli --selftest
  okscan-cli --list files.txt --out result.tsv   (for the REAPER script)

Exit codes: 0: everything analyzed, 1: errors in some files,
2: wrong parameters or nothing found.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path


def _data_dir():
    """Cache folder next to the program: everything sits in one folder and is deleted with
    it. If that folder is not writable, the user profile is used."""
    base = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    for d in (base / 'cache', Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'OpenKeyScan'):
        try:
            d.mkdir(parents=True, exist_ok=True)
            tempfile.TemporaryFile(dir=d).close()   # write check: os.access does not see folder permissions
            return d
        except OSError:
            continue
    return Path(tempfile.gettempdir())


DATA_DIR = _data_dir()


def main():
    # UTF-8 output, otherwise non-English file names break when redirected to a file.
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(
        prog='okscan-cli',
        description="Detects the key (Camelot, Open Key) and tempo (BPM) of audio files. "
                    "Files are only read, no tags are written.",
        epilog="Examples:\n  okscan-cli track.mp3\n  okscan-cli D:\\Music --csv result.csv\n"
               "  okscan-cli D:\\Music --range 0   (no BPM folding)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='*', help="files or folders (folders with all subfolders)")
    p.add_argument('--range', type=int, default=50, metavar='LOW',
                   help="fold BPM into the range from LOW to 2*LOW, default 50 (50 to 100, anything faster is halved); 0: no folding")
    p.add_argument('--csv', metavar='FILE', help="save results to CSV (semicolon separated)")
    p.add_argument('--workers', type=int, default=1, help="how many files to analyze at once (1 to 8)")
    p.add_argument('--no-cache', action='store_true',
                   help="analyze again even if the file is in the cache (the cache folder next to the program)")
    p.add_argument('--selftest', action='store_true', help="test the engine on synthetic tracks")
    p.add_argument('--list', metavar='FILE', help="list of files in UTF-8, one path per line")
    p.add_argument('--out', metavar='FILE',
                   help="write results to TSV (UTF-8): path, BPM, Camelot, key, Open Key, error, first downbeat (s), start of sound in the file (s), BPM confidence (0 to 1). "
                        "The file appears whole at the very end: that is how the script knows the analysis is finished")
    args = p.parse_args()
    if args.out:
        # Whatever happens, the result file must appear, otherwise the REAPER script would wait in vain.
        try:
            return run(p, args)
        except Exception as e:
            write_tsv(args.out, [['ERROR', '', '', '', '', f"analyzer failure: {e}"]])
            return 2
    return run(p, args)


def write_tsv(path, rows):
    """Writes to a temporary file and renames it: a reader never sees half a result."""
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
            write_tsv(args.out, [['ERROR', '', '', '', '', "no files were given"]])
        else:
            p.print_help()
        return 2

    from concurrent.futures import ThreadPoolExecutor
    from okscan.library import ResultCache, collect_audio, format_bpm, write_csv, Track

    # Files from --list are taken as is: the analyzer reports an unsupported format itself.
    files = listed + collect_audio(args.paths)
    if not files:
        print("Error: no supported audio files found. Check the path.", file=sys.stderr)
        if args.out:
            write_tsv(args.out, [['ERROR', '', '', '', '', "no audio files found"]])
        return 2
    workers = max(1, min(8, args.workers))
    cache = ResultCache(DATA_DIR / 'results.json')
    cached = [None if args.no_cache else cache.get(f, args.range) for f in files]
    todo = [f for f, r in zip(files, cached) if r is None]
    print(f"Files found: {len(files)}, already in cache: {len(files) - len(todo)}.", file=sys.stderr)

    tracks, errors = [], 0
    with ThreadPoolExecutor(workers) as pool:
        fresh = iter(())
        if todo:
            # Load the heavy libraries only if there is something to compute: cached answers are instant.
            print("Loading models...", file=sys.stderr)
            from okscan.analyzer import Analyzer
            analyzer = Analyzer(workers=workers)
            fresh = pool.map(lambda f: analyzer.analyze(f, bpm_low=args.range), todo)
        for n, (f, r) in enumerate(zip(files, cached), 1):
            if r is None:
                r = next(fresh)
                cache.put(f, r)
            name = Path(r.path).name
            if r.error:
                errors += 1
                print(f"File: {name}, error: {r.error}")
            else:
                print(f"File: {name}, BPM: {format_bpm(r.bpm) or 'none'}, Camelot: {r.camelot}, "
                      f"Key: {r.key_name}, Open Key: {r.openkey}")
            tracks.append(Track(path=r.path, order=n, result=r))
            # occasional progress lines instead of redrawing a line: readable in a log and with a screen reader
            if n % 25 == 0 and n < len(files):
                print(f"Processed {n} of {len(files)}", file=sys.stderr)
    cache.save()

    if args.out:
        write_tsv(args.out, [[t.result.path, format_bpm(t.result.bpm), t.result.camelot, t.result.key_name,
                              t.result.openkey, t.result.error,
                              "" if t.result.downbeat is None else f"{t.result.downbeat:.4f}",
                              "" if t.result.lead is None else f"{t.result.lead:.6f}",
                              "" if t.result.bpm_confidence is None else f"{t.result.bpm_confidence:.3f}"]
                             for t in tracks])
    if args.csv:
        write_csv(args.csv, tracks, args.range)
        print(f"Results saved to {args.csv}", file=sys.stderr)
    print(f"Done. Files: {len(files)}, errors: {errors}.", file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
