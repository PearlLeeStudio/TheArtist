"""Bounded live reharmonization timing: three identical requests to POST /api/reharmonize.

No source scores or musical output are saved. Timing includes HTTP.
Use the same public URL and fixture before and after a change.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error('--out must be absolute')
    payload = dict(key='C major', maxChanges=4, modelKey='ft_f1',
                   context=[dict(measure=i, chords=row) for i, row in enumerate([
                       ['Cmaj7', 'Am7'], ['Dm7', 'G7'], ['Em7', 'Am7'], ['Dm7', 'G7'], ['Cmaj7', 'Cmaj7']])],
                   selectedSlots=[dict(measureIndex=m, slotIndex=s) for m in (1, 2) for s in (0, 1)])
    rows = []
    for _ in range(3):
        started = time.perf_counter()
        req = Request(args.url.rstrip('/') + '/api/reharmonize', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=110) as response:
            data = json.load(response)
            if response.status != 200 or len(data.get('candidates', [])) != 3:
                raise AssertionError('Expected HTTP 200 and three candidates')
        row = dict(seconds=round(time.perf_counter()-started, 3),
                   changes=[len(c['changes']) for c in data['candidates']])
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(dict(runs=3, fixture=payload,
                                       median_seconds=statistics.median(r['seconds'] for r in rows),
                                       samples=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
