"""Hosted catalog smoke: every catalog model is available and generates an 8-bar section.

Sends only the fixed requests below and saves timing metadata, not generated chords.
This checks execution/provenance, not musical quality or comparative latency.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error('--out must be absolute')
    rows = []

    def call(route, payload=None):
        request = Request(args.url.rstrip('/') + route,
                          data=json.dumps(payload).encode() if payload is not None else None,
                          headers={'Content-Type': 'application/json'})
        started = time.perf_counter()
        with urlopen(request, timeout=200) as response:
            data = json.load(response)
        return data, round(time.perf_counter() - started, 3)

    def record(row):
        rows.append(row)
        args.out.write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    catalog, _ = call('/api/models/list')
    assert len(catalog['models']) == 18 and all(m['available'] for m in catalog['models'])
    for model in catalog['models']:
        key = model['key']
        genre = key.removeprefix('ft_f1_lora_') if key.startswith('ft_f1_lora_') else 'pop'
        data, seconds = call('/api/generate/section', dict(
            key='C major', genre=genre, sectionType='verse', chordCount=16, modelKey=key, seed=17))
        assert data['source'] == 'ai' and data['modelUsed'] == key, data
        assert len(data['suggestions']) == 3 and all(len(s['chords']) == 8 for s in data['suggestions'])
        record(dict(case=key, seconds=seconds, candidates=3, bars=8))


if __name__ == '__main__':
    main()
