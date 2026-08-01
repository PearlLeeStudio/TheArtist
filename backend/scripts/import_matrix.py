"""Controlled original PDF reader matrix and first/repeated HTTP-handler timings.

Twelve synthetic engravings, not twelve independent real musical scores.
No private score, model call, download or work date is included.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
from app.main import create_app
from tests.vector_score_fixture import build_vector_score_pdf
from scripts.experience_benchmark import percentiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.is_absolute() or args.out.exists():
        parser.error("Use a new absolute output path")
    client = TestClient(create_app({"THEARTIST_PUBLIC_HOST": "1"}))
    cases = []
    for tonic, clef, expected, key in (("C4", "treble", "C", "C major"), ("F2", "bass", "F", "F major")):
        for meter in ((4, 4), (2, 4)):
            for scale, dx, dy in ((.65, 14, 23), (1, 0, 0), (1.7, 33, 61)):
                raw = build_vector_score_pdf(tonic=tonic, clef=clef, scale=scale, offset_x=dx, offset_y=dy,
                                             time_signature=meter)
                body = dict(filename="original.pdf", mimeType="application/pdf", dataBase64=base64.b64encode(raw).decode(),
                            mode="infer_harmony")
                timings = {"first": [], "repeat": []}
                readings = []
                for trial in range(5):
                    for condition in ("first", "repeat"):
                        start = time.perf_counter()
                        response = client.post("/api/import/score", json=body)
                        timings[condition].append(round((time.perf_counter() - start) * 1000, 3))
                        if response.status_code != 200:
                            raise RuntimeError(f"Local import failed: HTTP {response.status_code}")
                        result = response.json()
                        readings.append(dict(chords=[bar["chords"] for bar in result["measures"]], key=result["key"],
                                             meter=result["timeSignature"]))
                correct = all(reading == dict(chords=[[expected]], key=key, meter=list(meter)) for reading in readings)
                cases.append(dict(tonic=tonic, clef=clef, meter=meter, scale=scale, offset=[dx, dy],
                                  fixtureSha256=hashlib.sha256(raw).hexdigest(), exact=correct,
                                  expected=dict(chords=[[expected]], key=key, meter=meter), actual=readings[0],
                                  durationMs=timings, summary={c: percentiles(v) for c, v in timings.items()}))
    report = dict(method="Original vector glyphs; two triads, two meters, three transforms; five first/repeated pairs per case; no score cache",
                  timingScope="In-process HTTP handler including validation, parsing, analysis and serialization; excludes network/browser",
                  cases=cases, exactCases=sum(case["exact"] for case in cases), totalCases=len(cases),
                  limits=["Synthetic layout coverage, not real-world score accuracy", "Vector PDF input only", "No human correction-time measurement"])
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("exactCases", "totalCases", "timingScope")}))
    if report["exactCases"] != report["totalCases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
