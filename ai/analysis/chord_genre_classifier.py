"""Chord-only genre classifier for the LoRA paper.

This is an independent diagnostic for "chord distinctiveness": if a classifier
can identify a genre from chord tokens alone, then that genre's label is at
least partly expressed in the chord layer of this corpus.

The classifier intentionally uses only chord symbols from `bars`; it does not
see title, source, key, time signature, or the explicit genre token.

Outputs:
  ai/results/chord_genre_classifier.csv
  ai/results/chord_genre_classifier.md

Usage:
  cd ai
  UV_CACHE_DIR=/tmp/uv-cache uv run python analysis/chord_genre_classifier.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "splits"
RESULTS_DIR = PROJECT_DIR / "results"
SELECTED_TEST_CSV = RESULTS_DIR / "lora_selected_test.csv"

GENRES = [
    "blues", "bossa", "classical", "country", "electronic",
    "folk", "funk", "gospel", "hip_hop", "rnb_soul", "rock",
]


def _song_to_chord_text(song: dict) -> str:
    chords: list[str] = []
    for bar in song.get("bars", []):
        chords.extend(str(chord) for chord in bar if chord)
    return " ".join(chords)


def _load_split(split: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    wanted_sources = {f"chordonomicon_{g}": g for g in GENRES}
    with open(DATA_DIR / f"{split}.jsonl") as f:
        for line in f:
            song = json.loads(line)
            source = song.get("source")
            if source not in wanted_sources:
                continue
            text = _song_to_chord_text(song)
            if not text:
                continue
            texts.append(text)
            labels.append(wanted_sources[source])
    return texts, labels


def _load_lora_gains() -> dict[str, float]:
    gains: dict[str, float] = {}
    if not SELECTED_TEST_CSV.exists():
        return gains
    with open(SELECTED_TEST_CSV) as f:
        for row in csv.DictReader(f):
            gains[row["genre"]] = float(row["delta_top1"])
    return gains


def _safe_corr(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return math.nan, math.nan, math.nan, math.nan
    pr = pearsonr(xs, ys)
    sr = spearmanr(xs, ys)
    return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)


def _write_outputs(
    y_test: list[str],
    y_pred: np.ndarray,
    report: dict,
    matrix: np.ndarray,
    summary: dict,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gains = _load_lora_gains()

    out_csv = RESULTS_DIR / "chord_genre_classifier.csv"
    rows: list[dict] = []
    for genre in GENRES:
        metrics = report[genre]
        row = {
            "genre": genre,
            "test_support": int(metrics["support"]),
            "precision": round(float(metrics["precision"]), 4),
            "recall": round(float(metrics["recall"]), 4),
            "f1": round(float(metrics["f1-score"]), 4),
            "lora_test_gain": gains.get(genre, ""),
        }
        rows.append(row)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["genre", "test_support", "precision", "recall", "f1", "lora_test_gain"],
        )
        writer.writeheader()
        writer.writerows(rows)

    recall_with_gain = [
        (float(r["recall"]), float(r["lora_test_gain"]))
        for r in rows
        if r["lora_test_gain"] != ""
    ]
    non_chorale = [
        (float(r["recall"]), float(r["lora_test_gain"]))
        for r in rows
        if r["lora_test_gain"] != "" and r["genre"] != "classical"
    ]
    pearson_all, pearson_all_p, spearman_all, spearman_all_p = _safe_corr(
        [x for x, _ in recall_with_gain], [y for _, y in recall_with_gain]
    )
    pearson_non, pearson_non_p, spearman_non, spearman_non_p = _safe_corr(
        [x for x, _ in non_chorale], [y for _, y in non_chorale]
    )

    pred_counts = Counter(y_pred)
    truth_counts = Counter(y_test)
    out_md = RESULTS_DIR / "chord_genre_classifier.md"
    lines = [
        "# Chord-only genre classifier",
        "",
        "Diagnostic for chord-distinctiveness. The classifier uses chord tokens only: no key, time signature, title, source, or explicit genre token.",
        "",
        "Model: `CountVectorizer(ngram_range=(1,2))` + `MultinomialNB(alpha=1.0, fit_prior=True)`.",
        "",
        "## Overall",
        "",
        f"- Test accuracy: {summary['accuracy']:.3f}",
        f"- Balanced accuracy: {summary['balanced_accuracy']:.3f}",
        f"- Macro F1: {summary['macro_f1']:.3f}",
        f"- Weighted F1: {summary['weighted_f1']:.3f}",
        f"- Chance balanced accuracy for 11 classes: {1 / len(GENRES):.3f}",
        "",
        "## Per-class metrics",
        "",
        "| Genre | Test support | Precision | Recall | F1 | LoRA test gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        gain = r["lora_test_gain"]
        gain_s = f"{gain:+.2f}" if gain != "" else "--"
        lines.append(
            f"| {r['genre']} | {r['test_support']} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {gain_s} |"
        )
    lines += [
        "",
        "## Correlation with selected-rank LoRA test gain",
        "",
        f"- Pearson(recall, gain), all genres: r={pearson_all:.3f}, p={pearson_all_p:.3g}",
        f"- Spearman(recall, gain), all genres: rho={spearman_all:.3f}, p={spearman_all_p:.3g}",
        f"- Pearson(recall, gain), excluding Bach chorale: r={pearson_non:.3f}, p={pearson_non_p:.3g}",
        f"- Spearman(recall, gain), excluding Bach chorale: rho={spearman_non:.3f}, p={spearman_non_p:.3g}",
        "",
        "## Confusion matrix",
        "",
        "Rows are true genres; columns are predicted genres.",
        "",
        "| true \\ pred | " + " | ".join(GENRES) + " |",
        "|---|" + "---:|" * len(GENRES),
    ]
    for genre, row in zip(GENRES, matrix):
        lines.append(f"| {genre} | " + " | ".join(str(int(v)) for v in row) + " |")
    lines += [
        "",
        "## Prediction distribution",
        "",
        "| Genre | True n | Predicted n |",
        "|---|---:|---:|",
    ]
    for genre in GENRES:
        lines.append(f"| {genre} | {truth_counts[genre]} | {pred_counts[genre]} |")
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


def main() -> None:
    train_texts, y_train = _load_split("train")
    test_texts, y_test = _load_split("test")
    print(f"Train songs: {len(train_texts):,}")
    print(f"Test songs:  {len(test_texts):,}")

    clf = Pipeline([
        (
            "vectorizer",
            CountVectorizer(
                lowercase=False,
                token_pattern=r"[^ ]+",
                ngram_range=(1, 2),
                min_df=2,
            ),
        ),
        ("nb", MultinomialNB(alpha=1.0, fit_prior=True)),
    ])
    clf.fit(train_texts, y_train)
    y_pred = clf.predict(test_texts)

    report = classification_report(
        y_test,
        y_pred,
        labels=GENRES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=GENRES)
    summary = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, labels=GENRES, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, labels=GENRES, average="weighted", zero_division=0),
    }
    print(summary)
    _write_outputs(y_test, y_pred, report, matrix, summary)


if __name__ == "__main__":
    main()
