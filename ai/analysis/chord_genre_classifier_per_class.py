"""Per-class precision / recall breakdown for the chord-only genre classifier.

Thin wrapper that re-runs the existing chord_genre_classifier pipeline
and writes the per-class classification_report to a separate CSV. The
aggregate numbers (balanced acc, macro-F1) are already in
chord_genre_classifier.csv; this script surfaces the per-genre breakdown
that paper §6.3 references for boundary mapping.

Output: ai/results/chord_genre_classifier_per_class.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

# Import from the existing module
import sys
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "analysis"))

from chord_genre_classifier import (  # noqa: E402
    _load_split,
    GENRES,
    CountVectorizer,
    MultinomialNB,
    Pipeline,
    classification_report,
    confusion_matrix,
)

RESULTS_DIR = PROJECT_DIR / "results"


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

    rows = []
    for genre in GENRES:
        r = report.get(genre, {})
        rows.append({
            "genre": genre,
            "precision": round(r.get("precision", 0.0), 4),
            "recall": round(r.get("recall", 0.0), 4),
            "f1": round(r.get("f1-score", 0.0), 4),
            "support": int(r.get("support", 0)),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "chord_genre_classifier_per_class.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")

    # Markdown summary
    out_md = RESULTS_DIR / "chord_genre_classifier_per_class.md"
    with open(out_md, "w") as f:
        f.write("# Chord-only classifier — per-class precision / recall\n\n")
        f.write("MultinomialNB on chord-token CountVectorizer features (n-gram 1-2).\n")
        f.write("Per-genre breakdown reveals which genres the chord layer carries information for.\n\n")
        f.write("| Genre | Precision | Recall | F1 | Support |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['genre']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['support']} |\n")
        f.write("\n**Interpretation:**\n")
        f.write("- High precision + high recall = chord-distinctive genre (chord layer carries signal).\n")
        f.write("- Low precision + low recall = chord-indistinct genre (chord layer alone insufficient).\n")
        f.write("- Bach chorale (classical) typically dominates; non-chorale genres cluster low.\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
