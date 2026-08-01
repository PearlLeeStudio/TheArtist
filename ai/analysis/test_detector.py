"""Sanity tests for the modal interchange detector."""
from analysis.modal_baseline import classify_chord, parse_key

def check(label, chord, key, expected):
    key_root, mode = parse_key(key)
    cls, deg, qc = classify_chord(chord, key_root, mode)
    ok = "OK " if cls == expected else "FAIL"
    print(f"  {ok}  {label:<40} {chord:>8} in {key:<10} → {cls:<22} deg={deg} qc={qc}")
    return cls == expected

CASES = [
    # --- DIATONIC in C major ---
    ("I",         "Cmaj",  "C major", "DIATONIC"),
    ("ii",        "Dm",    "C major", "DIATONIC"),
    ("V7",        "G7",    "C major", "DIATONIC"),
    ("vi",        "Am",    "C major", "DIATONIC"),
    ("IV",        "Fmaj",  "C major", "DIATONIC"),
    ("vii°",      "Bdim",  "C major", "DIATONIC"),
    # --- MODAL INTERCHANGE in C major ---
    ("bVII (Mixolydian)", "Bb",   "C major", "MODAL_INTERCHANGE"),
    ("bIII",              "Eb",   "C major", "MODAL_INTERCHANGE"),
    ("bVI",               "Ab",   "C major", "MODAL_INTERCHANGE"),
    ("iv (parallel min)", "Fm",   "C major", "MODAL_INTERCHANGE"),
    ("bII (Phrygian)",    "Db",   "C major", "MODAL_INTERCHANGE"),
    # --- SECONDARY DOMINANT in C major ---
    ("V7/V (D7 → G7)",    "D7",   "C major", "SECONDARY_DOMINANT"),
    ("V7/ii (A7 → Dm)",   "A7",   "C major", "SECONDARY_DOMINANT"),
    ("V7/IV (C7 → F)",    "C7",   "C major", "SECONDARY_DOMINANT"),
    # --- TRITONE SUB in C major ---
    ("bII7 → I (Db7)",    "Db7",  "C major", "TRITONE_SUB"),
    # --- DIATONIC in A minor ---
    ("i",                 "Am",     "A minor", "DIATONIC"),
    ("V7 (harmonic)",     "E7",     "A minor", "DIATONIC"),
    ("iv",                "Dm",     "A minor", "DIATONIC"),
    ("bVII",              "Gmaj",   "A minor", "DIATONIC"),
    ("ii°",               "Bm7b5",  "A minor", "DIATONIC"),
    # --- MODAL INTERCHANGE in A minor (borrowing from major) ---
    ("Picardy I (Amaj)",  "Amaj",   "A minor", "MODAL_INTERCHANGE"),
    ("IV (Dorian)",       "Dmaj",   "A minor", "MODAL_INTERCHANGE"),
]

if __name__ == "__main__":
    import sys
    print("\nDetector sanity tests:\n")
    n_pass = sum(check(*c) for c in CASES)
    print(f"\n  {n_pass}/{len(CASES)} passed")
    sys.exit(0 if n_pass == len(CASES) else 1)
