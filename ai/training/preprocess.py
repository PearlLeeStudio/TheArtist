"""Preprocess raw datasets into unified chord sequence format.

Reads five datasets and writes data/processed/songs.jsonl with unified format:
    {"title": "...", "source": "...", "key": "Cmaj", "time_signature": "4/4",
     "genre": "jazz", "bars": [["Dm7", "G7"], ["Cmaj7"], ...]}

Usage:
    uv run python training/preprocess.py
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

from tokenizer import ChordTokenizer

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

_tok = ChordTokenizer()


# ======================================================================
# Chordonomicon  (Harte-style, ~680K songs)
# ======================================================================

_SHARP_RE = re.compile(r"^([A-G])s(?!us)")  # Cs→C#, Fs→F# but NOT sus


def _chordonomicon_chord(raw: str) -> str | None:
    """Normalize a Chordonomicon chord token to our vocab."""
    raw = re.sub(r"no3d$", "", raw)          # power chords → root
    raw = _SHARP_RE.sub(r"\1#", raw)         # Cs→C#, Fs→F# (not sus)
    return _tok.normalize_chord(raw)


_GENRE_MAP = {
    "pop": "pop", "pop rock": "pop", "country": "pop",
    "rock": "rock", "alternative": "rock", "punk": "rock", "metal": "rock",
    "soul": "pop", "rap": "pop", "reggae": "pop",
    "jazz": "jazz", "electronic": "pop",
}


def parse_chordonomicon() -> list[dict]:
    """Parse Chordonomicon CSV (HuggingFace ailsntua/Chordonomicon)."""
    csv_path = RAW_DIR / "chordonomicon" / "chordonomicon.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return []

    songs: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chords_str = row.get("chords", "")
            if not chords_str:
                continue

            # Extract chord tokens (skip section tags like <verse_1>)
            raw_tokens = [t for t in chords_str.split() if not t.startswith("<")]
            if len(raw_tokens) < 4:
                continue

            # Normalize chords
            chords = []
            for t in raw_tokens:
                norm = _chordonomicon_chord(t)
                if norm:
                    chords.append(norm)
            if len(chords) < 4:
                continue

            # Group into bars (2 chords per bar for 4/4, common in pop)
            bars: list[list[str]] = []
            for i in range(0, len(chords), 2):
                bars.append(chords[i : i + 2])

            genre = _GENRE_MAP.get(row.get("main_genre", ""), "pop")

            songs.append({
                "title": row.get("spotify_song_id", ""),
                "source": "chordonomicon",
                "key": "Cmaj",        # no key info — inferred during augmentation
                "time_signature": "4/4",
                "genre": genre,
                "bars": bars,
            })

    return songs


# ======================================================================
# Billboard
# ======================================================================

def parse_billboard() -> list[dict]:
    """Parse McGill Billboard salami_chords annotations."""
    salami_dir = RAW_DIR / "billboard" / "billboard-2.0-salami_chords"
    if not salami_dir.exists():
        print(f"  [skip] {salami_dir} not found")
        return []

    songs: list[dict] = []
    for song_dir in sorted(salami_dir.iterdir()):
        if not song_dir.is_dir():
            continue
        chord_file = song_dir / "salami_chords.txt"
        if not chord_file.exists():
            continue
        try:
            song = _parse_billboard_file(chord_file)
            if song and len(song["bars"]) >= 4:
                songs.append(song)
        except Exception as e:
            print(f"  [warn] {chord_file.name}: {e}")
    return songs


def _parse_billboard_file(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    tonic = metre = title = ""

    chord_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# tonic:"):
            tonic = line.split(":", 1)[1].strip()
        elif line.startswith("# metre:"):
            metre = line.split(":", 1)[1].strip()
        elif line.startswith("# title:"):
            title = line.split(":", 1)[1].strip()
        elif not line.startswith("#") and "|" in line:
            chord_lines.append(line)

    if not tonic or not chord_lines:
        return None

    root = _tok.normalize_root(tonic)
    if root is None:
        return None

    ts = metre.strip() if metre.strip() in ("4/4", "3/4", "6/8", "2/4", "5/4") else "4/4"

    bars: list[list[str]] = []
    for line in chord_lines:
        # Find bar-delimited section (may be preceded by timestamp + section label)
        pipe_idx = line.find("|")
        if pipe_idx < 0:
            continue
        bar_section = line[pipe_idx:]
        # Remove parenthetical annotations like (voice)
        bar_section = re.sub(r"\(.*?\)", "", bar_section)

        for bar_str in bar_section.split("|"):
            bar_str = bar_str.strip()
            if not bar_str or bar_str in ("silence", "end"):
                continue
            chords: list[str] = []
            for tok_str in bar_str.split():
                tok_str = tok_str.strip().strip(",.")
                if not tok_str or tok_str in ("silence", "end", "&pause", "x", "X", "."):
                    continue
                norm = _normalize_billboard_chord(tok_str)
                if norm:
                    chords.append(norm)
            if chords:
                bars.append(chords)

    if not bars:
        return None
    return {
        "title": title,
        "source": "billboard",
        "key": f"{root}maj",
        "time_signature": ts,
        "genre": "pop",
        "bars": bars,
    }


def _normalize_billboard_chord(s: str) -> str | None:
    """Billboard ``Root:Quality[/bass]`` → our format."""
    if "/" in s and ":" not in s:
        s = s.split("/")[0]

    if ":" not in s:
        root = _tok.normalize_root(s)
        return f"{root}maj" if root else None

    root_part, qual_part = s.split(":", 1)
    if "/" in qual_part:
        qual_part = qual_part.split("/")[0]

    root = _tok.normalize_root(root_part)
    if root is None:
        return None

    _bb = {
        "maj": "maj", "min": "m", "dim": "dim", "aug": "aug",
        "7": "7", "maj7": "maj7", "min7": "m7", "maj6": "6", "min6": "m6",
        "9": "9", "min9": "m9", "maj9": "maj9",
        "11": "11", "min11": "m11", "13": "13",
        "sus4": "sus4", "sus2": "sus2",
        "hdim7": "m7b5", "dim7": "dim7",
        "1": "maj", "5": "maj", "(1)": "maj",
        "(1,5)": "maj", "sus4(b7)": "7", "sus4(b7,9)": "9",
    }
    quality = _bb.get(qual_part) or _tok._normalize_quality(qual_part) or "maj"

    token = f"{root}{quality}"
    return token if token in _tok.token2id else None


# ======================================================================
# Jazz Harmony Treebank
# ======================================================================

def parse_jazz_treebank() -> list[dict]:
    """Parse Jazz Harmony Treebank treebank.json."""
    jht_file = RAW_DIR / "jazz-harmony-treebank" / "treebank.json"
    if not jht_file.exists():
        print(f"  [skip] {jht_file} not found")
        return []

    data = json.loads(jht_file.read_text())
    songs: list[dict] = []

    for entry in data:
        try:
            song = _parse_jht_entry(entry)
            if song and len(song["bars"]) >= 4:
                songs.append(song)
        except Exception as e:
            print(f"  [warn] JHT '{entry.get('title', '?')}': {e}")
    return songs


def _parse_jht_entry(entry: dict) -> dict | None:
    measures = entry.get("measures", [])
    chords = entry.get("chords", [])
    if not chords or not measures:
        return None

    key_str = entry.get("key", "C")
    root = _tok.normalize_root(key_str) or "C"

    meter = entry.get("meter", {})
    num, den = meter.get("numerator", 4), meter.get("denominator", 4)
    ts = f"{num}/{den}"
    if ts not in ("4/4", "3/4", "6/8", "2/4", "5/4"):
        ts = "4/4"

    bars_dict: dict[int, list[str]] = {}
    for meas_num, chord_str in zip(measures, chords):
        norm = _tok.normalize_chord(chord_str)
        if norm:
            bars_dict.setdefault(meas_num, []).append(norm)

    bars = [bars_dict[k] for k in sorted(bars_dict)]
    if not bars:
        return None

    return {
        "title": entry.get("title", ""),
        "source": "jht",
        "key": f"{root}maj",
        "time_signature": ts,
        "genre": "jazz",
        "bars": bars,
    }


# ======================================================================
# Weimar Jazz Database
# ======================================================================

def parse_wjazzd() -> list[dict]:
    """Parse Weimar Jazz Database (SQLite)."""
    db_path = RAW_DIR / "wjazzd" / "wjazzd.db"
    if not db_path.exists():
        print(f"  [skip] {db_path} not found")
        return []

    # Verify it's a real SQLite file
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT 1 FROM composition_info LIMIT 1")
    except (sqlite3.DatabaseError, sqlite3.OperationalError):
        print(f"  [skip] {db_path} is not a valid database")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    songs: list[dict] = []

    try:
        # Composition metadata
        comp_info: dict[int, dict] = {}
        for row in conn.execute("SELECT * FROM composition_info"):
            comp_info[row["compid"]] = dict(row)

        # Track → composition mapping
        track_comp: dict[int, int] = {}
        for row in conn.execute("SELECT trackid, compid FROM track_info"):
            track_comp[row["trackid"]] = row["compid"]

        # Melody → track mapping
        mel_track: dict[int, int] = {}
        for row in conn.execute("SELECT melid, trackid FROM solo_info"):
            mel_track[row["melid"]] = row["trackid"]

        # Unique melody IDs with chord data
        mel_ids = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT melid FROM beats "
                "WHERE chord IS NOT NULL AND chord != '' AND bar >= 0"
            ).fetchall()
        ]

        seen_comps: set[int] = set()
        for melid in mel_ids:
            trackid = mel_track.get(melid)
            compid = track_comp.get(trackid) if trackid else None
            if compid and compid in seen_comps:
                continue
            if compid:
                seen_comps.add(compid)

            beats = conn.execute(
                "SELECT bar, beat, chord, signature FROM beats "
                "WHERE melid = ? AND bar >= 0 ORDER BY bar, beat",
                (melid,),
            ).fetchall()
            if not beats:
                continue

            bars_dict: dict[int, list[str]] = {}
            ts = "4/4"
            for b in beats:
                if b["signature"]:
                    ts = b["signature"]
                chord_str = b["chord"]
                if not chord_str:
                    continue
                norm = _tok.normalize_chord(chord_str)
                if norm:
                    cur = bars_dict.setdefault(b["bar"], [])
                    if not cur or cur[-1] != norm:
                        cur.append(norm)

            bars = [bars_dict[k] for k in sorted(bars_dict)]
            if len(bars) < 4:
                continue

            if ts not in ("4/4", "3/4", "6/8", "2/4", "5/4"):
                ts = "4/4"

            comp = comp_info.get(compid, {})
            songs.append({
                "title": comp.get("title", f"wjazzd_{melid}"),
                "source": "wjazzd",
                "key": "Cmaj",  # WJazzD lacks consistent key info
                "time_signature": ts,
                "genre": "jazz",
                "bars": bars,
            })
    finally:
        conn.close()

    return songs


# ======================================================================
# JAAH  (Jazz Audio-Aligned Harmony, ~113 songs)
# ======================================================================

# Map JAAH interval notation → TheArtist quality
_JAAH_INTERVAL_MAP: dict[tuple[str, ...], str] = {
    # dom7 variants
    ("3", "5", "b7"): "7",
    ("3", "5", "b7", "b9"): "7",
    ("3", "5", "b7", "9"): "9",
    ("3", "5", "b7", "#9"): "7",
    ("3", "5", "b7", "#11"): "7",
    ("3", "5", "b7", "b13"): "7",
    ("3", "5", "b7", "b9", "13"): "7",
    ("3", "5", "b7", "#9", "#11", "13"): "7",
    ("3", "5", "b7", "11"): "7",
    # aug dom
    ("3", "#5", "b7"): "7",
    ("3", "#5", "b7", "9"): "7",
    ("3", "#5", "b7", "b9"): "7",
    ("3", "#5", "9"): "7",
    # minor variants
    ("b3", "5", "b7"): "m7",
    ("b3", "5", "b7", "11"): "m7",
}


def _jaah_quality(quality_str: str) -> str | None:
    """Convert JAAH quality notation to TheArtist quality."""
    # Standard qualities
    q_map = {
        "maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7",
        "dim": "dim", "dim7": "dim7", "hdim7": "m7b5", "aug": "aug",
        "9": "9", "maj9": "maj9", "min9": "m9",
        "maj6": "6", "min6": "m6",
        "sus4": "sus4", "sus2": "sus2",
    }
    if quality_str in q_map:
        return q_map[quality_str]

    # Interval notation: (3,5,b7,...)
    if quality_str.startswith("(") and quality_str.endswith(")"):
        intervals = tuple(quality_str[1:-1].split(","))
        return _JAAH_INTERVAL_MAP.get(intervals, "7")  # default dom7 for unknown

    return None


def _parse_jaah_chord(token: str) -> str | None:
    """Parse a single JAAH chord token into TheArtist format."""
    if not token or token == "N":
        return None

    # Handle slash chords: A:9/3 → just A:9
    if "/" in token:
        token = token.split("/")[0]

    if ":" not in token:
        # Just a root (e.g., "A") → major triad
        norm = _tok.normalize_chord(token)
        return norm

    root, quality = token.split(":", 1)
    root_norm = _tok.normalize_root(root)
    if not root_norm:
        return None

    q = _jaah_quality(quality)
    if q is None:
        # Try direct normalization
        norm = _tok.normalize_chord(token.replace(":", ""))
        return norm

    chord = f"{root_norm}{q}" if q else root_norm
    norm = _tok.normalize_chord(chord)
    return norm


def parse_jaah() -> list[dict]:
    """Parse JAAH JSON annotations."""
    ann_dir = RAW_DIR / "jaah" / "annotations"
    if not ann_dir.exists():
        print(f"  [skip] {ann_dir} not found")
        return []

    songs: list[dict] = []
    for fpath in sorted(ann_dir.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)

        bars: list[list[str]] = []
        for part in data.get("parts", []):
            for chord_line in part.get("chords", []):
                # Split by bar lines
                for bar_str in chord_line.split("|"):
                    bar_str = bar_str.strip()
                    if not bar_str:
                        continue
                    bar_chords: list[str] = []
                    for token in bar_str.split():
                        token = token.strip()
                        chord = _parse_jaah_chord(token)
                        if chord and (not bar_chords or bar_chords[-1] != chord):
                            bar_chords.append(chord)
                    if bar_chords:
                        bars.append(bar_chords)

        if len(bars) < 4:
            continue

        # Parse key
        key_str = data.get("key", "")
        if "major" in key_str.lower():
            root = key_str.split()[0]
            root_n = _tok.normalize_root(root)
            key = f"{root_n}maj" if root_n else "Cmaj"
        elif "minor" in key_str.lower():
            root = key_str.split()[0]
            root_n = _tok.normalize_root(root)
            key = f"{root_n}min" if root_n else "Amin"
        else:
            key = "Cmaj"

        ts = data.get("metre", "4/4")
        if ts not in ("4/4", "3/4", "6/8", "2/4", "5/4"):
            ts = "4/4"

        songs.append({
            "title": data.get("title", fpath.stem),
            "source": "jaah",
            "key": key,
            "time_signature": ts,
            "genre": "jazz",
            "bars": bars,
        })

    return songs


# ======================================================================
# JazzStandards  (iReal Pro DB, ~1,382 songs)
# ======================================================================

def parse_jazzstandards(existing_titles: set[str] | None = None) -> list[dict]:
    """Parse JazzStandards JSON. Dedup against existing_titles (e.g. JHT)."""
    json_path = RAW_DIR / "JazzStandards" / "JazzStandards.json"
    if not json_path.exists():
        print(f"  [skip] {json_path} not found")
        return []

    with open(json_path) as f:
        data = json.load(f)

    songs: list[dict] = []
    skipped_dup = 0

    for entry in data:
        title = entry.get("Title", "")

        # Dedup against JHT (both are iReal Pro based)
        if existing_titles and title.lower().strip() in existing_titles:
            skipped_dup += 1
            continue

        bars: list[list[str]] = []
        for section in entry.get("Sections", []):
            # Main segment
            chords_str = section.get("MainSegment", {}).get("Chords", "")
            for bar_str in chords_str.split("|"):
                bar_chords: list[str] = []
                for ch in bar_str.split(","):
                    ch = ch.strip()
                    if not ch:
                        continue
                    # Strip optional-chord parentheses: (Am7) → Am7
                    if ch.startswith("(") and ch.endswith(")"):
                        ch = ch[1:-1]
                    norm = _tok.normalize_chord(ch)
                    if norm and (not bar_chords or bar_chords[-1] != norm):
                        bar_chords.append(norm)
                if bar_chords:
                    bars.append(bar_chords)

            # Endings (use first ending only for simplicity)
            endings = section.get("Endings", [])
            if endings:
                for bar_str in endings[0].get("Chords", "").split("|"):
                    bar_chords = []
                    for ch in bar_str.split(","):
                        ch = ch.strip()
                        if not ch:
                            continue
                        if ch.startswith("(") and ch.endswith(")"):
                            ch = ch[1:-1]
                        norm = _tok.normalize_chord(ch)
                        if norm and (not bar_chords or bar_chords[-1] != norm):
                            bar_chords.append(norm)
                    if bar_chords:
                        bars.append(bar_chords)

        if len(bars) < 4:
            continue

        ts = entry.get("TimeSignature", "4/4")
        if ts not in ("4/4", "3/4", "6/8", "2/4", "5/4"):
            ts = "4/4"

        songs.append({
            "title": title,
            "source": "jazzstandards",
            "key": "Cmaj",  # iReal Pro doesn't always store key reliably
            "time_signature": ts,
            "genre": "jazz",
            "bars": bars,
        })

    if skipped_dup:
        print(f"  Dedup: skipped {skipped_dup} songs already in JHT")

    return songs


# ======================================================================
# ChoCo  (JAMS-format meta-corpus by de Berardinis et al. 2023)
# ======================================================================

def _choco_key(key_ann: dict | None) -> str:
    if not key_ann or not key_ann.get("data"):
        return "Cmaj"
    v = key_ann["data"][0].get("value")
    if isinstance(v, dict):
        root = v.get("tonic", "C")
        mode = v.get("mode", "major")
        return f"{root}{'maj' if 'maj' in str(mode).lower() else 'min'}"
    if isinstance(v, str) and ":" in v:
        root, mode = v.split(":", 1)
        return f"{root}{'maj' if 'maj' in mode.lower() else 'min'}"
    return "Cmaj"


def _choco_timesig(ts_ann: dict | None) -> str:
    if not ts_ann or not ts_ann.get("data"):
        return "4/4"
    v = ts_ann["data"][0].get("value")
    if isinstance(v, dict):
        return f"{v.get('numerator', 4)}/{v.get('denominator', 4)}"
    if isinstance(v, str):
        return v
    return "4/4"


def parse_choco(partition: str, existing_titles: set[str] | None = None) -> list[dict]:
    """Parse ChoCo JAMS annotations from a named partition."""
    existing_titles = existing_titles or set()
    p_dir = RAW_DIR / "choco" / "partitions" / partition / "choco" / "jams-converted"
    if not p_dir.exists():
        print(f"  [skip] {p_dir} not found")
        return []

    songs: list[dict] = []
    skipped_dup = 0
    for jams_path in sorted(p_dir.glob("*.jams")):
        try:
            with open(jams_path) as f:
                j = json.load(f)
        except Exception:
            continue
        title = (j.get("file_metadata", {}).get("title") or jams_path.stem).strip()
        if not title:
            continue
        tlow = title.lower()
        if tlow in existing_titles:
            skipped_dup += 1
            continue

        chord_ann = key_ann = ts_ann = None
        for a in j.get("annotations", []):
            ns = a.get("namespace")
            if ns == "chord_harte" and chord_ann is None:
                chord_ann = a
            elif ns == "key_mode" and key_ann is None:
                key_ann = a
            elif ns == "timesig" and ts_ann is None:
                ts_ann = a
        if chord_ann is None or not chord_ann.get("data"):
            continue

        key = _choco_key(key_ann)
        ts = _choco_timesig(ts_ann)

        bars_dict: dict[int, list[str]] = {}
        for d in chord_ann["data"]:
            t = d.get("time", 0)
            try:
                bar_num = int(str(t).split(".")[0])
            except (ValueError, TypeError):
                continue
            ch = _tok.normalize_chord(str(d.get("value", "")))
            if ch:
                bars_dict.setdefault(bar_num, []).append(ch)

        bars = [bars_dict[k] for k in sorted(bars_dict) if bars_dict[k]]
        if len(bars) < 4:
            continue

        songs.append({
            "title": title,
            "source": f"choco:{partition}",
            "key": key,
            "time_signature": ts,
            "genre": "jazz",
            "bars": bars,
        })
        existing_titles.add(tlow)

    print(f"  choco:{partition}: {len(songs)} songs (skipped {skipped_dup} dup)")
    return songs


# ======================================================================
# Bunks Jazz-Chord-Progressions-Corpus  (Impro-Visor lineage)
# ======================================================================

def parse_bunks(existing_titles: set[str] | None = None) -> list[dict]:
    """Parse the ASCII ``| bar | bar |`` corpus from carey-bunks repo."""
    existing_titles = existing_titles or set()
    root = RAW_DIR / "bunks" / "SongDB"
    if not root.exists():
        print("  [skip] bunks not found")
        return []

    songs: list[dict] = []
    skipped_dup = 0
    for letter_dir in sorted(root.iterdir()):
        if not letter_dir.is_dir():
            continue
        for song_file in sorted(letter_dir.glob("*.txt")):
            try:
                text = song_file.read_text(errors="ignore")
            except Exception:
                continue
            title = song_file.stem
            key = "Cmaj"
            ts = "4/4"
            bars: list[list[str]] = []
            for raw_line in text.splitlines():
                s = raw_line.strip()
                if not s:
                    continue
                if s.startswith("Title"):
                    if "=" in s:
                        title = s.split("=", 1)[1].strip()
                elif s.startswith("DBKeySig"):
                    if "=" in s:
                        k = s.split("=", 1)[1].strip()
                        key = k[:-1] + "min" if k.endswith("m") else k + "maj"
                elif s.startswith("TimeSig"):
                    if "=" in s:
                        parts = s.split("=", 1)[1].strip().split()
                        if len(parts) == 2:
                            ts = f"{parts[0]}/{parts[1]}"
                elif "|" in s and "=" not in s:
                    for bar_str in s.split("|"):
                        bar_str = bar_str.strip()
                        if not bar_str:
                            continue
                        chords: list[str] = []
                        for t in bar_str.split():
                            n = _tok.normalize_chord(t)
                            if n:
                                chords.append(n)
                        if chords:
                            bars.append(chords)
            tlow = title.lower().strip()
            if tlow in existing_titles:
                skipped_dup += 1
                continue
            if len(bars) < 4:
                continue
            songs.append({
                "title": title,
                "source": "bunks",
                "key": key,
                "time_signature": ts,
                "genre": "jazz",
                "bars": bars,
            })
            existing_titles.add(tlow)

    print(f"  bunks: {len(songs)} songs (skipped {skipped_dup} dup)")
    return songs


# ======================================================================
# Main
# ======================================================================

def preprocess_all() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing Chordonomicon...")
    chordonomicon = parse_chordonomicon()
    print(f"  {len(chordonomicon)} songs")

    print("Parsing Billboard...")
    billboard = parse_billboard()
    print(f"  {len(billboard)} songs")

    print("Parsing Jazz Harmony Treebank...")
    jht = parse_jazz_treebank()
    print(f"  {len(jht)} songs")

    print("Parsing WJazzD...")
    wjazzd = parse_wjazzd()
    print(f"  {len(wjazzd)} songs")

    print("Parsing JAAH...")
    jaah = parse_jaah()
    print(f"  {len(jaah)} songs")

    # JazzStandards — dedup against JHT (both iReal Pro based)
    jht_titles = {s["title"].lower().strip() for s in jht}
    print("Parsing JazzStandards...")
    jazzstandards = parse_jazzstandards(existing_titles=jht_titles)
    print(f"  {len(jazzstandards)} songs")

    # ChoCo Real Book + Jazz Corpus partitions, dedup against everything jazz so far
    existing_jazz_titles = {
        s["title"].lower().strip()
        for s in (jht + wjazzd + jaah + jazzstandards)
    }
    print("Parsing ChoCo real-book partition...")
    choco_rb = parse_choco("real-book", existing_titles=existing_jazz_titles)
    print("Parsing ChoCo jazz-corpus partition...")
    choco_jc = parse_choco("jazz-corpus", existing_titles=existing_jazz_titles)

    # Bunks Jazz-Chord-Progressions-Corpus (Impro-Visor lineage), dedup against all jazz
    print("Parsing Bunks Jazz-Chord-Progressions-Corpus...")
    bunks = parse_bunks(existing_titles=existing_jazz_titles)

    all_songs = (
        chordonomicon + billboard + jht + wjazzd + jaah + jazzstandards
        + choco_rb + choco_jc + bunks
    )
    print(f"\nTotal: {len(all_songs)} songs")

    # Per-source stats
    by_source = {}
    for s in all_songs:
        by_source.setdefault(s["source"], []).append(s)
    for src, songs in by_source.items():
        print(f"  {src}: {len(songs)} songs")

    out_path = PROCESSED_DIR / "songs.jsonl"
    with open(out_path, "w") as f:
        for song in all_songs:
            f.write(json.dumps(song, ensure_ascii=False) + "\n")
    print(f"Written → {out_path}")

    # Stats
    total_bars = sum(len(s["bars"]) for s in all_songs)
    total_chords = sum(sum(len(b) for b in s["bars"]) for s in all_songs)
    print(f"Bars: {total_bars}  |  Chords: {total_chords}")

    # Tokenizer coverage check
    tokenizer = ChordTokenizer()
    ok = fail = 0
    for song in all_songs:
        for bar in song["bars"]:
            for chord in bar:
                if tokenizer.encode_chord(chord) is not None:
                    ok += 1
                else:
                    fail += 1
    total = ok + fail
    if total > 0:
        print(f"Tokenizer coverage: {ok}/{total} ({ok/total*100:.1f}%)")
    else:
        print("Tokenizer coverage: no chords to check")
    print(f"Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    preprocess_all()
