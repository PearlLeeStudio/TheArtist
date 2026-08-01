"""Voyager primary endpoint — `POST /api/generate/song`.

Composite endpoint: voyager sends a genre and a length in bars and may pick
the key, tempo and bass and the editor's Play-panel choices
(`harmony_rhythm` / `drum_pattern` / `harmony_instrument`); every choice it
leaves out is the genre's own, and the time signature always is. The
genre-appropriate checkpoint writes the chord progression; when the request
picked the key, a pass that clearly sits in another key is moved into it
(key_anchor). Harmony + bass + drum are then layered into 3 symbolic tracks (melody
track was dropped 2026-05-10) and a MIDI render ships along.
An optional seed controls sampling without reseeding global RNG state.
This composite endpoint returns one arrangement: n_candidates must be 1.

Response shape (per `SongGenerateResponse`):
    {genre, key, bpm, time_signature,
     harmony_rhythm, drum_pattern,     # effective choices; drum_pattern "off" = no drums
     bars: [{"chords": [...]}, ...],
     tracks: {harmony, bass, drum},   # each: {events, instrument, source}
     midi_b64,
     model_used, seed}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ContextMeasure,
    SongGenerateRequest,
    SongGenerateResponse,
    Track,
    TrackEvent,
)
from app.services.chord_generator import generate_suggestions
from app.services.genre_tables import (
    bass_instrument,
    drum_pattern,
    drum_pattern_by_id,
    genre_defaults,
    genre_drum_pattern_id,
    genre_rhythm_preset,
    harmony_instrument,
    harmony_rhythm_preset,
)
from app.services.inference import (
    TORCH_AVAILABLE,
    ModelRegistry,
    _bars_to_chord_map,
    build_prompt,
    dispatch_model_for_genre,
    generate_with_model,
)
from app.services.key_anchor import anchor_to_key
from app.services.song_builder import (
    build_bass_events,
    build_drum_events,
    build_harmony_events,
    midi_to_b64,
    render_midi,
)

log = logging.getLogger(__name__)

router = APIRouter()


def _anchor_pass(bars: list[list[str]], key: str) -> list[list[str]]:
    """A model pass moved into `key` when it clearly sits in another one (key_anchor, as the
    editor does for a passage with nothing before it). Each written chord counts once: the
    pass is anchored before it is fitted to two slots per bar."""
    root, mode = key.split()
    written = [chord for bar in bars for chord in bar if isinstance(chord, str) and chord.strip()]
    moved = iter(anchor_to_key(written, root, mode == "minor"))
    return [[next(moved) if isinstance(chord, str) and chord.strip() else chord for chord in bar] for bar in bars]


def _model_bars(
    model_key: str,
    *,
    genre: str,
    key: str,
    bpm: int,
    time_signature: tuple[int, int],
    n_bars: int,
    selected: list[int],
    seed: int | None,
    anchor: bool = False,
) -> list[dict] | None:
    """One `n_bars`-bar model pass over the `selected` measures, fitted like
    the editor's (`_bars_to_chord_map`): a short pass repeats its own bars
    and each bar keeps two chords. With `anchor` (a key the request picked)
    the pass is first moved into that key when it clearly sits in another.
    None, for the caller's rule fallback, when the checkpoint does not load,
    the pass fails or it writes no bars.
    """
    registry = ModelRegistry.get()
    tokenizer = registry.tokenizer
    prompt_ids = build_prompt(
        tokenizer, key, genre, time_signature,
        context=[],
        selected_measures=selected,
        bpm=bpm,
    )
    loaded = registry.load_model(model_key)
    if loaded is None:
        return None
    try:
        bars = generate_with_model(
            loaded, tokenizer, prompt_ids, n_bars,
            temperature=0.8,
            seed=seed,
        )
        if bars:
            if anchor:
                bars = _anchor_pass(bars, key)
            chord_map = _bars_to_chord_map(bars, selected)
            return [{"chords": chord_map[str(m)]} for m in selected]
        log.warning("AI inference returned no bars, falling back to rules")
    except Exception:
        log.exception("AI inference failed, falling back to rules")
    return None


def _generate_bars(
    *,
    genre: str,
    key: str,
    bpm: int,
    time_signature: tuple[int, int],
    n_bars: int,
    seed: int | None = None,
    anchor: bool = False,
) -> tuple[list[dict], str]:
    """Run the existing chord-generation chain (AI when available, rules
    otherwise) and return (bars, model_used): `n_bars` bars of two chords.
    Mirrors the path in `routes/chords.py:generate_chords` but with no
    candidate fan-out. Both paths use the request's seed; model_used
    identifies a rule fallback.
    """
    model_key = dispatch_model_for_genre(genre)
    selected = list(range(n_bars))

    if TORCH_AVAILABLE:
        ai_bars = _model_bars(
            model_key, genre=genre, key=key, bpm=bpm,
            time_signature=time_signature, n_bars=n_bars, selected=selected, seed=seed,
            anchor=anchor,
        )
        if ai_bars is not None:
            return ai_bars, model_key

    # Rule fallback
    full_context = [ContextMeasure(measure=m, chords=[None, None]) for m in selected]
    suggestions = generate_suggestions(
        key=key, genre=genre, context=full_context,
        selected_measures=selected, time_signature=time_signature,
        seed=seed,
    )
    if suggestions:
        sug = suggestions[0]
        sorted_keys = sorted(sug.chords.keys(), key=int)
        bars = [sug.chords[k] for k in sorted_keys]
    else:
        bars = [["Cmaj", "Cmaj"]] * n_bars
    return [{"chords": b} for b in bars], "rules"


def _drum_pattern_id(req: SongGenerateRequest) -> int | None:
    """The drum pattern to play: the request's pick, the genre's own when it
    picks none, and None (no drums) for "off"."""
    if req.drum_pattern is None:
        return genre_drum_pattern_id(req.genre)
    if req.drum_pattern == "off":
        return None
    return req.drum_pattern


def _events_to_track(events: list[dict], instrument: str) -> Track:
    return Track(
        events=[TrackEvent(**e) for e in events],
        instrument=instrument,
        source="rule",
    )


@router.post("/generate/song")
def generate_song(req: SongGenerateRequest) -> SongGenerateResponse:
    """Voyager primary path: genre + length, and any picked choices → full multi-track render."""
    defaults = genre_defaults(req.genre)
    key = req.key or defaults["key"]
    bpm = req.bpm or defaults["bpm"]
    time_signature: tuple[int, int] = defaults["time_signature"]
    beats_per_measure = time_signature[0]

    if drum_pattern(req.genre) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported genre '{req.genre}'. "
                "Expected one of: jazz, pop, rock, blues, bossa, classical, "
                "country, rnb_soul, hip_hop, electronic, funk, folk, gospel."
            ),
        )

    # Play-panel choices (ids already validated by the schema); None = the genre's.
    rhythm = harmony_rhythm_preset(req.harmony_rhythm) or genre_rhythm_preset(req.genre)
    drum_id = _drum_pattern_id(req)
    h_instrument = req.harmony_instrument or harmony_instrument(req.genre)
    b_instrument = req.bass_instrument or bass_instrument(req.genre)

    bars, model_used = _generate_bars(
        genre=req.genre,
        key=key,
        bpm=bpm,
        time_signature=time_signature,
        n_bars=req.length_bars,
        seed=req.seed,
        # Only a picked key is anchored: a request that leaves the key to the genre
        # gets the model's pass as written, as before the key field existed.
        anchor=req.key is not None,
    )

    h_events = build_harmony_events(bars, req.genre, beats_per_measure=beats_per_measure,
                                    pattern=rhythm["pattern"])
    b_events = build_bass_events(bars, req.genre, beats_per_measure=beats_per_measure)
    d_events = build_drum_events(len(bars), req.genre, beats_per_measure=beats_per_measure,
                                 pattern=drum_pattern_by_id(drum_id), enabled=drum_id is not None)

    midi = render_midi(
        bars, req.genre,
        bpm=bpm, beats_per_measure=beats_per_measure,
        harmony_events=h_events,
        bass_events=b_events,
        drum_events=d_events,
        harmony_instrument=h_instrument,
        bass_instrument_name=b_instrument,
    )

    return SongGenerateResponse(
        genre=req.genre,
        key=key,
        bpm=bpm,
        time_signature=time_signature,
        harmony_rhythm=rhythm["id"],
        drum_pattern=drum_id if drum_id is not None else "off",
        bars=bars,
        tracks={
            "harmony": _events_to_track(h_events, h_instrument),
            "bass":    _events_to_track(b_events, b_instrument),
            "drum":    _events_to_track(d_events, "gm_drum_kit"),
        },
        midi_b64=midi_to_b64(midi),
        model_used=model_used,
        seed=req.seed,
    )
