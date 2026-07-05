"""Voyager primary endpoint — `POST /api/generate/song`.

Composite endpoint: voyager
sends only `(genre, length_bars)`, artist owns key / bpm / time-signature
defaults, generates the chord progression with the genre-appropriate
checkpoint, then layers harmony + bass + drum into 3 symbolic tracks
(melody track was dropped 2026-05-10) and ships a MIDI render along.

Response shape (per `SongGenerateResponse`):
    {genre, key, bpm, time_signature,
     bars: [{"chords": [...]}, ...],
     tracks: {harmony, bass, drum},   # each: {events, instrument, source}
     midi_b64,
     model_used}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ChordGenerateRequest,
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
    genre_defaults,
    harmony_instrument,
)
from app.services.inference import (
    TORCH_AVAILABLE,
    ModelRegistry,
    build_prompt,
    dispatch_model_for_genre,
    generate_with_model,
)
from app.services.song_builder import (
    build_bass_events,
    build_drum_events,
    build_harmony_events,
    midi_to_b64,
    render_midi,
)

log = logging.getLogger(__name__)

router = APIRouter()


def _generate_bars(
    *,
    genre: str,
    key: str,
    bpm: int,
    time_signature: tuple[int, int],
    n_bars: int,
) -> tuple[list[dict], str]:
    """Run the existing chord-generation chain (AI when available, rules
    otherwise) and return (bars, model_used). Mirrors the path in
    `routes/chords.py:generate_chords` but with no candidate fan-out.
    """
    model_key = dispatch_model_for_genre(genre)

    if TORCH_AVAILABLE:
        registry = ModelRegistry.get()
        tokenizer = registry.tokenizer
        prompt_ids = build_prompt(
            tokenizer, key, genre, time_signature,
            context=[],
            selected_measures=list(range(n_bars)),
            bpm=bpm,
        )
        loaded = registry.load_model(model_key)
        if loaded is not None:
            try:
                bars = generate_with_model(
                    loaded, tokenizer, prompt_ids, n_bars,
                    temperature=0.8,
                )
                return [{"chords": b} for b in bars], model_key
            except Exception:
                log.exception("AI inference failed, falling back to rules")

    # Rule fallback
    selected = list(range(n_bars))
    full_context = [ContextMeasure(measure=m, chords=[None, None]) for m in selected]
    suggestions = generate_suggestions(
        key=key, genre=genre, context=full_context,
        selected_measures=selected, time_signature=time_signature,
    )
    if suggestions:
        sug = suggestions[0]
        sorted_keys = sorted(sug.chords.keys(), key=int)
        bars = [sug.chords[k] for k in sorted_keys]
    else:
        bars = [["Cmaj", "Cmaj"]] * n_bars
    return [{"chords": b} for b in bars], "rules"


def _events_to_track(events: list[dict], instrument: str) -> Track:
    return Track(
        events=[TrackEvent(**e) for e in events],
        instrument=instrument,
        source="rule",
    )


@router.post("/generate/song")
def generate_song(req: SongGenerateRequest) -> SongGenerateResponse:
    """Voyager primary path: genre + length → full multi-track render."""
    defaults = genre_defaults(req.genre)
    key = defaults["key"]
    bpm = defaults["bpm"]
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

    bars, model_used = _generate_bars(
        genre=req.genre,
        key=key,
        bpm=bpm,
        time_signature=time_signature,
        n_bars=req.length_bars,
    )

    h_events = build_harmony_events(bars, req.genre, beats_per_measure=beats_per_measure)
    b_events = build_bass_events(bars, req.genre, beats_per_measure=beats_per_measure)
    d_events = build_drum_events(req.length_bars, req.genre, beats_per_measure=beats_per_measure)

    midi = render_midi(
        bars, req.genre,
        bpm=bpm, beats_per_measure=beats_per_measure,
        harmony_events=h_events,
        bass_events=b_events,
        drum_events=d_events,
    )

    return SongGenerateResponse(
        genre=req.genre,
        key=key,
        bpm=bpm,
        time_signature=time_signature,
        bars=bars,
        tracks={
            "harmony": _events_to_track(h_events, harmony_instrument(req.genre)),
            "bass":    _events_to_track(b_events, bass_instrument(req.genre)),
            "drum":    _events_to_track(d_events, "gm_drum_kit"),
        },
        midi_b64=midi_to_b64(midi),
        model_used=model_used,
    )
