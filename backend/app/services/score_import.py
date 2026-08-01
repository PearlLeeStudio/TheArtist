"""Check an uploaded score, read it on this server and complete sparse harmony with our own model.

Every upload is checked here before the reader sees it. The reader handles vector
PDFs; in infer_harmony mode TheArtist's Music Transformer then proposes chords for
note-bearing bars that the reading left without harmony.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from pathlib import PurePath

from app.models.score_import import MAX_SCORE_BASE64, MAX_SCORE_BYTES, ScoreImportRequest, ScoreImportResponse
from app.services.score_context import ScoreContextError

# Images pass the upload check; the reader then refuses them with 422.
_EXTENSIONS = {
    "application/pdf": {".pdf"}, "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"}, "image/webp": {".webp"},
}


class ScoreInputError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _attachment(req: ScoreImportRequest) -> None:
    """Check the upload's type, name, size and file signature."""
    if req.mimeType not in _EXTENSIONS:
        raise ScoreInputError("Choose a PDF, PNG, JPEG, or WebP score.", 415)
    if not req.filename.strip() or any(ord(char) < 32 for char in req.filename) \
            or "/" in req.filename or "\\" in req.filename:
        raise ScoreInputError("Use a plain filename without paths or control characters.")
    if PurePath(req.filename).suffix.lower() not in _EXTENSIONS[req.mimeType]:
        raise ScoreInputError("The filename extension does not match the file type.", 415)
    if len(req.dataBase64) > MAX_SCORE_BASE64:
        raise ScoreInputError("The score must be 8 MiB or smaller.", 413)
    try:
        raw = base64.b64decode(req.dataBase64, validate=True)
    except (binascii.Error, ValueError):
        raise ScoreInputError("The score data is not valid base64; choose the file again.") from None
    if not raw:
        raise ScoreInputError("The score file is empty.")
    if len(raw) > MAX_SCORE_BYTES:
        raise ScoreInputError("The score must be 8 MiB or smaller.", 413)
    signatures = {
        "application/pdf": raw.startswith(b"%PDF-"),
        "image/png": raw.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": raw.startswith(b"\xff\xd8\xff"),
        "image/webp": len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP",
    }
    if not signatures[req.mimeType]:
        raise ScoreInputError("The file contents do not match its PDF or image type.", 415)


def import_score(req: ScoreImportRequest, *,
                 on_progress: Callable[[str], None] | None = None) -> ScoreImportResponse:
    """Read a vector PDF and complete sparse harmony using our Music Transformer.

    Uploads never leave the application server.
    """
    _attachment(req)
    progress = on_progress or (lambda _label: None)
    from app.services.score_analysis import LocalScoreUnsupported, analyze_local_score
    try:
        progress("Reading the score's notes, chord symbols and bar lines…")
        response = analyze_local_score(req)
        if req.mode == "infer_harmony":
            from app.services.score_completion import complete_sparse_harmony
            progress("Completing sparse harmony with TheArtist's Music Transformer…")
            response = complete_sparse_harmony(response)
        progress("Preparing the recognized score for review…")
        return response
    except (ScoreContextError, LocalScoreUnsupported) as exc:
        raise ScoreInputError(f"{exc} Import supports readable vector PDFs. Scanned PDFs and images are not supported.", 422) from None
