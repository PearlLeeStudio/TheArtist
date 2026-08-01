"""Local macOS handoff. No shell commands, uploads, or user-file replacement."""

import os
import platform
import subprocess
import uuid
from pathlib import Path

from app.models.export import (
    MAX_EXPORT_BYTES, MIN_MIDI_BYTES, DawOpenResponse, DawTarget, DawTargetsResponse, ExportSpec,
)
from app.services.smf_export import render_midi, safe_title

EXPORT_DIR = Path(__file__).resolve().parents[2] / ".exports"
_APPLICATIONS = Path("/Applications")
_LOGIC_BUNDLE = "com.apple.logic10"
_SPOTLIGHT_TIMEOUT_S = 3
_OPEN_TIMEOUT_S = 10
_SAVE_ATTEMPTS = 5  # fresh random names to try before giving up


class DawError(RuntimeError):
    """A handoff failure with its HTTP status.

    422 unknown target, 409 DAW not installed, 501 not macOS, 500 save or open failed.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _spotlight_finds_logic() -> bool:
    try:
        result = subprocess.run(
            ["/usr/bin/mdfind", f"kMDItemCFBundleIdentifier == '{_LOGIC_BUNDLE}'"],
            capture_output=True, text=True, timeout=_SPOTLIGHT_TIMEOUT_S, check=False, shell=False,
        )
        return result.returncode == 0 and any(
            line.startswith("/") and line.endswith(".app") and Path(line).is_dir()
            for line in result.stdout.splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        return False  # Spotlight may be disabled; ordinary /Applications installs still work.


def _installed(target: str) -> bool:
    """Logic by its Spotlight bundle id or an /Applications bundle; Ableton by an /Applications bundle."""
    if target == "logic" and _spotlight_finds_logic():
        return True
    pattern = "Logic Pro*.app" if target == "logic" else "Ableton Live*.app"
    return any(path.is_dir() for path in _APPLICATIONS.glob(pattern))


def daw_targets() -> DawTargetsResponse:
    mac = platform.system() == "Darwin"
    return DawTargetsResponse(targets=[
        DawTarget(id="logic", name="Logic Pro", available=mac and _installed("logic"), action="open"),
        DawTarget(id="ableton", name="Ableton Live", available=mac and _installed("ableton"), action="reveal"),
    ])


def _write_new_file(path: Path, data: bytes) -> bool:
    """Write data to a file that does not exist yet; False when the name is already taken."""
    try:
        # Create only, never overwrite or follow. Read at call time: O_NOFOLLOW is POSIX-only.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    return True


def _save(data: bytes, title: str) -> Path:
    if (not isinstance(data, bytes) or not data.startswith(b"MThd")
            or not MIN_MIDI_BYTES <= len(data) <= MAX_EXPORT_BYTES):
        raise DawError("The MIDI could not be saved because its data is invalid.", 500)
    try:
        if EXPORT_DIR.is_symlink():
            raise OSError("export directory must not be a symlink")
        EXPORT_DIR.mkdir(mode=0o700, parents=False, exist_ok=True)
        stem = safe_title(title)
        for _ in range(_SAVE_ATTEMPTS):
            path = EXPORT_DIR / f"{stem}-{uuid.uuid4().hex[:12]}.mid"
            if _write_new_file(path, data):
                return path
        raise OSError("could not allocate a unique export filename")
    except OSError:
        raise DawError("The MIDI could not be saved to the backend export directory.", 500) from None


def _open(arguments: list[str]) -> bool:
    """Run /usr/bin/open with an argument array, never through a shell; True when it succeeds."""
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=_OPEN_TIMEOUT_S,
                                check=False, shell=False)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def open_daw(target: str, spec: ExportSpec) -> DawOpenResponse:
    if target not in ("logic", "ableton"):
        raise DawError("Choose Logic Pro or Ableton Live.", 422)
    if platform.system() != "Darwin":
        raise DawError("Opening a DAW is available only on the local macOS backend. Download the MIDI instead.", 501)
    if not _installed(target):
        raise DawError("The selected DAW is not installed on the backend Mac. Download the MIDI instead.", 409)
    path = _save(render_midi(spec), spec.title)
    if target == "logic":
        action = "open"
        opened = (_open(["/usr/bin/open", "-b", _LOGIC_BUNDLE, str(path)])
                  or _open(["/usr/bin/open", "-a", "Logic Pro", str(path)]))
    else:
        action = "reveal"  # Finder shows the file for dragging into Ableton
        opened = _open(["/usr/bin/open", "-R", str(path)])
    if not opened:
        raise DawError(f"macOS could not {action} the MIDI. The saved file remains at {path}.", 500)
    return DawOpenResponse(action=action, path=str(path))
