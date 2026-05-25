from __future__ import annotations

import hashlib
import wave
from pathlib import Path

from mutagen import File as MutagenFile

from src.common.exceptions import AudioValidationError

_SUPPORTED_FORMATS = ("wav", "mp3", "flac", "m4a")


def detect_format(data: bytes) -> str:
    """Return the audio format by inspecting magic bytes, ignoring file extension.

    Raises AudioValidationError for unsupported content.
    """
    if len(data) < 12:
        raise AudioValidationError(
            "File too small to determine audio format",
            context={"size_bytes": len(data)},
        )

    if data[0:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"

    if data[0:4] == b"fLaC":
        return "flac"

    # ISO Base Media (M4A, MP4): ftyp box at offset 4
    if data[4:8] == b"ftyp":
        return "m4a"

    # MP3: ID3 tag header, or MPEG sync word (FF E*/FF F*)
    if data[0:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"

    raise AudioValidationError(
        f"Unsupported audio format. Supported formats: {', '.join(_SUPPORTED_FORMATS)}",
        context={"magic_bytes": data[:8].hex()},
    )


def get_duration(path: str | Path, fmt: str) -> float:
    """Return audio duration in seconds.

    Uses the stdlib wave module for WAV (reads RIFF header directly) and
    mutagen for all other formats.

    Raises AudioValidationError if duration cannot be determined.
    """
    path = str(path)

    if fmt == "wav":
        try:
            with wave.open(path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate == 0:
                    raise AudioValidationError("WAV file has zero sample rate", context={"path": path})
                return frames / float(rate)
        except wave.Error as e:
            raise AudioValidationError(
                f"Could not read WAV duration: {e}", context={"path": path}
            ) from e

    audio = MutagenFile(path)
    if audio is None or not hasattr(audio, "info") or audio.info is None:
        raise AudioValidationError(
            f"Could not read audio metadata for format '{fmt}'",
            context={"path": path, "format": fmt},
        )
    return float(audio.info.length)


def compute_sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest (64 lowercase hex characters) of data."""
    return hashlib.sha256(data).hexdigest()


def cleanup_temp_files(temp_dir: Path, max_files: int) -> int:
    """Remove the oldest files in temp_dir until at most max_files remain.

    Returns the number of files removed.
    """
    files = sorted(
        [f for f in temp_dir.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
    )
    to_remove = max(0, len(files) - max_files)
    for f in files[:to_remove]:
        f.unlink(missing_ok=True)
    return to_remove
