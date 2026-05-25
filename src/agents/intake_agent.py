from __future__ import annotations

import re
import uuid
from pathlib import Path

from src.common.exceptions import AudioValidationError
from src.common.logger import get_logger
from src.config.loader import Settings, get_settings
from src.models.schemas import AudioInput, AudioProperties, IntakeResult
from src.services.audio_utils import (
    cleanup_temp_files,
    compute_sha256,
    detect_format,
    get_duration,
)

# Lightweight PII patterns for metadata field scanning (caller_id, department).
# These warn but do NOT block — actual redaction happens in Phase 3.
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                          # SSN
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),                          # credit card
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),  # email
]

_TEMP_SUBDIR = "data/temp"


class IntakeAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(__name__)

    def run(self, audio_input: AudioInput) -> IntakeResult:
        call_id = str(uuid.uuid4())
        self._logger.info("intake call_id=%s file=%s", call_id, audio_input.filename)

        # --- read bytes -------------------------------------------------------
        try:
            data = Path(audio_input.file_path).read_bytes()
        except OSError as e:
            return self._reject(call_id, f"Could not read file: {e}")

        # --- format detection (magic bytes, not extension) --------------------
        try:
            fmt = detect_format(data)
        except AudioValidationError as e:
            return self._reject(call_id, str(e))

        # --- size gate --------------------------------------------------------
        max_bytes = self._settings.max_file_size_mb * 1024 * 1024
        if len(data) > max_bytes:
            return self._reject(
                call_id,
                f"File size {len(data) / 1024**2:.1f} MB exceeds limit of "
                f"{self._settings.max_file_size_mb} MB",
            )

        # --- write temp file (needed by mutagen / whisper for duration/transcription) --
        temp_dir = Path(_TEMP_SUBDIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{call_id}.{fmt}"
        temp_path.write_bytes(data)

        # --- duration gate ----------------------------------------------------
        try:
            duration = get_duration(temp_path, fmt)
        except AudioValidationError as e:
            temp_path.unlink(missing_ok=True)
            return self._reject(call_id, str(e))

        max_seconds = self._settings.max_duration_minutes * 60
        if duration > max_seconds:
            temp_path.unlink(missing_ok=True)
            return self._reject(
                call_id,
                f"Audio duration {duration / 60:.1f} min exceeds limit of "
                f"{self._settings.max_duration_minutes} min",
            )

        # --- SHA-256 ----------------------------------------------------------
        sha256 = compute_sha256(data)

        # --- metadata PII scan (warn only) ------------------------------------
        self._scan_metadata_pii(audio_input)

        # --- temp file cleanup ------------------------------------------------
        cleanup_temp_files(temp_dir, self._settings.max_temp_files)

        # --- success ----------------------------------------------------------
        audio_properties = AudioProperties(
            format=fmt,
            duration_seconds=round(duration, 3),
            file_size_bytes=len(data),
            sha256_hash=sha256,
        )
        self._logger.info(
            "intake ok call_id=%s format=%s duration=%.1fs size=%d sha256=%s...",
            call_id, fmt, duration, len(data), sha256[:8],
        )
        return IntakeResult(
            valid=True,
            call_id=call_id,
            audio_properties=audio_properties,
            temp_file_path=str(temp_path),
        )

    # ------------------------------------------------------------------

    def _reject(self, call_id: str, reason: str) -> IntakeResult:
        self._logger.warning("intake rejected call_id=%s reason=%s", call_id, reason)
        return IntakeResult(valid=False, call_id=call_id, validation_error=reason)

    def _scan_metadata_pii(self, audio_input: AudioInput) -> None:
        fields = {
            "caller_id": audio_input.caller_id or "",
            "department": audio_input.department or "",
        }
        for field_name, value in fields.items():
            for pattern in _PII_PATTERNS:
                if pattern.search(value):
                    self._logger.warning(
                        "PII pattern detected in metadata field=%s filename=%s",
                        field_name,
                        audio_input.filename,
                    )
                    break
