from __future__ import annotations

from src.models.schemas import TranscriptionResult, TranscriptionSegment


def load_transcription_result(fixture: dict) -> TranscriptionResult:
    t = fixture["transcript"]
    return TranscriptionResult(
        call_id=t["call_id"],
        full_text=t["full_text"],
        segments=[TranscriptionSegment(**seg) for seg in t.get("segments", [])],
        language=t.get("language", "en"),
        duration_seconds=t["duration_seconds"],
        sha256_hash=t["sha256_hash"],
    )
