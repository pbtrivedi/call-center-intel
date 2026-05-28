from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from src.common.logger import get_logger
from src.database.models import AuditLog, CallRecord, TranscriptionCache
from src.models.schemas import CallReport, TranscriptionResult

_logger = get_logger(__name__)


def save_call_record(session: Session, report: CallReport) -> CallRecord:
    """Upsert a CallRecord from a CallReport. Returns the persisted row."""
    row_data = {
        "call_id": report.call_id,
        "filename": report.filename,
        "status": report.status,
        "overall_qa_score": report.qa_scores.overall_score,
        "resolution_status": report.summary.resolution_status,
        "duration_seconds": report.audio_properties.duration_seconds,
        "audio_format": report.audio_properties.format,
        "sha256_hash": report.audio_properties.sha256_hash,
        "report_json": json.loads(report.model_dump_json()),
        "analyzed_at": report.analyzed_at,
    }
    stmt = (
        sqlite_insert(CallRecord)
        .values(**row_data)
        .on_conflict_do_update(index_elements=["call_id"], set_=row_data)
    )
    session.execute(stmt)
    session.commit()

    record = session.execute(
        select(CallRecord).where(CallRecord.call_id == report.call_id)
    ).scalar_one()
    _logger.info("saved call_record call_id=%s status=%s", report.call_id, report.status)
    return record


def get_call_history(session: Session, limit: int = 50) -> list[CallRecord]:
    """Return the most recent call records ordered by analyzed_at DESC."""
    rows = session.execute(
        select(CallRecord).order_by(CallRecord.analyzed_at.desc()).limit(limit)
    ).scalars().all()
    return list(rows)


def get_cached_transcription(session: Session, sha256: str) -> TranscriptionResult | None:
    """Return a cached TranscriptionResult by SHA-256 hash, or None on miss."""
    row = session.execute(
        select(TranscriptionCache).where(TranscriptionCache.sha256_hash == sha256)
    ).scalar_one_or_none()

    if row is None:
        return None

    # Increment hit counter
    row.hit_count += 1
    session.commit()

    result = TranscriptionResult.model_validate_json(row.transcription_json)
    _logger.info("cache hit sha256=%s... hits=%d", sha256[:8], row.hit_count)
    return result


def save_transcription_cache(
    session: Session, sha256: str, result: TranscriptionResult
) -> None:
    """Insert a transcription result into the cache. No-op if hash already exists."""
    existing = session.execute(
        select(TranscriptionCache).where(TranscriptionCache.sha256_hash == sha256)
    ).scalar_one_or_none()
    if existing is not None:
        return

    row = TranscriptionCache(
        sha256_hash=sha256,
        transcription_json=result.model_dump_json(),
    )
    session.add(row)
    session.commit()
    _logger.info("cache saved sha256=%s...", sha256[:8])


def log_audit_event(
    session: Session, call_id: str, action: str, details: dict | None = None
) -> None:
    """INSERT an audit event. This is the only write path — no UPDATE or DELETE."""
    row = AuditLog(call_id=call_id, action=action, details=details)
    session.add(row)
    session.commit()
