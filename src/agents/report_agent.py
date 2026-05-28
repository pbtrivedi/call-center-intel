from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from src.common.logger import get_logger
from src.database.database import get_session
from src.database.repository import log_audit_event, save_call_record, save_transcription_cache
from src.models.schemas import CallReport

_logger = get_logger(__name__)


def run(
    call_id: str,
    filename: str,
    state: dict,
    status: Literal["completed", "failed", "supervisor_review"] = "completed",
) -> CallReport:
    """
    Assemble a CallReport from upstream pipeline state and persist to SQLite.

    Writes:
      - CallRecord (upsert) with full report JSON
      - TranscriptionCache (insert-if-missing) keyed by SHA-256
      - AuditLog event for report_generated
    """
    intake = state["intake_result"]
    transcription = state["transcription_result"]
    summary = state["summary_result"]
    qa_scores = state["qa_score_result"]

    report = CallReport(
        call_id=call_id,
        filename=filename,
        analyzed_at=datetime.now(timezone.utc),
        audio_properties=intake.audio_properties,
        transcription=transcription,
        summary=summary,
        qa_scores=qa_scores,
        status=status,
    )

    with get_session() as session:
        save_call_record(session, report)
        save_transcription_cache(session, intake.audio_properties.sha256_hash, transcription)
        log_audit_event(session, call_id, "report_generated", {"status": status})

    _logger.info("report assembled call_id=%s status=%s", call_id, status)
    return report
