from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from src.common.logger import get_logger
from src.models.schemas import CallReport
from src.security.audit_logger import log_event

_logger = get_logger(__name__)


def run(
    call_id: str,
    filename: str,
    state: dict,
    status: Literal["completed", "failed", "supervisor_review"] = "completed",
) -> CallReport:
    """
    Assemble a CallReport from upstream pipeline state.

    Phase 6 will add SQLite persistence; this phase builds and returns the report.
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

    log_event(call_id, "report_generated", {"status": status})
    _logger.info("report assembled call_id=%s status=%s", call_id, status)
    return report
