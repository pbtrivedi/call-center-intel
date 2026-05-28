"""Integration tests for Phase 6 — SQLite persistence and PDF/JSON exports."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.database.models import AuditLog, Base, CallRecord, TranscriptionCache
from src.database.repository import (
    get_call_history,
    get_cached_transcription,
    log_audit_event,
    save_call_record,
    save_transcription_cache,
)
from src.models.schemas import (
    ActionItem,
    AudioProperties,
    CallReport,
    ComplianceFlag,
    QADimension,
    QAScoreResult,
    SummaryResult,
    TranscriptionResult,
    TranscriptionSegment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHA256 = "a" * 64
_CALL_ID = "test-call-" + "0" * 27


# ---------------------------------------------------------------------------
# In-memory SQLite fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        yield s


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------


def _segment() -> TranscriptionSegment:
    return TranscriptionSegment(
        speaker="Agent", text="Hello, how can I help?",
        start_time=0.0, end_time=2.0, confidence=0.95,
    )


def _transcription(call_id: str = _CALL_ID, sha256: str = _SHA256) -> TranscriptionResult:
    return TranscriptionResult(
        call_id=call_id,
        segments=[_segment()],
        full_text="Hello, how can I help?",
        language="en",
        duration_seconds=10.0,
        model_used="base",
        sha256_hash=sha256,
    )


def _audio_props(sha256: str = _SHA256) -> AudioProperties:
    return AudioProperties(
        format="mp3", duration_seconds=10.0,
        file_size_bytes=32_000, sha256_hash=sha256,
    )


def _summary() -> SummaryResult:
    return SummaryResult(
        call_purpose="Customer called about billing.",
        key_discussion_points=["billing", "account balance", "payment"],
        resolution_status="resolved",
        sentiment_trajectory="frustrated → satisfied",
    )


def _qa(call_id: str = _CALL_ID) -> QAScoreResult:
    names = ["professionalism", "empathy", "problem_resolution", "compliance", "clarity"]
    dims = [QADimension(name=n, score=4.0, justification="Good.") for n in names]
    return QAScoreResult(call_id=call_id, dimensions=dims)


def _report(call_id: str = _CALL_ID, status: str = "completed") -> CallReport:
    return CallReport(
        call_id=call_id,
        filename="test.mp3",
        analyzed_at=datetime.now(timezone.utc),
        audio_properties=_audio_props(),
        transcription=_transcription(call_id),
        summary=_summary(),
        qa_scores=_qa(call_id),
        status=status,
    )


# ---------------------------------------------------------------------------
# CallRecord tests
# ---------------------------------------------------------------------------


class TestCallRecord:
    def test_save_then_history_returns_record(self, session):
        report = _report()
        save_call_record(session, report)

        history = get_call_history(session, limit=10)

        assert len(history) == 1
        assert history[0].call_id == _CALL_ID
        assert history[0].status == "completed"

    def test_history_ordered_by_analyzed_at_desc(self, session):
        for i in range(3):
            call_id = f"call-{i:04d}" + "0" * 27
            save_call_record(session, _report(call_id=call_id[:36]))

        history = get_call_history(session, limit=10)
        timestamps = [r.analyzed_at for r in history]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_upsert_updates_existing_record(self, session):
        save_call_record(session, _report(status="completed"))

        # Re-save same call_id with different status
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        updated_data = {
            "call_id": _CALL_ID,
            "filename": "test.mp3",
            "status": "supervisor_review",
            "overall_qa_score": 4.0,
            "resolution_status": "resolved",
            "duration_seconds": 10.0,
            "audio_format": "mp3",
            "sha256_hash": _SHA256,
            "report_json": {},
            "analyzed_at": datetime.now(timezone.utc),
        }
        stmt = (
            sqlite_insert(CallRecord)
            .values(**updated_data)
            .on_conflict_do_update(index_elements=["call_id"], set_=updated_data)
        )
        session.execute(stmt)
        session.commit()

        history = get_call_history(session, limit=10)
        assert len(history) == 1
        assert history[0].status == "supervisor_review"

    def test_report_json_round_trips(self, session):
        report = _report()
        save_call_record(session, report)

        row = session.execute(
            select(CallRecord).where(CallRecord.call_id == _CALL_ID)
        ).scalar_one()

        assert row.report_json is not None
        assert row.report_json["call_id"] == _CALL_ID
        assert row.report_json["status"] == "completed"

    def test_qa_score_and_resolution_stored(self, session):
        save_call_record(session, _report())
        row = session.execute(select(CallRecord).where(CallRecord.call_id == _CALL_ID)).scalar_one()
        assert row.overall_qa_score == pytest.approx(4.0)
        assert row.resolution_status == "resolved"


# ---------------------------------------------------------------------------
# TranscriptionCache tests
# ---------------------------------------------------------------------------


class TestTranscriptionCache:
    def test_save_then_retrieve_returns_same_result(self, session):
        tx = _transcription()
        save_transcription_cache(session, _SHA256, tx)

        retrieved = get_cached_transcription(session, _SHA256)

        assert retrieved is not None
        assert retrieved.call_id == _CALL_ID
        assert retrieved.full_text == tx.full_text
        assert len(retrieved.segments) == len(tx.segments)

    def test_miss_returns_none(self, session):
        result = get_cached_transcription(session, "b" * 64)
        assert result is None

    def test_hit_increments_counter(self, session):
        save_transcription_cache(session, _SHA256, _transcription())
        get_cached_transcription(session, _SHA256)
        get_cached_transcription(session, _SHA256)

        row = session.execute(
            select(TranscriptionCache).where(TranscriptionCache.sha256_hash == _SHA256)
        ).scalar_one()
        assert row.hit_count == 2

    def test_duplicate_save_is_noop(self, session):
        tx = _transcription()
        save_transcription_cache(session, _SHA256, tx)
        save_transcription_cache(session, _SHA256, tx)  # second call — no error, no duplicate

        rows = session.execute(select(TranscriptionCache)).scalars().all()
        assert len(rows) == 1

    def test_transcription_json_round_trips(self, session):
        tx = _transcription()
        save_transcription_cache(session, _SHA256, tx)
        retrieved = get_cached_transcription(session, _SHA256)
        assert retrieved.model_dump_json() == tx.model_dump_json()


# ---------------------------------------------------------------------------
# AuditLog tests
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_three_events_produce_three_rows(self, session):
        for action in ("intake", "transcription", "report_generated"):
            log_audit_event(session, _CALL_ID, action, {"ok": True})

        rows = session.execute(select(AuditLog)).scalars().all()
        assert len(rows) == 3

    def test_events_stored_with_correct_fields(self, session):
        log_audit_event(session, _CALL_ID, "intake", {"valid": True})
        row = session.execute(select(AuditLog)).scalar_one()
        assert row.call_id == _CALL_ID
        assert row.action == "intake"
        assert row.details == {"valid": True}

    def test_no_update_path_on_model(self, session):
        # AuditLog must not have an `updated_at` column
        cols = {c.name for c in AuditLog.__table__.columns}
        assert "updated_at" not in cols

    def test_null_details_allowed(self, session):
        log_audit_event(session, _CALL_ID, "ping", None)
        row = session.execute(select(AuditLog)).scalar_one()
        assert row.details is None


# ---------------------------------------------------------------------------
# Session persistence tests
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_record_survives_new_session(self, tmp_path):
        """Data persists when a new session is opened on the same file."""
        db_file = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)

        with factory() as s1:
            save_call_record(s1, _report())

        with factory() as s2:
            history = get_call_history(s2, limit=10)

        assert len(history) == 1
        assert history[0].call_id == _CALL_ID


# ---------------------------------------------------------------------------
# PDF generator tests
# ---------------------------------------------------------------------------


class TestPdfGenerator:
    def test_returns_non_empty_bytes(self):
        from src.services.pdf_generator import generate
        pdf_bytes = generate(_report())
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_output_is_valid_pdf(self):
        from src.services.pdf_generator import generate
        pdf_bytes = generate(_report())
        # PDF files start with the %PDF magic bytes
        assert pdf_bytes[:4] == b"%PDF"

    def test_pdf_with_compliance_flags(self):
        from src.services.pdf_generator import generate
        report = _report()
        # Build report with a compliance flag
        names = ["professionalism", "empathy", "problem_resolution", "compliance", "clarity"]
        dims = [QADimension(name=n, score=3.0, justification="Ok.") for n in names]
        qa = QAScoreResult(
            call_id=_CALL_ID,
            dimensions=dims,
            compliance_flags=[ComplianceFlag(description="Missing verification.", severity="high")],
        )
        report_with_flag = CallReport(
            call_id=_CALL_ID,
            filename="test.mp3",
            analyzed_at=datetime.now(timezone.utc),
            audio_properties=_audio_props(),
            transcription=_transcription(),
            summary=_summary(),
            qa_scores=qa,
            status="completed",
        )
        pdf_bytes = generate(report_with_flag)
        assert pdf_bytes[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------


class TestJsonExport:
    def test_json_export_is_valid(self):
        report = _report()
        raw = report.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["call_id"] == _CALL_ID

    def test_json_round_trips_through_model(self):
        report = _report()
        restored = CallReport.model_validate_json(report.model_dump_json())
        assert restored.call_id == report.call_id
        assert restored.qa_scores.overall_score == report.qa_scores.overall_score
        assert restored.summary.resolution_status == report.summary.resolution_status
