from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    ActionItem,
    AudioInput,
    AudioProperties,
    AuditEvent,
    CallReport,
    ComplianceFlag,
    InjectionCheckResult,
    IntakeResult,
    QADimension,
    QAScoreResult,
    RedactedTranscript,
    SummaryResult,
    TranscriptionCacheEntry,
    TranscriptionResult,
    TranscriptionSegment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _segment(**kwargs) -> TranscriptionSegment:
    defaults = dict(speaker="Agent", text="Hello.", start_time=0.0, end_time=1.5, confidence=0.95)
    return TranscriptionSegment(**{**defaults, **kwargs})


def _audio_props(**kwargs) -> AudioProperties:
    defaults = dict(format="mp3", duration_seconds=120.0, file_size_bytes=2_000_000, sha256_hash="abc123")
    return AudioProperties(**{**defaults, **kwargs})


def _transcription(**kwargs) -> TranscriptionResult:
    defaults = dict(
        call_id="call-1",
        full_text="Hello.",
        segments=[_segment()],
        language="en",
        duration_seconds=120.0,
        sha256_hash="abc123",
    )
    return TranscriptionResult(**{**defaults, **kwargs})


def _qa_dimension(name: str, score: float = 4.0) -> QADimension:
    return QADimension(name=name, score=score, justification="good")


def _all_dimensions(score: float = 4.0) -> list[QADimension]:
    return [_qa_dimension(n, score) for n in QAScoreResult.WEIGHTS]


def _summary(**kwargs) -> SummaryResult:
    defaults = dict(
        call_purpose="billing inquiry",
        key_discussion_points=["point A", "point B", "point C"],
        resolution_status="resolved",
        sentiment_trajectory="neutral",
    )
    return SummaryResult(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# 1. AudioInput
# ---------------------------------------------------------------------------


def test_audio_input_minimal():
    obj = AudioInput(file_path="/tmp/call.mp3", filename="call.mp3")
    assert obj.file_path == "/tmp/call.mp3"
    assert obj.caller_id is None


def test_audio_input_with_optional_fields():
    obj = AudioInput(file_path="/tmp/x.wav", filename="x.wav", caller_id="555-1234", department="billing")
    assert obj.caller_id == "555-1234"
    assert obj.department == "billing"


# ---------------------------------------------------------------------------
# 2. AudioProperties
# ---------------------------------------------------------------------------


def test_audio_properties_valid():
    obj = _audio_props()
    assert obj.format == "mp3"
    assert obj.sha256_hash == "abc123"


def test_audio_properties_optional_sample_rate_none():
    obj = _audio_props()
    assert obj.sample_rate is None


def test_audio_properties_invalid_format():
    with pytest.raises(ValidationError):
        _audio_props(format="ogg")


# ---------------------------------------------------------------------------
# 3. IntakeResult
# ---------------------------------------------------------------------------


def test_intake_result_valid():
    obj = IntakeResult(valid=True, call_id="uuid-1", audio_properties=_audio_props(), temp_file_path="/tmp/x.mp3")
    assert obj.valid is True


def test_intake_result_invalid_no_props():
    obj = IntakeResult(valid=False, call_id="uuid-2", validation_error="unsupported format")
    assert obj.audio_properties is None
    assert obj.validation_error == "unsupported format"


# ---------------------------------------------------------------------------
# 4. TranscriptionResult
# ---------------------------------------------------------------------------


def test_transcription_result_valid():
    obj = _transcription()
    assert obj.call_id == "call-1"
    assert obj.from_cache is False


def test_transcription_result_from_cache():
    obj = _transcription(from_cache=True)
    assert obj.from_cache is True


# ---------------------------------------------------------------------------
# 5. InjectionCheckResult
# ---------------------------------------------------------------------------


def test_injection_check_no_match():
    obj = InjectionCheckResult(matched=False)
    assert obj.matched_patterns == []


def test_injection_check_with_match():
    obj = InjectionCheckResult(matched=True, matched_patterns=["ignore_previous"], risk_level="high")
    assert len(obj.matched_patterns) == 1
    assert obj.risk_level == "high"


def test_injection_check_invalid_risk_level():
    with pytest.raises(ValidationError):
        InjectionCheckResult(matched=True, risk_level="extreme")


# ---------------------------------------------------------------------------
# 6. RedactedTranscript
# ---------------------------------------------------------------------------


def test_redacted_transcript_valid():
    obj = RedactedTranscript(call_id="c1", full_text="[REDACTED_SSN]", segments=[_segment()], redaction_count=1)
    assert obj.redaction_count == 1


def test_redacted_transcript_defaults():
    obj = RedactedTranscript(call_id="c1", full_text="clean", segments=[])
    assert obj.redacted_types == []
    assert obj.redaction_count == 0


# ---------------------------------------------------------------------------
# 7. ActionItem
# ---------------------------------------------------------------------------


def test_action_item_with_deadline():
    obj = ActionItem(description="Send refund", owner="Agent", deadline="2025-06-01")
    assert obj.deadline == date(2025, 6, 1)


def test_action_item_no_deadline():
    obj = ActionItem(description="Follow up", owner="Customer")
    assert obj.deadline is None


# ---------------------------------------------------------------------------
# 8. SummaryResult
# ---------------------------------------------------------------------------


def test_summary_result_valid():
    obj = _summary()
    assert obj.resolution_status == "resolved"


def test_summary_result_invalid_resolution_status():
    with pytest.raises(ValidationError):
        _summary(resolution_status="done")


@pytest.mark.parametrize("status", ["resolved", "unresolved", "escalated", "pending"])
def test_summary_all_valid_statuses(status):
    obj = _summary(resolution_status=status)
    assert obj.resolution_status == status


# ---------------------------------------------------------------------------
# 9. QADimension
# ---------------------------------------------------------------------------


def test_qa_dimension_valid():
    obj = _qa_dimension("empathy", 3.5)
    assert obj.score == 3.5


def test_qa_dimension_score_too_low():
    with pytest.raises(ValidationError):
        _qa_dimension("empathy", 0.5)


def test_qa_dimension_score_too_high():
    with pytest.raises(ValidationError):
        _qa_dimension("empathy", 5.5)


def test_qa_dimension_score_boundary_min():
    obj = _qa_dimension("clarity", 1.0)
    assert obj.score == 1.0


def test_qa_dimension_score_boundary_max():
    obj = _qa_dimension("compliance", 5.0)
    assert obj.score == 5.0


def test_qa_dimension_invalid_name():
    with pytest.raises(ValidationError):
        QADimension(name="tone", score=3.0, justification="n/a")


@pytest.mark.parametrize("name", ["professionalism", "empathy", "problem_resolution", "compliance", "clarity"])
def test_qa_dimension_all_valid_names(name):
    obj = _qa_dimension(name)
    assert obj.name == name


# ---------------------------------------------------------------------------
# 10. ComplianceFlag
# ---------------------------------------------------------------------------


def test_compliance_flag_valid():
    obj = ComplianceFlag(description="No verification", severity="high")
    assert obj.severity == "high"


@pytest.mark.parametrize("severity", ["low", "medium", "high", "critical"])
def test_compliance_flag_all_valid_severities(severity):
    obj = ComplianceFlag(description="test", severity=severity)
    assert obj.severity == severity


def test_compliance_flag_invalid_severity():
    with pytest.raises(ValidationError):
        ComplianceFlag(description="bad", severity="urgent")


def test_compliance_flag_invalid_severity_uppercase():
    with pytest.raises(ValidationError):
        ComplianceFlag(description="bad", severity="HIGH")


# ---------------------------------------------------------------------------
# 11. QAScoreResult + compute_overall
# ---------------------------------------------------------------------------


def test_qa_score_result_valid():
    dims = _all_dimensions(4.0)
    obj = QAScoreResult(call_id="c1", dimensions=dims, overall_score=4.0)
    assert obj.overall_score == pytest.approx(4.0, abs=1e-4)


def test_qa_score_overall_always_computed():
    dims = _all_dimensions(4.0)
    obj = QAScoreResult(call_id="c1", dimensions=dims, overall_score=999)
    assert obj.overall_score == pytest.approx(4.0, abs=1e-4)


def test_compute_overall_uniform_scores():
    dims = _all_dimensions(4.0)
    # weights sum to 1.0, so overall = 4.0
    result = QAScoreResult.compute_overall(dims)
    assert result == pytest.approx(4.0, abs=1e-4)


def test_compute_overall_weighted_formula():
    scores = {"professionalism": 3.0, "empathy": 5.0, "problem_resolution": 4.0, "compliance": 2.0, "clarity": 4.0}
    dims = [_qa_dimension(name, score) for name, score in scores.items()]
    expected = 3.0 * 0.15 + 5.0 * 0.20 + 4.0 * 0.30 + 2.0 * 0.20 + 4.0 * 0.15
    assert QAScoreResult.compute_overall(dims) == pytest.approx(expected, abs=1e-4)


def test_qa_weights_sum_to_one():
    total = sum(QAScoreResult.WEIGHTS.values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_qa_dimensions_missing_dimension():
    dims = [_qa_dimension(n) for n in ["professionalism", "empathy", "problem_resolution", "compliance"]]
    with pytest.raises(ValidationError):
        QAScoreResult(call_id="c1", dimensions=dims, overall_score=0.0)


def test_qa_dimensions_duplicate_names():
    dims = [_qa_dimension("empathy") for _ in range(5)]
    with pytest.raises(ValidationError):
        QAScoreResult(call_id="c1", dimensions=dims, overall_score=0.0)


# ---------------------------------------------------------------------------
# 12. CallReport
# ---------------------------------------------------------------------------


def test_call_report_valid():
    dims = _all_dimensions(4.0)
    qa = QAScoreResult(call_id="c1", dimensions=dims, overall_score=4.0)
    obj = CallReport(
        call_id="c1",
        filename="call.mp3",
        analyzed_at=datetime.now(timezone.utc),
        audio_properties=_audio_props(),
        transcription=_transcription(),
        summary=_summary(),
        qa_scores=qa,
        status="completed",
    )
    assert obj.status == "completed"
    assert obj.pdf_bytes is None


def test_call_report_invalid_status():
    dims = _all_dimensions(4.0)
    qa = QAScoreResult(call_id="c1", dimensions=dims, overall_score=4.0)
    with pytest.raises(ValidationError):
        CallReport(
            call_id="c1",
            filename="call.mp3",
            analyzed_at=datetime.now(timezone.utc),
            audio_properties=_audio_props(),
            transcription=_transcription(),
            summary=_summary(),
            qa_scores=qa,
            status="done",
        )


# ---------------------------------------------------------------------------
# 13. AuditEvent
# ---------------------------------------------------------------------------


def test_audit_event_defaults():
    obj = AuditEvent(call_id="c1", action="intake", details="file accepted")
    assert obj.actor == "pipeline"
    assert isinstance(obj.timestamp, datetime)


def test_audit_event_custom_actor():
    obj = AuditEvent(call_id="c1", action="upload", details="user triggered", actor="user")
    assert obj.actor == "user"


# ---------------------------------------------------------------------------
# 14. TranscriptionCacheEntry
# ---------------------------------------------------------------------------


def test_transcription_cache_entry_defaults():
    obj = TranscriptionCacheEntry(sha256_hash="deadbeef", transcription_json='{"call_id": "c1"}')
    assert obj.hit_count == 0
    assert isinstance(obj.created_at, datetime)


def test_transcription_cache_entry_explicit_hit_count():
    obj = TranscriptionCacheEntry(sha256_hash="abc", transcription_json="{}", hit_count=5)
    assert obj.hit_count == 5
