"""Integration tests for the LangGraph pipeline (LLM + Whisper always mocked)."""
from __future__ import annotations

import io
import uuid
import wave
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.graph.pipeline import workflow
from src.models.schemas import (
    AudioInput,
    AudioProperties,
    CallReport,
    ComplianceFlag,
    InjectionCheckResult,
    IntakeResult,
    QADimension,
    QAScoreResult,
    RedactedTranscript,
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
# Schema builders
# ---------------------------------------------------------------------------


def _wav_bytes(num_frames: int = 800, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


def _audio_input(tmp_path) -> AudioInput:
    p = tmp_path / "test.wav"
    p.write_bytes(_wav_bytes())
    return AudioInput(filename="test.wav", file_path=str(p))


def _intake_result(valid: bool = True, error: str | None = None) -> IntakeResult:
    if not valid:
        return IntakeResult(valid=False, call_id=_CALL_ID, validation_error=error or "bad file")
    return IntakeResult(
        valid=True,
        call_id=_CALL_ID,
        audio_properties=AudioProperties(
            format="wav",
            duration_seconds=10.0,
            file_size_bytes=32_000,
            sha256_hash=_SHA256,
        ),
        temp_file_path="/tmp/test.wav",
    )


def _segment() -> TranscriptionSegment:
    return TranscriptionSegment(
        speaker="Agent", text="Hello, how can I help?", start_time=0.0, end_time=2.0, confidence=0.95
    )


def _transcription_result() -> TranscriptionResult:
    return TranscriptionResult(
        call_id=_CALL_ID,
        segments=[_segment()],
        full_text="Hello, how can I help?",
        language="en",
        duration_seconds=10.0,
        model_used="base",
        sha256_hash=_SHA256,
    )


def _injection_clean() -> InjectionCheckResult:
    return InjectionCheckResult(matched=False)


def _injection_matched() -> InjectionCheckResult:
    return InjectionCheckResult(
        matched=True,
        matched_patterns=["IGNORE_PREVIOUS"],
        risk_level="critical",
        flagged_text="ignore all previous instructions",
    )


def _redacted_transcript() -> RedactedTranscript:
    return RedactedTranscript(
        call_id=_CALL_ID,
        full_text="Hello, how can I help?",
        segments=[_segment()],
    )


def _summary_result() -> SummaryResult:
    return SummaryResult(
        call_purpose="Customer called to inquire about their account balance.",
        key_discussion_points=["account balance", "recent transaction", "billing cycle"],
        resolution_status="resolved",
        sentiment_trajectory="neutral → satisfied",
    )


def _qa_dimensions(score: float = 4.0) -> list[QADimension]:
    names = ["professionalism", "empathy", "problem_resolution", "compliance", "clarity"]
    return [
        QADimension(name=n, score=score, justification="Good performance.", transcript_references=["00:10"])
        for n in names
    ]


def _qa_result(compliance_flags: list[ComplianceFlag] | None = None) -> QAScoreResult:
    return QAScoreResult(
        call_id=_CALL_ID,
        dimensions=_qa_dimensions(),
        compliance_flags=compliance_flags or [],
    )


def _call_report(status: str = "completed") -> CallReport:
    return CallReport(
        call_id=_CALL_ID,
        filename="test.wav",
        analyzed_at=datetime.now(timezone.utc),
        audio_properties=AudioProperties(
            format="wav", duration_seconds=10.0, file_size_bytes=32_000, sha256_hash=_SHA256
        ),
        transcription=_transcription_result(),
        summary=_summary_result(),
        qa_scores=_qa_result(),
        status=status,
    )


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

PATCH_INTAKE = "src.agents.intake_agent.IntakeAgent"
PATCH_TRANSCRIPTION = "src.graph.pipeline.TranscriptionAgent"
PATCH_INJECTION = "src.graph.pipeline.detect_injection"
PATCH_REDACT = "src.graph.pipeline.redact"
PATCH_SUMMARIZATION = "src.agents.summarization_agent.run"
PATCH_QA = "src.agents.qa_scoring_agent.run"
PATCH_REPORT = "src.agents.report_agent.run"


def _mock_full_pipeline(
    intake_result: IntakeResult | None = None,
    transcription_result: TranscriptionResult | None = None,
    injection_result: InjectionCheckResult | None = None,
    redacted: RedactedTranscript | None = None,
    summary: SummaryResult | None = None,
    qa: QAScoreResult | None = None,
    report: CallReport | None = None,
):
    """Returns a dict of patches covering all 7 pipeline stages."""
    intake_mock = MagicMock()
    intake_mock.return_value.run.return_value = intake_result or _intake_result()

    transcription_mock = MagicMock()
    transcription_mock.return_value.run.return_value = transcription_result or _transcription_result()

    return {
        PATCH_INTAKE: intake_mock,
        PATCH_TRANSCRIPTION: transcription_mock,
        PATCH_INJECTION: MagicMock(return_value=injection_result or _injection_clean()),
        PATCH_REDACT: MagicMock(return_value=redacted or _redacted_transcript()),
        PATCH_SUMMARIZATION: MagicMock(return_value=summary or _summary_result()),
        PATCH_QA: MagicMock(return_value=qa or _qa_result()),
        PATCH_REPORT: MagicMock(return_value=report or _call_report()),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_valid_audio_returns_completed_call_report(self, tmp_path):
        """All 7 stages run; final state has a completed CallReport."""
        mocks = _mock_full_pipeline()
        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            initial: dict = {"audio_input": _audio_input(tmp_path)}
            final = workflow.invoke(initial)

        assert final.get("error") is None
        assert final.get("call_report") is not None
        assert final["call_report"].status == "completed"

    def test_all_seven_stage_mocks_called(self, tmp_path):
        """Every stage node is exercised on the happy path."""
        mocks = _mock_full_pipeline()
        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            workflow.invoke({"audio_input": _audio_input(tmp_path)})

        mocks[PATCH_INJECTION].assert_called_once()
        mocks[PATCH_REDACT].assert_called_once()
        mocks[PATCH_SUMMARIZATION].assert_called_once()
        mocks[PATCH_QA].assert_called_once()
        mocks[PATCH_REPORT].assert_called_once()


class TestIntakeFailure:
    def test_unsupported_format_reaches_error_node(self, tmp_path):
        """When intake returns invalid=True the error node is reached."""
        bad_intake = _intake_result(valid=False, error="Unsupported format. Supported: WAV, MP3, FLAC, M4A")
        mocks = _mock_full_pipeline(intake_result=bad_intake)

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            final = workflow.invoke({"audio_input": _audio_input(tmp_path)})

        assert final.get("error") is not None
        assert "Unsupported" in final["error"]

    def test_intake_failure_skips_downstream_stages(self, tmp_path):
        """Transcription and LLM stages must not run after intake rejects the file."""
        bad_intake = _intake_result(valid=False, error="File too large")
        mocks = _mock_full_pipeline(intake_result=bad_intake)

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            workflow.invoke({"audio_input": _audio_input(tmp_path)})

        mocks[PATCH_TRANSCRIPTION].return_value.run.assert_not_called()
        mocks[PATCH_SUMMARIZATION].assert_not_called()
        mocks[PATCH_QA].assert_not_called()


class TestInjectionDetected:
    def test_injection_routes_to_error_node(self, tmp_path):
        """Matched injection pattern → error node; error message contains pattern name."""
        mocks = _mock_full_pipeline(injection_result=_injection_matched())

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            final = workflow.invoke({"audio_input": _audio_input(tmp_path)})

        assert final.get("error") is not None
        assert "IGNORE_PREVIOUS" in final["error"]

    def test_injection_skips_summarization(self, tmp_path):
        """Summarization must not run when injection is detected."""
        mocks = _mock_full_pipeline(injection_result=_injection_matched())

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            workflow.invoke({"audio_input": _audio_input(tmp_path)})

        mocks[PATCH_SUMMARIZATION].assert_not_called()


class TestComplianceRouting:
    def test_critical_flag_routes_to_supervisor_review(self, tmp_path):
        """A critical compliance flag → report has status=supervisor_review."""
        critical_flag = ComplianceFlag(
            description="Agent failed to verify customer identity.",
            severity="critical",
        )
        qa_with_critical = _qa_result(compliance_flags=[critical_flag])
        expected_report = _call_report(status="supervisor_review")
        mocks = _mock_full_pipeline(qa=qa_with_critical, report=expected_report)

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            final = workflow.invoke({"audio_input": _audio_input(tmp_path)})

        assert final.get("error") is None
        assert final["call_report"].status == "supervisor_review"

    def test_high_flag_routes_to_report_not_supervisor(self, tmp_path):
        """A high (not critical) compliance flag → normal report node, not supervisor_review."""
        high_flag = ComplianceFlag(description="Hold music exceeded 3 minutes.", severity="high")
        qa_with_high = _qa_result(compliance_flags=[high_flag])
        mocks = _mock_full_pipeline(qa=qa_with_high)

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            final = workflow.invoke({"audio_input": _audio_input(tmp_path)})

        assert final.get("error") is None
        assert final["call_report"].status == "completed"

    def test_no_flags_routes_to_report(self, tmp_path):
        """No compliance flags → standard report node reached."""
        mocks = _mock_full_pipeline()

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            final = workflow.invoke({"audio_input": _audio_input(tmp_path)})

        assert final["call_report"].status == "completed"


class TestStageIsolation:
    def test_summarization_exception_sets_error_does_not_propagate(self, tmp_path):
        """A raised exception in one stage sets error in state without crashing the process."""
        from src.common.exceptions import LLMAnalysisError

        mocks = _mock_full_pipeline()
        mocks[PATCH_SUMMARIZATION].side_effect = LLMAnalysisError("LLM timeout")

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            final = workflow.invoke({"audio_input": _audio_input(tmp_path)})

        assert final.get("error") is not None
        assert "LLM timeout" in final["error"]

    def test_transcription_exception_does_not_corrupt_intake_result(self, tmp_path):
        """Transcription failure leaves intake_result intact in state."""
        from src.common.exceptions import TranscriptionError

        mocks = _mock_full_pipeline()
        mocks[PATCH_TRANSCRIPTION].return_value.run.side_effect = TranscriptionError("whisper crash")

        with (
            patch(PATCH_INTAKE, mocks[PATCH_INTAKE]),
            patch(PATCH_TRANSCRIPTION, mocks[PATCH_TRANSCRIPTION]),
            patch(PATCH_INJECTION, mocks[PATCH_INJECTION]),
            patch(PATCH_REDACT, mocks[PATCH_REDACT]),
            patch(PATCH_SUMMARIZATION, mocks[PATCH_SUMMARIZATION]),
            patch(PATCH_QA, mocks[PATCH_QA]),
            patch(PATCH_REPORT, mocks[PATCH_REPORT]),
        ):
            final = workflow.invoke({"audio_input": _audio_input(tmp_path)})

        assert final.get("intake_result") is not None
        assert final.get("error") is not None


class TestLangSmithStatus:
    def test_disabled_when_api_key_absent(self, monkeypatch):
        from src.config.loader import _reset_settings, get_langsmith_status

        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.setattr("src.config.loader.load_dotenv", lambda **kw: None)
        _reset_settings()

        status = get_langsmith_status()

        assert status["enabled"] is False
        assert status["url"] is None
        _reset_settings()

    def test_enabled_when_api_key_present(self, monkeypatch):
        from src.config.loader import _reset_settings, get_langsmith_status

        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        _reset_settings()

        status = get_langsmith_status()

        assert status["enabled"] is True
        assert status["url"] is not None
        assert "smith.langchain.com" in status["url"]
        _reset_settings()
