"""Tests for src/agents/qa_scoring_agent.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents import qa_scoring_agent
from src.common.exceptions import LLMAnalysisError
from src.models.schemas import (
    ComplianceFlag,
    QADimension,
    QAScoreResult,
    RedactedTranscript,
    SummaryResult,
    TranscriptionSegment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_redacted(call_id: str = "qa-call") -> RedactedTranscript:
    return RedactedTranscript(
        call_id=call_id,
        full_text="Agent: Hello. Customer: Hi.",
        segments=[
            TranscriptionSegment(speaker="Agent", text="Hello.", start_time=0, end_time=2, confidence=0.9),
            TranscriptionSegment(speaker="Customer", text="Hi.", start_time=2, end_time=4, confidence=0.9),
        ],
        redacted_types=[],
        redaction_count=0,
    )


def _make_summary() -> SummaryResult:
    return SummaryResult(
        call_purpose="Balance inquiry",
        key_discussion_points=["balance", "recent transactions", "fees"],
        resolution_status="resolved",
        sentiment_trajectory="neutral → satisfied",
    )


def _make_dimensions(scores: dict[str, float] | None = None) -> list[QADimension]:
    default_scores = {
        "professionalism": 4.0,
        "empathy": 3.5,
        "problem_resolution": 4.0,
        "compliance": 3.0,
        "clarity": 4.5,
    }
    scores = {**default_scores, **(scores or {})}
    return [
        QADimension(name=name, score=score, justification="test")
        for name, score in scores.items()
    ]


def _make_llm_output(dimensions=None, compliance_flags=None, overall_score=9.9):
    """Build a _LLMQAOutput instance with configurable overall_score."""
    dims = dimensions or _make_dimensions()
    flags = compliance_flags or []
    # Bypass QAScoreResult's validator by constructing _LLMQAOutput directly
    from src.agents.qa_scoring_agent import _LLMQAOutput
    return _LLMQAOutput(
        call_id="qa-call",
        dimensions=dims,
        compliance_flags=flags,
        overall_score=overall_score,
    )


def _mock_llm(llm_output):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = llm_output
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


# ---------------------------------------------------------------------------
# overall_score is always recomputed — LLM value discarded
# ---------------------------------------------------------------------------


def test_overall_score_is_recomputed_not_taken_from_llm():
    llm_output = _make_llm_output(overall_score=9.9)  # absurd LLM value
    llm = _mock_llm(llm_output)
    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            result = qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=llm)
    assert result.overall_score != 9.9


def test_overall_score_matches_weighted_formula():
    scores = {
        "professionalism": 4.0,
        "empathy": 3.0,
        "problem_resolution": 5.0,
        "compliance": 2.0,
        "clarity": 4.0,
    }
    dims = _make_dimensions(scores)
    expected = QAScoreResult.compute_overall(dims)
    llm_output = _make_llm_output(dimensions=dims, overall_score=0.0)
    llm = _mock_llm(llm_output)
    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            result = qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=llm)
    assert abs(result.overall_score - expected) < 0.0001


def test_overall_score_correct_for_known_values():
    # professionalism×0.15 + empathy×0.20 + problem_resolution×0.30 + compliance×0.20 + clarity×0.15
    # = 5×0.15 + 5×0.20 + 5×0.30 + 5×0.20 + 5×0.15 = 5.0
    dims = _make_dimensions({
        "professionalism": 5.0,
        "empathy": 5.0,
        "problem_resolution": 5.0,
        "compliance": 5.0,
        "clarity": 5.0,
    })
    llm_output = _make_llm_output(dimensions=dims)
    llm = _mock_llm(llm_output)
    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            result = qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=llm)
    assert abs(result.overall_score - 5.0) < 0.0001


# ---------------------------------------------------------------------------
# Compliance flags
# ---------------------------------------------------------------------------


def test_critical_compliance_flag_in_result():
    flags = [ComplianceFlag(description="Identity not verified", severity="critical", timestamp="00:11")]
    llm_output = _make_llm_output(compliance_flags=flags)
    llm = _mock_llm(llm_output)
    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            result = qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=llm)
    assert any(f.severity == "critical" for f in result.compliance_flags)


def test_no_compliance_flags_when_llm_returns_none():
    llm_output = _make_llm_output(compliance_flags=[])
    llm = _mock_llm(llm_output)
    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            result = qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=llm)
    assert result.compliance_flags == []


# ---------------------------------------------------------------------------
# MCP context injection
# ---------------------------------------------------------------------------


def test_mcp_rules_injected_into_prompt():
    llm_output = _make_llm_output()
    llm = _mock_llm(llm_output)
    compliance_rules = "Call type: billing_issue\nRequired disclosures:\n  - Inform of dispute rights"
    with patch("src.services.mcp_client.get_compliance_rules", return_value=compliance_rules):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=llm)
    human_msg = llm.with_structured_output.return_value.invoke.call_args[0][0][1]
    assert "billing_issue" in human_msg.content


def test_graceful_degradation_when_mcp_unavailable():
    llm_output = _make_llm_output()
    llm = _mock_llm(llm_output)
    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            result = qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=llm)
    assert isinstance(result, QAScoreResult)


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_run_retries_on_transient_error():
    llm_output = _make_llm_output()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = [
        RuntimeError("timeout"),
        RuntimeError("timeout"),
        llm_output,
    ]
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            with patch("time.sleep"):
                result = qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=mock_llm)

    assert mock_structured.invoke.call_count == 3
    assert isinstance(result, QAScoreResult)


def test_run_raises_after_three_failures():
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("API down")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            with patch("time.sleep"):
                with pytest.raises(LLMAnalysisError, match="3 attempts"):
                    qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=mock_llm)

    assert mock_structured.invoke.call_count == 3


def test_parse_error_not_retried():
    from langchain_core.exceptions import OutputParserException

    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = OutputParserException("bad json")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("src.services.mcp_client.get_compliance_rules", return_value=""):
        with patch("src.services.mcp_client.get_agent_benchmarks", return_value=""):
            with pytest.raises(LLMAnalysisError, match="could not be parsed"):
                qa_scoring_agent.run(_make_redacted(), _make_summary(), llm=mock_llm)

    assert mock_structured.invoke.call_count == 1
