"""Tests for src/agents/summarization_agent.py."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.agents import summarization_agent
from src.common.exceptions import LLMAnalysisError
from src.models.schemas import (
    ActionItem,
    RedactedTranscript,
    SummaryResult,
    TranscriptionSegment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_redacted(call_id: str = "test-call", full_text: str = "Agent: Hello. Customer: Hi.") -> RedactedTranscript:
    return RedactedTranscript(
        call_id=call_id,
        full_text=full_text,
        segments=[
            TranscriptionSegment(speaker="Agent", text="Hello.", start_time=0, end_time=2, confidence=0.9),
            TranscriptionSegment(speaker="Customer", text="Hi.", start_time=2, end_time=4, confidence=0.9),
        ],
        redacted_types=[],
        redaction_count=0,
    )


def _make_summary(**overrides) -> SummaryResult:
    defaults = dict(
        call_purpose="Customer inquired about their account balance.",
        key_discussion_points=["balance inquiry", "account details", "transaction history"],
        action_items=[],
        resolution_status="resolved",
        sentiment_trajectory="neutral → satisfied",
        named_entities=["Acme Bank"],
    )
    return SummaryResult(**{**defaults, **overrides})


def _mock_structured_llm(return_value):
    """Return a mock LLM whose with_structured_output().invoke() returns return_value."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = return_value
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_returns_summary_result():
    summary = _make_summary()
    llm = _mock_structured_llm(summary)
    result = summarization_agent.run(_make_redacted(), llm=llm)
    assert isinstance(result, SummaryResult)
    assert result.call_purpose == summary.call_purpose


def test_run_passes_call_id_in_message():
    summary = _make_summary()
    llm = _mock_structured_llm(summary)
    summarization_agent.run(_make_redacted(call_id="my-call-99"), llm=llm)
    _, messages = llm.with_structured_output.return_value.invoke.call_args
    # messages is passed as positional arg
    invoke_args = llm.with_structured_output.return_value.invoke.call_args[0][0]
    human_content = invoke_args[1].content
    assert "my-call-99" in human_content


def test_run_uses_structured_output_with_summary_result_schema():
    summary = _make_summary()
    llm = _mock_structured_llm(summary)
    summarization_agent.run(_make_redacted(), llm=llm)
    llm.with_structured_output.assert_called_once_with(SummaryResult)


def test_run_with_action_items():
    summary = _make_summary(
        action_items=[ActionItem(description="Send refund", owner="Agent", deadline=date(2026, 6, 1))],
    )
    llm = _mock_structured_llm(summary)
    result = summarization_agent.run(_make_redacted(), llm=llm)
    assert len(result.action_items) == 1
    assert result.action_items[0].owner == "Agent"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_run_retries_on_transient_error():
    summary = _make_summary()
    mock_structured = MagicMock()
    # Fail twice, succeed on third attempt
    mock_structured.invoke.side_effect = [
        RuntimeError("503 Service Unavailable"),
        RuntimeError("503 Service Unavailable"),
        summary,
    ]
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("time.sleep"):  # skip actual sleep
        result = summarization_agent.run(_make_redacted(), llm=mock_llm)

    assert mock_structured.invoke.call_count == 3
    assert isinstance(result, SummaryResult)


def test_run_raises_after_three_failures():
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("API down")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("time.sleep"):
        with pytest.raises(LLMAnalysisError, match="3 attempts"):
            summarization_agent.run(_make_redacted(), llm=mock_llm)

    assert mock_structured.invoke.call_count == 3


def test_run_parse_error_not_retried():
    from langchain_core.exceptions import OutputParserException

    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = OutputParserException("bad json")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with pytest.raises(LLMAnalysisError, match="could not be parsed"):
        summarization_agent.run(_make_redacted(), llm=mock_llm)

    assert mock_structured.invoke.call_count == 1


# ---------------------------------------------------------------------------
# Segment formatting in prompt
# ---------------------------------------------------------------------------


def test_run_includes_speaker_labels_in_prompt():
    summary = _make_summary()
    llm = _mock_structured_llm(summary)
    transcript = _make_redacted(full_text="Agent: Hello. Customer: Hi.")
    summarization_agent.run(transcript, llm=llm)
    human_msg = llm.with_structured_output.return_value.invoke.call_args[0][0][1]
    assert "Agent" in human_msg.content
    assert "Customer" in human_msg.content
