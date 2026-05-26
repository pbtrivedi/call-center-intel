"""Tests for src/services/mcp_client.py."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.services.mcp_client import get_agent_benchmarks, get_compliance_rules


# ---------------------------------------------------------------------------
# get_compliance_rules
# ---------------------------------------------------------------------------


def test_get_compliance_rules_known_call_type():
    result = get_compliance_rules("billing_issue")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "billing_issue" in result


def test_get_compliance_rules_contains_requirements():
    result = get_compliance_rules("account_inquiry")
    assert "verification" in result.lower() or "disclosure" in result.lower()


def test_get_compliance_rules_unknown_type_returns_empty():
    result = get_compliance_rules("unknown_call_type_xyz")
    assert result == ""


def test_get_compliance_rules_empty_string_returns_empty():
    result = get_compliance_rules("")
    assert result == ""


def test_get_compliance_rules_returns_empty_when_file_missing(monkeypatch):
    # Simulate I/O failure when reading the rules file
    with patch("yaml.safe_load", side_effect=OSError("no file")):
        result = get_compliance_rules("billing_issue")
    assert result == ""


def test_get_compliance_rules_all_defined_types():
    for call_type in ("credit_dispute", "account_inquiry", "billing_issue", "password_reset"):
        result = get_compliance_rules(call_type)
        assert len(result) > 0, f"Expected non-empty rules for {call_type}"


# ---------------------------------------------------------------------------
# get_agent_benchmarks
# ---------------------------------------------------------------------------


def test_get_agent_benchmarks_known_call_type():
    result = get_agent_benchmarks("billing_issue")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "billing_issue" in result


def test_get_agent_benchmarks_contains_all_dimensions():
    result = get_agent_benchmarks("account_inquiry")
    for dim in ("professionalism", "empathy", "problem_resolution", "compliance", "clarity"):
        assert dim in result


def test_get_agent_benchmarks_unknown_type_returns_empty():
    result = get_agent_benchmarks("unknown_xyz")
    assert result == ""


def test_get_agent_benchmarks_all_defined_types():
    for call_type in ("credit_dispute", "account_inquiry", "billing_issue", "password_reset"):
        result = get_agent_benchmarks(call_type)
        assert len(result) > 0, f"Expected non-empty benchmarks for {call_type}"


def test_get_agent_benchmarks_scores_in_valid_range():
    import re
    result = get_agent_benchmarks("credit_dispute")
    # Extract all numeric scores like "3.8/5.0"
    scores = re.findall(r"(\d+\.\d+)/5\.0", result)
    assert len(scores) == 5
    for s in scores:
        assert 1.0 <= float(s) <= 5.0
