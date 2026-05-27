"""Tests for src/services/mcp_client.py."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.services.mcp_client import get_agent_benchmarks, get_compliance_rules, get_recent_flags


# ---------------------------------------------------------------------------
# get_compliance_rules — tests mock the server so no filesystem access needed
# ---------------------------------------------------------------------------

_RULES_BILLING = "Call type: billing_issue\nRequired disclosures:\n  - Inform of right to dispute"
_RULES_ACCOUNT = "Call type: account_inquiry\nVerification steps:\n  - Verify full name"


def test_get_compliance_rules_delegates_to_server():
    with patch("mcp_servers.compliance_rules_server.get_compliance_rules", return_value=_RULES_BILLING) as mock_srv:
        result = get_compliance_rules("billing_issue")
    mock_srv.assert_called_once_with("billing_issue")
    assert result == _RULES_BILLING


def test_get_compliance_rules_contains_requirements():
    with patch("mcp_servers.compliance_rules_server.get_compliance_rules", return_value=_RULES_ACCOUNT):
        result = get_compliance_rules("account_inquiry")
    assert "verification" in result.lower() or "disclosure" in result.lower()


def test_get_compliance_rules_unknown_type_returns_empty():
    with patch("mcp_servers.compliance_rules_server.get_compliance_rules", return_value=""):
        result = get_compliance_rules("unknown_call_type_xyz")
    assert result == ""


def test_get_compliance_rules_empty_string_returns_empty():
    with patch("mcp_servers.compliance_rules_server.get_compliance_rules", return_value=""):
        result = get_compliance_rules("")
    assert result == ""


def test_get_compliance_rules_returns_empty_on_server_error():
    with patch("mcp_servers.compliance_rules_server.get_compliance_rules", side_effect=OSError("no file")):
        result = get_compliance_rules("billing_issue")
    assert result == ""


def test_get_compliance_rules_all_defined_types():
    for call_type in ("credit_dispute", "account_inquiry", "billing_issue", "password_reset"):
        stub = f"Call type: {call_type}\nRequired disclosures:\n  - Some rule"
        with patch("mcp_servers.compliance_rules_server.get_compliance_rules", return_value=stub):
            result = get_compliance_rules(call_type)
        assert len(result) > 0, f"Expected non-empty rules for {call_type}"


# ---------------------------------------------------------------------------
# get_agent_benchmarks — server uses in-memory data, no filesystem access
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
    scores = re.findall(r"(\d+\.\d+)/5\.0", result)
    assert len(scores) == 5
    for s in scores:
        assert 1.0 <= float(s) <= 5.0


# ---------------------------------------------------------------------------
# get_recent_flags — server uses in-memory data, no filesystem access
# ---------------------------------------------------------------------------


def test_get_recent_flags_delegates_to_server():
    stub = "Most common compliance flags for 'billing_issue':\n  [MEDIUM] Dispute rights not communicated (seen 15x recently)"
    with patch("mcp_servers.historical_stats_server.get_recent_flags", return_value=stub) as mock_srv:
        result = get_recent_flags("billing_issue")
    mock_srv.assert_called_once_with("billing_issue")
    assert result == stub


def test_get_recent_flags_known_call_type():
    result = get_recent_flags("billing_issue")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "billing_issue" in result


def test_get_recent_flags_unknown_type_returns_empty():
    result = get_recent_flags("unknown_xyz")
    assert result == ""


def test_get_recent_flags_returns_empty_on_server_error():
    with patch("mcp_servers.historical_stats_server.get_recent_flags", side_effect=RuntimeError("down")):
        result = get_recent_flags("billing_issue")
    assert result == ""


def test_get_recent_flags_all_defined_types():
    for call_type in ("credit_dispute", "account_inquiry", "billing_issue", "password_reset"):
        result = get_recent_flags(call_type)
        assert len(result) > 0, f"Expected non-empty flags for {call_type}"
