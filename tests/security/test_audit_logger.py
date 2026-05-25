"""Tests for src/security/audit_logger.py."""
from __future__ import annotations

import json
from unittest.mock import patch

from src.security.audit_logger import log_event


def test_log_event_does_not_raise():
    log_event("call-123", "transcription_complete")


def test_log_event_with_details_does_not_raise():
    log_event("call-123", "pii_redacted", details={"count": 3, "types": ["SSN"]})


def test_log_event_emits_structured_log():
    with patch("src.security.audit_logger._logger") as mock_log:
        log_event("call-abc", "injection_check", details={"matched": False})
        mock_log.info.assert_called_once()
        call_args = mock_log.info.call_args
        # Second positional arg is the JSON string
        logged_json = call_args[0][1]
        entry = json.loads(logged_json)
        assert entry["call_id"] == "call-abc"
        assert entry["action"] == "injection_check"
        assert "timestamp" in entry
        assert entry["details"]["matched"] is False


def test_log_event_no_details_uses_empty_dict():
    with patch("src.security.audit_logger._logger") as mock_log:
        log_event("call-xyz", "intake_validated")
        call_args = mock_log.info.call_args
        logged_json = call_args[0][1]
        entry = json.loads(logged_json)
        assert entry["details"] == {}


def test_log_event_timestamp_is_iso_format():
    with patch("src.security.audit_logger._logger") as mock_log:
        log_event("call-ts", "test_action")
        logged_json = mock_log.info.call_args[0][1]
        entry = json.loads(logged_json)
        from datetime import datetime
        # Should parse without raising
        datetime.fromisoformat(entry["timestamp"])
