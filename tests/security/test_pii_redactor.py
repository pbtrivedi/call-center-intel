"""Tests for src/security/pii_redactor.py."""
from __future__ import annotations

import pytest

from src.models.schemas import TranscriptionResult, TranscriptionSegment
from src.security.pii_redactor import _redact_text, redact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transcript(
    full_text: str,
    segments: list[tuple[str, str]] | None = None,
) -> TranscriptionResult:
    """Build a minimal TranscriptionResult for testing."""
    seg_list: list[TranscriptionSegment] = []
    if segments:
        for i, (speaker, text) in enumerate(segments):
            seg_list.append(
                TranscriptionSegment(
                    speaker=speaker,  # type: ignore[arg-type]
                    text=text,
                    start_time=float(i * 2),
                    end_time=float(i * 2 + 1),
                    confidence=0.9,
                )
            )
    return TranscriptionResult(
        call_id="test-call",
        full_text=full_text,
        segments=seg_list,
        language="en",
        duration_seconds=10.0,
        sha256_hash="a" * 64,
    )


# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------


def test_ssn_dashes():
    text, types, count = _redact_text("My SSN is 123-45-6789.")
    assert "[REDACTED_SSN]" in text
    assert "123-45-6789" not in text
    assert "SSN" in types
    assert count >= 1


def test_ssn_spaces():
    text, types, _ = _redact_text("SSN 123 45 6789 on file.")
    assert "[REDACTED_SSN]" in text
    assert "SSN" in types


def test_ssn_no_separators():
    text, types, _ = _redact_text("SSN: 123456789")
    assert "[REDACTED_SSN]" in text


def test_ssn_mid_sentence():
    text, _, _ = _redact_text("Please verify: the SSN is 987-65-4321 for this account.")
    assert "987-65-4321" not in text
    assert "[REDACTED_SSN]" in text


# ---------------------------------------------------------------------------
# Credit card
# ---------------------------------------------------------------------------


def test_credit_card_no_separators():
    text, types, _ = _redact_text("Card number: 4111111111111111")
    assert "[REDACTED_CREDIT_CARD]" in text
    assert "4111111111111111" not in text
    assert "CREDIT_CARD" in types


def test_credit_card_dashes():
    text, types, _ = _redact_text("Visa card: 4111-1111-1111-1111")
    assert "[REDACTED_CREDIT_CARD]" in text


def test_credit_card_spaces():
    text, types, _ = _redact_text("Card on file: 4111 1111 1111 1111")
    assert "[REDACTED_CREDIT_CARD]" in text


def test_amex_format():
    text, types, _ = _redact_text("Amex card: 3714 496353 98431")
    assert "[REDACTED_CREDIT_CARD]" in text


def test_mastercard_format():
    text, types, _ = _redact_text("MC: 5500-0000-0000-0004")
    assert "[REDACTED_CREDIT_CARD]" in text


def test_card_mid_sentence():
    text, _, _ = _redact_text("my card is 4111111111111111 and my email is")
    assert "4111111111111111" not in text
    assert "[REDACTED_CREDIT_CARD]" in text


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def test_email_standard():
    text, types, _ = _redact_text("Email me at john.doe@example.com please.")
    assert "[REDACTED_EMAIL]" in text
    assert "john.doe@example.com" not in text
    assert "EMAIL" in types


def test_email_subdomain():
    text, types, _ = _redact_text("Reach me at user@mail.company.co.uk")
    assert "[REDACTED_EMAIL]" in text


def test_email_plus_addressing():
    text, types, _ = _redact_text("I use john+filter@gmail.com for newsletters.")
    assert "[REDACTED_EMAIL]" in text


def test_email_mid_sentence():
    text, _, _ = _redact_text("my card is 4111111111111111 and my email is user@test.com here.")
    assert "user@test.com" not in text
    assert "[REDACTED_EMAIL]" in text


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------


def test_phone_us_parentheses():
    text, types, _ = _redact_text("Call me at (555) 123-4567.")
    assert "[REDACTED_PHONE]" in text
    assert "PHONE" in types


def test_phone_us_dashes():
    text, types, _ = _redact_text("My number is 555-123-4567.")
    assert "[REDACTED_PHONE]" in text


def test_phone_us_dots():
    text, types, _ = _redact_text("Reach me: 555.123.4567")
    assert "[REDACTED_PHONE]" in text


def test_phone_international_plus1():
    text, types, _ = _redact_text("Call +1 555 123 4567 for support.")
    assert "[REDACTED_PHONE]" in text


def test_phone_international_uk():
    text, types, _ = _redact_text("UK number: +44 20 7946 0958")
    assert "[REDACTED_PHONE]" in text


# ---------------------------------------------------------------------------
# Multiple PII types in one sentence
# ---------------------------------------------------------------------------


def test_multiple_pii_types_in_one_text():
    text = "my card is 4111111111111111 and my email is user@test.com here."
    redacted, types, count = _redact_text(text)
    assert "4111111111111111" not in redacted
    assert "user@test.com" not in redacted
    assert "[REDACTED_CREDIT_CARD]" in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "CREDIT_CARD" in types
    assert "EMAIL" in types
    assert count >= 2


def test_ssn_and_phone():
    text = "SSN 123-45-6789, phone (555) 123-4567"
    redacted, types, count = _redact_text(text)
    assert "123-45-6789" not in redacted
    assert "(555) 123-4567" not in redacted
    assert count >= 2


# ---------------------------------------------------------------------------
# redact() — full TranscriptionResult → RedactedTranscript
# ---------------------------------------------------------------------------


def test_redact_full_text():
    transcript = _make_transcript("My SSN is 123-45-6789.")
    result = redact(transcript)
    assert "123-45-6789" not in result.full_text
    assert "[REDACTED_SSN]" in result.full_text


def test_redact_segments():
    transcript = _make_transcript(
        full_text="My email is user@test.com",
        segments=[
            ("Agent", "Hello, how can I help?"),
            ("Customer", "My email is user@test.com"),
        ],
    )
    result = redact(transcript)
    assert all("user@test.com" not in seg.text for seg in result.segments)
    assert any("[REDACTED_EMAIL]" in seg.text for seg in result.segments)


def test_redact_both_full_and_segments():
    transcript = _make_transcript(
        full_text="Card: 4111111111111111 Email: a@b.com",
        segments=[
            ("Agent", "Card: 4111111111111111"),
            ("Customer", "My email: a@b.com"),
        ],
    )
    result = redact(transcript)
    assert "4111111111111111" not in result.full_text
    assert "a@b.com" not in result.full_text
    assert "CREDIT_CARD" in result.redacted_types
    assert "EMAIL" in result.redacted_types


def test_redact_preserves_call_id():
    transcript = _make_transcript("No PII here.")
    result = redact(transcript)
    assert result.call_id == "test-call"


def test_redact_no_pii_returns_unchanged_text():
    original = "Hello, how can I help you today?"
    transcript = _make_transcript(original)
    result = redact(transcript)
    assert result.full_text == original
    assert result.redacted_types == []
    assert result.redaction_count == 0


def test_redact_count_accurate():
    # Two distinct PII items in full_text + one in segment = 3 total replacements
    transcript = _make_transcript(
        full_text="SSN 123-45-6789 and card 4111111111111111",
        segments=[("Customer", "email is user@test.com")],
    )
    result = redact(transcript)
    assert result.redaction_count >= 3


def test_redact_types_sorted():
    transcript = _make_transcript("SSN 123-45-6789 email user@test.com phone 555-123-4567")
    result = redact(transcript)
    assert result.redacted_types == sorted(result.redacted_types)


def test_redact_segment_count_preserved():
    transcript = _make_transcript(
        full_text="some text",
        segments=[
            ("Agent", "Hello"),
            ("Customer", "Hi"),
            ("Agent", "Goodbye"),
        ],
    )
    result = redact(transcript)
    assert len(result.segments) == 3


def test_redact_segment_speaker_preserved():
    transcript = _make_transcript(
        full_text="SSN 123-45-6789",
        segments=[("Customer", "SSN 123-45-6789")],
    )
    result = redact(transcript)
    assert result.segments[0].speaker == "Customer"


def test_clean_transcript_passes_injection_check_after_redact():
    """Redaction must not introduce injection patterns."""
    transcript = _make_transcript(
        "My card is 4111111111111111. Please help.",
        segments=[("Customer", "My card is 4111111111111111. Please help.")],
    )
    from src.security.injection_detector import detect_injection
    result = redact(transcript)
    check = detect_injection(result.full_text)
    assert check.matched is False
