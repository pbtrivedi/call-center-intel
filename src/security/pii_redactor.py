from __future__ import annotations

import re
from typing import NamedTuple

from src.common.exceptions import PIIRedactionError
from src.models.schemas import RedactedTranscript, TranscriptionResult, TranscriptionSegment

# ---------------------------------------------------------------------------
# PII pattern definitions
# ---------------------------------------------------------------------------


class _PIIPattern(NamedTuple):
    label: str       # placeholder text e.g. "[REDACTED_SSN]"
    pii_type: str    # canonical type name e.g. "SSN"
    regex: re.Pattern[str]


_PII_PATTERNS: list[_PIIPattern] = [
    # SSN — 123-45-6789 / 123 45 6789 / 123456789
    _PIIPattern("[REDACTED_SSN]", "SSN", re.compile(
        r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"
        r"|\b\d{9}\b",
        re.IGNORECASE,
    )),

    # Credit cards — Visa/MC/Disc (16-digit), Amex (15-digit)
    # Accepts space or dash separators in standard groupings
    _PIIPattern("[REDACTED_CREDIT_CARD]", "CREDIT_CARD", re.compile(
        # Amex: 4-6-5
        r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b"
        # Visa/MC/Discover: 4-4-4-4
        r"|\b(?:\d{4}[-\s]?){3}\d{4}\b",
        re.IGNORECASE,
    )),

    # Email
    _PIIPattern("[REDACTED_EMAIL]", "EMAIL", re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
    )),

    # Phone — US and international
    # Requires an explicit phone-like structure: parentheses, + country code, or
    # a 3-3-4 grouping with separators. Does NOT match bare digit strings caught by
    # the more specific SSN / credit-card patterns above.
    _PIIPattern("[REDACTED_PHONE]", "PHONE", re.compile(
        # International with + prefix: +1 555 123 4567 / +44 20 7946 0958
        r"\+\d{1,3}[\s.\-]\(?\d{1,4}\)?[\s.\-]\d{1,4}[\s.\-]\d{1,9}"
        # US with parentheses: (555) 123-4567 / (555) 123 4567
        r"|\(\d{3}\)\s*\d{3}[\s.\-]\d{4}"
        # 3-3-4 with separator (dots or dashes): 555-123-4567 / 555.123.4567
        r"|\b\d{3}[.\-]\d{3}[.\-]\d{4}\b",
        re.IGNORECASE,
    )),
]


def _redact_text(text: str) -> tuple[str, list[str], int]:
    """
    Apply all PII patterns to text, replacing right-to-left to preserve offsets.

    Returns (redacted_text, [pii_types_found], match_count).
    """
    # Collect all matches first
    class _Match(NamedTuple):
        start: int
        end: int
        replacement: str
        pii_type: str

    all_matches: list[_Match] = []
    for pattern in _PII_PATTERNS:
        for m in pattern.regex.finditer(text):
            all_matches.append(_Match(m.start(), m.end(), pattern.label, pattern.pii_type))

    if not all_matches:
        return text, [], 0

    # De-overlap: sort by (start, longest first), then greedily pick non-overlapping spans.
    # This ensures SSN/CC (defined before PHONE) win when spans overlap.
    all_matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
    selected: list[_Match] = []
    last_end = 0
    for match in all_matches:
        if match.start >= last_end:
            selected.append(match)
            last_end = match.end

    # Replace right-to-left so earlier indices stay valid
    selected.sort(key=lambda m: m.start, reverse=True)

    redacted = text
    found_types: set[str] = set()
    count = 0

    for match in selected:
        redacted = redacted[: match.start] + match.replacement + redacted[match.end :]
        found_types.add(match.pii_type)
        count += 1

    return redacted, sorted(found_types), count


def redact(transcript: TranscriptionResult) -> RedactedTranscript:
    """
    Redact PII from a TranscriptionResult.

    Applies patterns to full_text AND to each segment independently.
    Raises PIIRedactionError on unexpected failure.
    """
    try:
        redacted_full, types_in_full, count_full = _redact_text(transcript.full_text)

        redacted_segments: list[TranscriptionSegment] = []
        all_types: set[str] = set(types_in_full)
        total_count = count_full

        for seg in transcript.segments:
            redacted_seg_text, seg_types, seg_count = _redact_text(seg.text)
            all_types.update(seg_types)
            total_count += seg_count
            # Rebuild frozen segment with redacted text
            redacted_segments.append(
                TranscriptionSegment(
                    speaker=seg.speaker,
                    text=redacted_seg_text,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    confidence=seg.confidence,
                )
            )

        return RedactedTranscript(
            call_id=transcript.call_id,
            full_text=redacted_full,
            segments=redacted_segments,
            redacted_types=sorted(all_types),
            redaction_count=total_count,
        )

    except Exception as e:
        raise PIIRedactionError(
            f"PII redaction failed: {e}",
            context={"call_id": transcript.call_id},
        ) from e
