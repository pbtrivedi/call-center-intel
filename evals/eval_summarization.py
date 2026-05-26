"""
Component evaluation script — Summarization Agent.

Runs each eval fixture through the real LLM and checks output quality.
Results are printed to stdout.  Phase 5 will route them to LangSmith.

Usage:
    python evals/eval_summarization.py
    LLM_PROVIDER=gemini python evals/eval_summarization.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import summarization_agent
from src.models.schemas import (
    RedactedTranscript,
    TranscriptionResult,
    TranscriptionSegment,
)
from src.security.pii_redactor import redact


_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_transcription_result(fixture: dict) -> TranscriptionResult:
    t = fixture["transcript"]
    segments = [
        TranscriptionSegment(**seg) for seg in t.get("segments", [])
    ]
    return TranscriptionResult(
        call_id=t["call_id"],
        full_text=t["full_text"],
        segments=segments,
        language=t.get("language", "en"),
        duration_seconds=t["duration_seconds"],
        sha256_hash=t["sha256_hash"],
    )


def _check_summary(result, expected: dict) -> list[str]:
    failures = []

    exp_purpose = expected["call_purpose"].lower()
    act_purpose = result.call_purpose.lower()
    # Check for key keywords rather than exact match
    key_words = [w for w in exp_purpose.split() if len(w) > 4]
    if not any(w in act_purpose for w in key_words):
        failures.append(
            f"call_purpose mismatch\n  expected keywords from: {exp_purpose!r}\n  got: {result.call_purpose!r}"
        )

    if result.resolution_status != expected["resolution_status"]:
        failures.append(
            f"resolution_status: expected {expected['resolution_status']!r}, got {result.resolution_status!r}"
        )

    if not (3 <= len(result.key_discussion_points) <= 7):
        failures.append(
            f"key_discussion_points count {len(result.key_discussion_points)} not in [3,7]"
        )

    return failures


def run_fixture(fixture_path: Path) -> dict:
    with fixture_path.open() as f:
        fixture = json.load(f)

    print(f"\n{'─' * 60}")
    print(f"Fixture: {fixture['id']} — {fixture['description']}")
    print(f"{'─' * 60}")

    transcription = _load_transcription_result(fixture)
    redacted: RedactedTranscript = redact(transcription)

    try:
        result = summarization_agent.run(redacted)
    except Exception as exc:
        print(f"  FAIL (exception): {exc}")
        return {"fixture_id": fixture["id"], "passed": False, "error": str(exc)}

    expected = fixture["expected"]["summary"]
    failures = _check_summary(result, expected)

    if failures:
        print("  FAIL")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print("  PASS")
        print(f"    call_purpose    : {result.call_purpose}")
        print(f"    resolution      : {result.resolution_status}")
        print(f"    sentiment       : {result.sentiment_trajectory}")
        print(f"    discussion pts  : {len(result.key_discussion_points)}")
        print(f"    action items    : {len(result.action_items)}")

    return {
        "fixture_id": fixture["id"],
        "passed": len(failures) == 0,
        "failures": failures,
        "result": result.model_dump() if not failures else None,
    }


def main() -> None:
    fixtures = sorted(_FIXTURES_DIR.glob("fixture_*.json"))
    if not fixtures:
        print("No fixtures found in", _FIXTURES_DIR)
        sys.exit(1)

    results = [run_fixture(f) for f in fixtures]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'═' * 60}")
    print(f"Summarization eval: {passed}/{total} fixtures passed")
    print(f"{'═' * 60}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
