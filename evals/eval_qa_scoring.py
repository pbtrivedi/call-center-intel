"""
Component evaluation script — QA Scoring Agent.

Runs each eval fixture through the real LLM and checks:
  - Dimension scores are within expected ranges (±tolerance)
  - overall_score matches the deterministic weighted formula
  - Known compliance violations produce the expected severity flags

Phase 5 will route results to LangSmith instead of printing.

Usage:
    python evals/eval_qa_scoring.py
    LLM_PROVIDER=groq python evals/eval_qa_scoring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.utils import load_transcription_result
from src.agents import qa_scoring_agent, summarization_agent
from src.models.schemas import QAScoreResult, RedactedTranscript
from src.security.pii_redactor import redact

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_SCORE_TOLERANCE = 0.5  # each dimension score must be within ±0.5 of expected range


def _check_qa(result: QAScoreResult, expected: dict) -> list[str]:
    failures = []
    score_expectations = expected["qa_scores"]

    dim_map = {d.name: d.score for d in result.dimensions}

    for dim_name, bounds in score_expectations.items():
        if dim_name in ("overall_min", "overall_max"):
            continue
        actual = dim_map.get(dim_name)
        if actual is None:
            failures.append(f"dimension {dim_name!r} missing from result")
            continue
        low, high = bounds["min"] - _SCORE_TOLERANCE, bounds["max"] + _SCORE_TOLERANCE
        if not (low <= actual <= high):
            failures.append(
                f"{dim_name} score {actual:.2f} outside expected [{bounds['min']},{bounds['max']}] "
                f"(tolerance ±{_SCORE_TOLERANCE})"
            )

    overall_min = score_expectations.get("overall_min", 1.0) - _SCORE_TOLERANCE
    overall_max = score_expectations.get("overall_max", 5.0) + _SCORE_TOLERANCE
    if not (overall_min <= result.overall_score <= overall_max):
        failures.append(
            f"overall_score {result.overall_score:.4f} outside expected range "
            f"[{score_expectations.get('overall_min')}, {score_expectations.get('overall_max')}]"
        )

    # Verify overall_score matches formula (key correctness check)
    recomputed = QAScoreResult.compute_overall(result.dimensions)
    if abs(result.overall_score - recomputed) > 0.0001:
        failures.append(
            f"overall_score {result.overall_score:.4f} does not match formula result {recomputed:.4f}"
        )

    return failures


def _check_compliance_flags(result: QAScoreResult, expected: dict) -> list[str]:
    failures = []

    if expected.get("requires_critical_flag"):
        severities = {f.severity for f in result.compliance_flags}
        if "critical" not in severities:
            description = expected.get("compliance_notes", "critical compliance violation expected")
            failures.append(
                f"Expected at least one critical compliance flag ({description}); "
                f"got severities: {severities}"
            )

    return failures


def run_fixture(fixture_path: Path) -> dict:
    with fixture_path.open() as f:
        fixture = json.load(f)

    print(f"\n{'─' * 60}")
    print(f"Fixture: {fixture['id']} — {fixture['description']}")
    print(f"{'─' * 60}")

    transcription = load_transcription_result(fixture)
    redacted: RedactedTranscript = redact(transcription)
    call_type = fixture.get("call_type", "general")

    try:
        summary = summarization_agent.run(redacted)
        result = qa_scoring_agent.run(redacted, summary, call_type=call_type)
    except Exception as exc:
        print(f"  FAIL (exception): {exc}")
        return {"fixture_id": fixture["id"], "passed": False, "error": str(exc)}

    expected = fixture["expected"]
    failures = _check_qa(result, expected)
    failures += _check_compliance_flags(result, expected)

    if failures:
        print("  FAIL")
        for msg in failures:
            print(f"    ✗ {msg}")
    else:
        print("  PASS")
        print(f"    overall_score   : {result.overall_score:.4f}")
        for d in result.dimensions:
            print(f"    {d.name:<22}: {d.score:.1f}")
        if result.compliance_flags:
            print(f"    compliance flags: {len(result.compliance_flags)}")
            for flag in result.compliance_flags:
                print(f"      [{flag.severity.upper()}] {flag.description}")

    return {
        "fixture_id": fixture["id"],
        "passed": len(failures) == 0,
        "failures": failures,
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
    print(f"QA scoring eval: {passed}/{total} fixtures passed")
    print(f"{'═' * 60}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
