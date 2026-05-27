"""
Component evaluation script — Summarization Agent.

Runs each eval fixture through the real LLM and checks output quality.
Results are logged to LangSmith when LANGSMITH_API_KEY is set; otherwise printed.

Usage:
    python evals/eval_summarization.py
    LLM_PROVIDER=gemini python evals/eval_summarization.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Initialize LangSmith tracing if API key is present
from src.config.loader import get_settings  # noqa: E402

_settings = get_settings()
if _settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = _settings.langsmith_project
    os.environ["LANGCHAIN_API_KEY"] = _settings.langsmith_api_key

from evals.utils import load_transcription_result  # noqa: E402
from src.agents import summarization_agent  # noqa: E402
from src.models.schemas import RedactedTranscript  # noqa: E402
from src.security.pii_redactor import redact  # noqa: E402

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# LangSmith client — optional
try:
    from langsmith import Client as LangSmithClient

    _ls_client = LangSmithClient() if _settings.langsmith_api_key else None
except ImportError:
    _ls_client = None


def _log_to_langsmith(fixture_id: str, passed: bool, result: dict | None) -> None:
    if _ls_client is None:
        return
    try:
        _ls_client.create_feedback(
            run_id=None,
            key=f"summarization_eval_{fixture_id}",
            score=1.0 if passed else 0.0,
            comment=json.dumps(result or {}),
        )
    except Exception:
        pass


def _check_summary(result, expected: dict) -> list[str]:
    failures = []

    exp_purpose = expected["call_purpose"].lower()
    act_purpose = result.call_purpose.lower()
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

    transcription = load_transcription_result(fixture)
    redacted: RedactedTranscript = redact(transcription)

    try:
        result = summarization_agent.run(redacted)
    except Exception as exc:
        print(f"  FAIL (exception): {exc}")
        outcome = {"fixture_id": fixture["id"], "passed": False, "error": str(exc)}
        _log_to_langsmith(fixture["id"], False, outcome)
        return outcome

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

    outcome = {
        "fixture_id": fixture["id"],
        "passed": len(failures) == 0,
        "failures": failures,
        "result": result.model_dump() if not failures else None,
    }
    _log_to_langsmith(fixture["id"], outcome["passed"], outcome)
    return outcome


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
    if _ls_client:
        print(f"Results logged to LangSmith project: {_settings.langsmith_project}")
    print(f"{'═' * 60}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
