"""
End-to-end golden eval — runs the full pipeline from real audio and asserts
against the ground-truth spec in evals/golden/account_unlock_e2e.yaml.

Layers tested:
  audio intake → whisper → PII redaction → summarization → QA scoring → persistence

Usage:
    conda activate call-center-intel
    python evals/eval_e2e_golden.py

Results are logged to LangSmith when LANGSMITH_API_KEY is set.

Prerequisite:
    python scripts/generate_test_audio.py   (creates the audio file)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import yaml  # PyYAML — already a transitive dep via langchain

from src.config.loader import get_settings

_settings = get_settings()
if _settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = _settings.langsmith_project
    os.environ["LANGCHAIN_API_KEY"] = _settings.langsmith_api_key

from src.models.schemas import AudioInput  # noqa: E402

try:
    from langsmith import Client as LangSmithClient
    _ls_client = LangSmithClient() if _settings.langsmith_api_key else None
except ImportError:
    _ls_client = None

_GOLDEN_DIR = Path(__file__).parent / "golden"
_REPO_ROOT = Path(__file__).parent.parent

# ── Result helpers ────────────────────────────────────────────────────────────

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"


class _Result:
    def __init__(self) -> None:
        self._rows: list[tuple[str, str, str]] = []   # (label, status, detail)

    def add(self, label: str, status: str, detail: str = "") -> None:
        self._rows.append((label, status, detail))

    def passed(self) -> int:
        return sum(1 for _, s, _ in self._rows if s == _PASS)

    def failed(self) -> int:
        return sum(1 for _, s, _ in self._rows if s == _FAIL)

    def warnings(self) -> int:
        return sum(1 for _, s, _ in self._rows if s == _WARN)

    def print(self, spec_id: str) -> None:
        width = 60
        print(f"\n{'─' * width}")
        print(f"  {spec_id}")
        print(f"{'─' * width}")
        for label, status, detail in self._rows:
            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(status, "?")
            line = f"  {icon} [{status}]  {label}"
            if detail:
                line += f"\n           {detail}"
            print(line)
        print(f"{'─' * width}")
        print(
            f"  {self.passed()} passed  |  "
            f"{self.failed()} failed  |  "
            f"{self.warnings()} warnings"
        )


# ── Assertion helpers ─────────────────────────────────────────────────────────


def _check_pii(spec: dict, state: dict, result: _Result) -> None:
    redacted = state.get("redacted_transcript")
    if redacted is None:
        result.add("pii/redacted_transcript_present", _FAIL, "redacted_transcript missing from state")
        return

    full_text = redacted.full_text

    for pattern in spec.get("must_be_absent", []):
        if pattern.lower() in full_text.lower():
            result.add(
                f"pii/must_be_absent: {pattern!r}",
                _FAIL,
                "string still present in redacted transcript",
            )
        else:
            result.add(f"pii/must_be_absent: {pattern!r}", _PASS)

    for pattern in spec.get("known_gaps", []):
        if pattern.lower() in full_text.lower():
            result.add(
                f"pii/known_gap not redacted: {pattern!r}",
                _WARN,
                "pii_redactor.py has no pattern for this PII type — consider adding one",
            )
        else:
            result.add(f"pii/known_gap already absent: {pattern!r}", _PASS)

    required_type = spec.get("must_include_pii_type")
    if required_type:
        found_types = list(redacted.redacted_types)
        if required_type in found_types:
            result.add(f"pii/redacted_types includes {required_type}", _PASS, str(found_types))
        else:
            result.add(
                f"pii/redacted_types includes {required_type}",
                _FAIL,
                f"got: {found_types}",
            )


def _check_summary(spec: dict, state: dict, result: _Result) -> None:
    report = state.get("call_report")
    if report is None:
        result.add("summary/call_report_present", _FAIL, "call_report missing from state")
        return

    s = report.summary

    expected_status = spec.get("resolution_status")
    if expected_status:
        if s.resolution_status == expected_status:
            result.add(f"summary/resolution_status == {expected_status!r}", _PASS)
        else:
            result.add(
                f"summary/resolution_status == {expected_status!r}",
                _FAIL,
                f"got: {s.resolution_status!r}",
            )

    sentiment_contains = spec.get("sentiment_trajectory_contains")
    if sentiment_contains:
        if sentiment_contains.lower() in s.sentiment_trajectory.lower():
            result.add(
                f"summary/sentiment_trajectory contains {sentiment_contains!r}",
                _PASS,
                f"got: {s.sentiment_trajectory!r}",
            )
        else:
            result.add(
                f"summary/sentiment_trajectory contains {sentiment_contains!r}",
                _WARN,
                f"got: {s.sentiment_trajectory!r}",
            )

    keywords = spec.get("call_purpose_keywords", [])
    if keywords:
        purpose_lower = s.call_purpose.lower()
        matched = [kw for kw in keywords if kw.lower() in purpose_lower]
        if matched:
            result.add(
                "summary/call_purpose keywords match",
                _PASS,
                f"matched {matched} in {s.call_purpose!r}",
            )
        else:
            result.add(
                "summary/call_purpose keywords match",
                _FAIL,
                f"none of {keywords} found in {s.call_purpose!r}",
            )

    min_points = spec.get("min_key_discussion_points", 0)
    actual_points = len(s.key_discussion_points)
    if actual_points >= min_points:
        result.add(
            f"summary/key_discussion_points >= {min_points}",
            _PASS,
            f"got {actual_points}",
        )
    else:
        result.add(
            f"summary/key_discussion_points >= {min_points}",
            _FAIL,
            f"got {actual_points}",
        )


def _check_qa(spec: dict, state: dict, result: _Result) -> None:
    report = state.get("call_report")
    if report is None:
        return  # already reported above

    qa = report.qa_scores
    overall_min = spec.get("overall_min", 0.0)
    overall_max = spec.get("overall_max", 5.0)

    if overall_min <= qa.overall_score <= overall_max:
        result.add(
            f"qa/overall_score in [{overall_min}, {overall_max}]",
            _PASS,
            f"got {qa.overall_score:.2f}",
        )
    else:
        result.add(
            f"qa/overall_score in [{overall_min}, {overall_max}]",
            _FAIL,
            f"got {qa.overall_score:.2f}",
        )

    dim_map = {d.name: d.score for d in qa.dimensions}
    for dim_name, minimum in spec.get("dimension_mins", {}).items():
        actual = dim_map.get(dim_name)
        if actual is None:
            result.add(f"qa/dimension {dim_name} present", _FAIL, "dimension missing")
        elif actual >= minimum:
            result.add(f"qa/{dim_name} >= {minimum}", _PASS, f"got {actual:.1f}")
        else:
            result.add(f"qa/{dim_name} >= {minimum}", _FAIL, f"got {actual:.1f}")


def _check_transcript(spec: dict, state: dict, result: _Result) -> None:
    transcription = state.get("transcription_result")
    if transcription is None:
        result.add("transcript/transcription_result present", _FAIL, "missing from state")
        return

    unique_speakers = {s.speaker for s in transcription.segments}
    expected_count = spec.get("expected_speaker_count")
    if expected_count is not None:
        if len(unique_speakers) == expected_count:
            result.add(
                f"transcript/speaker_count == {expected_count}",
                _PASS,
                f"speakers: {sorted(unique_speakers)}",
            )
        else:
            result.add(
                f"transcript/speaker_count == {expected_count}",
                _FAIL,
                f"got {len(unique_speakers)}: {sorted(unique_speakers)}",
            )


def _check_compliance(spec: dict, state: dict, result: _Result) -> None:
    report = state.get("call_report")
    if report is None:
        return

    flags = report.qa_scores.compliance_flags
    flag_descriptions = " | ".join(f.description for f in flags) if flags else "(none)"

    if spec.get("expect_ssn_flag"):
        ssn_flagged = any(
            "ssn" in f.description.lower()
            or "social" in f.description.lower()
            or "sensitive" in f.description.lower()
            or "personal" in f.description.lower()
            for f in flags
        )
        if ssn_flagged:
            result.add(
                "compliance/SSN disclosure flagged",
                _PASS,
                flag_descriptions,
            )
        else:
            result.add(
                "compliance/SSN disclosure flagged",
                _WARN,
                f"no SSN-related flag found (LLM non-deterministic). flags: {flag_descriptions}",
            )


# ── Main runner ───────────────────────────────────────────────────────────────


def run_golden(spec_path: Path) -> _Result:
    with spec_path.open() as f:
        spec = yaml.safe_load(f)

    result = _Result()
    audio_path = _REPO_ROOT / spec["audio_file"]

    if not audio_path.exists():
        result.add(
            "audio_file present",
            _FAIL,
            f"{audio_path} not found — run: python scripts/generate_test_audio.py",
        )
        return result

    result.add("audio_file present", _PASS, str(audio_path))

    # ── Run full pipeline ─────────────────────────────────────────────────────
    print(f"  Running pipeline on {audio_path.name}…")
    from src.graph.pipeline import workflow

    audio_input = AudioInput(
        file_path=str(audio_path),
        filename=audio_path.name,
        caller_id="eval-golden",
        department="eval",
    )

    try:
        final_state = workflow.invoke({"audio_input": audio_input})
    except Exception as exc:
        result.add("pipeline/invoke", _FAIL, str(exc))
        return result

    if final_state.get("error"):
        result.add("pipeline/no_error", _FAIL, final_state["error"])
        return result
    result.add("pipeline/no_error", _PASS)

    # ── Pipeline status ───────────────────────────────────────────────────────
    expected_status = spec.get("pipeline", {}).get("expected_status")
    actual_status = getattr(final_state.get("call_report"), "status", None)
    if expected_status and actual_status == expected_status:
        result.add(f"pipeline/status == {expected_status!r}", _PASS)
    elif expected_status:
        result.add(
            f"pipeline/status == {expected_status!r}",
            _FAIL,
            f"got: {actual_status!r}",
        )

    # ── Layer assertions ──────────────────────────────────────────────────────
    _check_transcript(spec.get("transcript", {}), final_state, result)
    _check_pii(spec.get("pii_redaction", {}), final_state, result)
    _check_summary(spec.get("summary", {}), final_state, result)
    _check_qa(spec.get("qa_scores", {}), final_state, result)
    _check_compliance(spec.get("compliance", {}), final_state, result)

    # ── Log to LangSmith ─────────────────────────────────────────────────────
    if _ls_client is not None:
        try:
            _ls_client.create_feedback(
                run_id=None,
                key=f"e2e_golden_{spec['id']}",
                score=1.0 if result.failed() == 0 else 0.0,
                comment=json.dumps({
                    "passed": result.passed(),
                    "failed": result.failed(),
                    "warnings": result.warnings(),
                }),
            )
        except Exception:
            pass

    return result


def main() -> None:
    golden_files = sorted(_GOLDEN_DIR.glob("*_e2e.yaml"))
    if not golden_files:
        print(f"No golden specs found in {_GOLDEN_DIR}")
        sys.exit(1)

    total_pass = total_fail = 0
    for spec_path in golden_files:
        result = run_golden(spec_path)
        with open(spec_path) as f:
            spec_id = yaml.safe_load(f).get("id", spec_path.stem)
        result.print(spec_id)
        total_pass += result.passed()
        total_fail += result.failed()

    print(f"\n{'═' * 60}")
    print(f"  Total: {total_pass} passed  |  {total_fail} failed")
    if _ls_client:
        print(f"  Logged to LangSmith project: {_settings.langsmith_project}")
    print(f"{'═' * 60}\n")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
