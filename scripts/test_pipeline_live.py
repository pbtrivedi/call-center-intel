"""
Live end-to-end pipeline test against real audio files.

Usage:
    python scripts/test_pipeline_live.py
    LLM_PROVIDER=groq python scripts/test_pipeline_live.py
"""
from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

# Load project .env — all API keys live there
load_dotenv(Path(__file__).parent.parent / ".env")

from src.graph.pipeline import workflow  # noqa: E402
from src.models.schemas import AudioInput  # noqa: E402

# ---------------------------------------------------------------------------
# Audio files to test
# ---------------------------------------------------------------------------

_AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio" / "Call center data samples EN"

_TEST_FILES = [
    _AUDIO_DIR / "1755884171.51632 (EN Support-Billing)" / "1755884171.51632.mp3",
    _AUDIO_DIR / "1735404531.458927 (EN customer support )" / "1735404531.458927.mp3",
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_W = 70


def _banner(text: str, char: str = "═") -> None:
    print(f"\n{char * _W}")
    print(f"  {text}")
    print(f"{char * _W}")


def _section(text: str) -> None:
    print(f"\n{'─' * _W}")
    print(f"  {text}")
    print(f"{'─' * _W}")


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_W - indent, initial_indent=prefix, subsequent_indent=prefix)


def _print_result(path: Path, final: dict, elapsed: float) -> None:
    print(f"\n  File     : {path.name}")
    print(f"  Elapsed  : {elapsed:.1f}s")

    if final.get("error"):
        print(f"  Status   : ERROR")
        print(_wrap(f"Error: {final['error']}"))
        return

    report = final.get("call_report")
    if report is None:
        print("  Status   : NO REPORT GENERATED")
        return

    print(f"  Status   : {report.status.upper()}")

    # --- Transcription summary ---
    _section("Transcription")
    tx = final.get("transcription_result")
    if tx:
        print(f"  Duration : {tx.duration_seconds:.1f}s")
        print(f"  Segments : {len(tx.segments)}")
        speakers = {s.speaker for s in tx.segments}
        print(f"  Speakers : {', '.join(sorted(speakers))}")
        if tx.segments:
            print("  Sample   :")
            for seg in tx.segments[:3]:
                line = f"    [{int(seg.start_time // 60):02d}:{int(seg.start_time % 60):02d}] {seg.speaker}: {seg.text}"
                print(textwrap.shorten(line, width=_W, placeholder="…"))

    # --- PII redaction ---
    redacted = final.get("redacted_transcript")
    if redacted and redacted.redacted_types:
        print(f"\n  PII redacted: {', '.join(redacted.redacted_types)} ({redacted.redaction_count} instances)")
    else:
        print("\n  PII redacted: none detected")

    # --- Summary ---
    _section("Summary")
    s = report.summary
    print(_wrap(f"Purpose: {s.call_purpose}"))
    print(f"    Resolution : {s.resolution_status}")
    print(f"    Sentiment  : {s.sentiment_trajectory}")
    print(f"    Discussion points ({len(s.key_discussion_points)}):")
    for pt in s.key_discussion_points:
        print(_wrap(f"• {pt}", indent=6))
    if s.action_items:
        print(f"    Action items ({len(s.action_items)}):")
        for ai in s.action_items:
            deadline = f" (by {ai.deadline})" if ai.deadline else ""
            print(_wrap(f"• [{ai.owner}] {ai.description}{deadline}", indent=6))

    # --- QA scores ---
    _section("QA Scorecard")
    qa = report.qa_scores
    print(f"  Overall: {qa.overall_score:.2f} / 5.00")
    print()
    weights = {"professionalism": 0.15, "empathy": 0.20, "problem_resolution": 0.30,
               "compliance": 0.20, "clarity": 0.15}
    for dim in qa.dimensions:
        bar = "█" * int(dim.score) + "░" * (5 - int(dim.score))
        print(f"  {dim.name:<22} {bar}  {dim.score:.1f}  (wt {weights[dim.name]:.0%})")
        just = textwrap.shorten(dim.justification, width=55, placeholder="…")
        print(f"    {just}")

    if qa.compliance_flags:
        print(f"\n  Compliance flags ({len(qa.compliance_flags)}):")
        for flag in qa.compliance_flags:
            ts = f" @{flag.timestamp}" if flag.timestamp else ""
            print(_wrap(f"[{flag.severity.upper()}]{ts} {flag.description}", indent=4))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_file(path: Path) -> bool:
    print(f"\n  Running pipeline on: {path.name}")
    audio_input = AudioInput(filename=path.name, file_path=str(path))
    t0 = time.time()
    try:
        final = workflow.invoke({"audio_input": audio_input})
    except Exception as exc:
        print(f"  EXCEPTION: {exc}")
        return False
    elapsed = time.time() - t0
    _print_result(path, final, elapsed)
    return final.get("error") is None


def main() -> None:
    provider = os.environ.get("LLM_PROVIDER", "groq")
    _banner(f"Phase 5 Live Pipeline Test  |  LLM: {provider.upper()}")

    missing = [p for p in _TEST_FILES if not p.exists()]
    if missing:
        for p in missing:
            print(f"  Missing audio file: {p}")
        sys.exit(1)

    results = []
    for audio_path in _TEST_FILES:
        _banner(f"File: {audio_path.parent.name}", char="─")
        ok = run_file(audio_path)
        results.append((audio_path.name, ok))

    _banner("Summary")
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} files processed successfully\n")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
