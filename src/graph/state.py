from __future__ import annotations

from typing import Optional, TypedDict

from src.models.schemas import (
    AudioInput,
    CallReport,
    InjectionCheckResult,
    IntakeResult,
    QAScoreResult,
    RedactedTranscript,
    SummaryResult,
    TranscriptionResult,
)


class PipelineState(TypedDict, total=False):
    """Accumulated state threaded through every node in the pipeline graph."""

    # --- input ---
    audio_input: AudioInput

    # --- stage outputs ---
    intake_result: Optional[IntakeResult]
    transcription_result: Optional[TranscriptionResult]
    injection_check_result: Optional[InjectionCheckResult]
    redacted_transcript: Optional[RedactedTranscript]
    summary_result: Optional[SummaryResult]
    qa_score_result: Optional[QAScoreResult]
    call_report: Optional[CallReport]

    # --- control flow ---
    error: Optional[str]
