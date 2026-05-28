from __future__ import annotations

from src.graph.state import PipelineState


def route_after_intake(state: PipelineState) -> str:
    """Route to 'error' if intake failed, otherwise continue to 'transcription'."""
    if state.get("error"):
        return "error"
    intake = state.get("intake_result")
    if intake is not None and not intake.valid:
        return "error"
    return "transcription"


def route_after_injection_check(state: PipelineState) -> str:
    """Route to 'error' if an injection pattern was detected, otherwise continue."""
    if state.get("error"):
        return "error"
    check = state.get("injection_check_result")
    if check is not None and check.matched:
        return "error"
    return "pii_redaction"


def route_after_qa_scoring(state: PipelineState) -> str:
    """Route to 'supervisor_review' if any compliance flag is critical, else 'report'."""
    if state.get("error"):
        return "error"
    qa = state.get("qa_score_result")
    if qa is not None:
        for flag in qa.compliance_flags:
            if flag.severity == "critical":
                return "supervisor_review"
    return "report"
