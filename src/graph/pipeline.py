"""LangGraph pipeline — wires all 7 processing stages into a single workflow."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents import (
    intake_agent as _intake,
    report_agent as _report,
    summarization_agent as _summarization,
    qa_scoring_agent as _qa,
)
from src.agents.transcription_agent import TranscriptionAgent
from src.common.exceptions import CallCenterIntelError
from src.common.logger import get_logger
from src.graph.routing import (
    route_after_intake,
    route_after_injection_check,
    route_after_qa_scoring,
)
from src.graph.state import PipelineState
from src.models.schemas import AudioInput
from src.security import audit_logger
from src.security.injection_detector import detect_injection
from src.security.pii_redactor import redact

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node functions — each receives PipelineState and returns a partial update dict
# ---------------------------------------------------------------------------


def _intake_node(state: PipelineState) -> dict:
    audio_input: AudioInput = state["audio_input"]
    try:
        agent = _intake.IntakeAgent()
        result = agent.run(audio_input)
        audit_logger.log_event(
            result.call_id,
            "intake",
            {"valid": result.valid, "error": result.validation_error},
        )
        if not result.valid:
            return {
                "intake_result": result,
                "error": result.validation_error or "Audio intake failed",
            }
        return {"intake_result": result}
    except CallCenterIntelError as exc:
        _logger.error("intake_node error: %s", exc)
        return {"error": str(exc)}


def _transcription_node(state: PipelineState) -> dict:
    intake = state["intake_result"]
    try:
        agent = TranscriptionAgent()
        result = agent.run(intake)
        audit_logger.log_event(intake.call_id, "transcription", {"segments": len(result.segments)})
        return {"transcription_result": result}
    except CallCenterIntelError as exc:
        _logger.error("transcription_node call_id=%s error: %s", intake.call_id, exc)
        return {"error": str(exc)}


def _injection_check_node(state: PipelineState) -> dict:
    if state.get("error") or state.get("transcription_result") is None:
        return {}
    transcription = state["transcription_result"]
    intake = state["intake_result"]
    try:
        result = detect_injection(transcription.full_text)
        audit_logger.log_event(
            intake.call_id,
            "injection_check",
            {"matched": result.matched, "patterns": result.matched_patterns},
        )
        if result.matched:
            return {
                "injection_check_result": result,
                "error": (
                    f"Prompt injection detected — patterns: "
                    f"{', '.join(result.matched_patterns)}"
                ),
            }
        return {"injection_check_result": result}
    except CallCenterIntelError as exc:
        _logger.error("injection_check_node call_id=%s error: %s", intake.call_id, exc)
        return {"error": str(exc)}


def _pii_redaction_node(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    transcription = state["transcription_result"]
    intake = state["intake_result"]
    try:
        redacted = redact(transcription)
        audit_logger.log_event(
            intake.call_id,
            "pii_redaction",
            {"pii_types_found": list(redacted.redacted_types)},
        )
        return {"redacted_transcript": redacted}
    except CallCenterIntelError as exc:
        _logger.error("pii_redaction_node call_id=%s error: %s", intake.call_id, exc)
        return {"error": str(exc)}


def _summarization_node(state: PipelineState) -> dict:
    if state.get("error") or state.get("redacted_transcript") is None:
        return {}
    redacted = state["redacted_transcript"]
    intake = state["intake_result"]
    try:
        result = _summarization.run(redacted)
        audit_logger.log_event(
            intake.call_id,
            "summarization",
            {"resolution_status": result.resolution_status},
        )
        return {"summary_result": result}
    except CallCenterIntelError as exc:
        _logger.error("summarization_node call_id=%s error: %s", intake.call_id, exc)
        return {"error": str(exc)}


def _qa_scoring_node(state: PipelineState) -> dict:
    if state.get("error") or state.get("summary_result") is None:
        return {}
    redacted = state["redacted_transcript"]
    summary = state["summary_result"]
    intake = state["intake_result"]
    try:
        result = _qa.run(redacted, summary)
        critical_count = sum(1 for f in result.compliance_flags if f.severity == "critical")
        audit_logger.log_event(
            intake.call_id,
            "qa_scoring",
            {
                "overall_score": result.overall_score,
                "critical_flags": critical_count,
            },
        )
        return {"qa_score_result": result}
    except CallCenterIntelError as exc:
        _logger.error("qa_scoring_node call_id=%s error: %s", intake.call_id, exc)
        return {"error": str(exc)}


def _report_node(state: PipelineState) -> dict:
    intake = state["intake_result"]
    try:
        report = _report.run(
            call_id=intake.call_id,
            filename=state["audio_input"].filename,
            state=state,
            status="completed",
        )
        return {"call_report": report}
    except Exception as exc:
        _logger.error("report_node call_id=%s error: %s", intake.call_id, exc)
        return {"error": str(exc)}


def _supervisor_review_node(state: PipelineState) -> dict:
    intake = state["intake_result"]
    try:
        report = _report.run(
            call_id=intake.call_id,
            filename=state["audio_input"].filename,
            state=state,
            status="supervisor_review",
        )
        audit_logger.log_event(
            intake.call_id,
            "supervisor_review_flagged",
            {"reason": "critical compliance flag"},
        )
        return {"call_report": report}
    except Exception as exc:
        _logger.error("supervisor_review_node call_id=%s error: %s", intake.call_id, exc)
        return {"error": str(exc)}


def _error_node(state: PipelineState) -> dict:
    # Three-level message fallback
    message = state.get("error")
    if not message:
        intake = state.get("intake_result")
        if intake is not None and intake.validation_error:
            message = intake.validation_error
        else:
            message = "An unexpected pipeline error occurred."

    call_id = "unknown"
    intake = state.get("intake_result")
    if intake is not None:
        call_id = intake.call_id

    _logger.error("pipeline error call_id=%s message=%s", call_id, message)
    audit_logger.log_event(call_id, "pipeline_error", {"message": message})
    return {"error": message}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("intake", _intake_node)
    graph.add_node("transcription", _transcription_node)
    graph.add_node("injection_check", _injection_check_node)
    graph.add_node("pii_redaction", _pii_redaction_node)
    graph.add_node("summarization", _summarization_node)
    graph.add_node("qa_scoring", _qa_scoring_node)
    graph.add_node("report", _report_node)
    graph.add_node("supervisor_review", _supervisor_review_node)
    graph.add_node("error", _error_node)

    graph.set_entry_point("intake")

    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {"transcription": "transcription", "error": "error"},
    )
    graph.add_edge("transcription", "injection_check")
    graph.add_conditional_edges(
        "injection_check",
        route_after_injection_check,
        {"pii_redaction": "pii_redaction", "error": "error"},
    )
    graph.add_edge("pii_redaction", "summarization")
    graph.add_edge("summarization", "qa_scoring")
    graph.add_conditional_edges(
        "qa_scoring",
        route_after_qa_scoring,
        {"report": "report", "supervisor_review": "supervisor_review", "error": "error"},
    )
    graph.add_edge("report", END)
    graph.add_edge("supervisor_review", END)
    graph.add_edge("error", END)

    return graph


# Module-level compiled workflow — import this in app.py and the UI
workflow = _build_graph().compile()
