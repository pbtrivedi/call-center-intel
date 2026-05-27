from __future__ import annotations

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from src.common.exceptions import LLMAnalysisError
from src.common.logger import get_logger
from src.models.schemas import (
    ComplianceFlag,
    QADimension,
    QAScoreResult,
    RedactedTranscript,
    SummaryResult,
)

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Intermediate schema for raw LLM output (overall_score excluded from
# validation so we never carry the LLM value into QAScoreResult).
# ---------------------------------------------------------------------------


class _LLMQAOutput(BaseModel):
    """Raw structured output from the LLM — overall_score is discarded."""
    call_id: str
    dimensions: list[QADimension]
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)
    overall_score: float = 0.0  # present in LLM JSON but never used


_SYSTEM_PROMPT = """\
You are a call-center QA specialist scoring an agent interaction. \
Score the agent on exactly five dimensions using a 1–5 scale (1=poor, 5=excellent). \
Base every score and justification strictly on evidence in the transcript — \
never infer facts not explicitly stated.

Scoring rubric:
- professionalism (weight 15%): language quality, greeting/closing, composure, no interruptions
- empathy (weight 20%): active listening, acknowledgement of feelings, rapport, personalised responses
- problem_resolution (weight 30%): root cause identified, solution quality, customer confirmation
- compliance (weight 20%): required disclosures made, identity verification, hold procedures, data safety
- clarity (weight 15%): clear explanations, minimal jargon, structured delivery, confirmed comprehension

Anti-hallucination rules:
1. Every justification must reference specific things said in the transcript.
2. transcript_references must be timestamp strings (e.g. "00:32", "01:15–01:45") from the transcript.
3. If a dimension has no direct evidence, score it 3 and say "no direct evidence" in the justification.
4. Do NOT invent compliance violations or facts not in the transcript.

{mcp_context}

Return a JSON object with:
- call_id: string (copy from context)
- dimensions: list of exactly 5 objects, each with:
    - name: one of professionalism | empathy | problem_resolution | compliance | clarity
    - score: float 1.0–5.0
    - justification: string
    - transcript_references: list of timestamp strings
- overall_score: 0.0 (will be recomputed deterministically — any value you provide is discarded)
- compliance_flags: list of objects, each with:
    - description: string
    - severity: one of low | medium | high | critical
    - timestamp: string or null
    - regulation: string or null

Respond with JSON only. No markdown fences, no explanation.
"""


def _build_user_message(
    transcript: RedactedTranscript,
    summary: SummaryResult,
    mcp_rules: str,
    mcp_benchmarks: str,
    mcp_recent_flags: str,
) -> str:
    lines = [
        f"Call ID: {transcript.call_id}",
        "",
        "## Summary",
        f"Purpose: {summary.call_purpose}",
        f"Resolution: {summary.resolution_status}",
        f"Sentiment: {summary.sentiment_trajectory}",
        "",
        "## Transcript",
    ]
    if transcript.segments:
        for seg in transcript.segments:
            start = f"{int(seg.start_time // 60):02d}:{int(seg.start_time % 60):02d}"
            lines.append(f"[{start}] {seg.speaker}: {seg.text}")
    else:
        lines.append(transcript.full_text)

    if mcp_rules:
        lines += ["", "## Compliance Rules", mcp_rules]
    if mcp_benchmarks:
        lines += ["", "## Historical Benchmarks", mcp_benchmarks]
    if mcp_recent_flags:
        lines += ["", "## Recent Compliance Patterns", mcp_recent_flags]

    return "\n".join(lines)


def run(
    transcript: RedactedTranscript,
    summary: SummaryResult,
    llm: BaseChatModel | None = None,
    call_type: str = "general",
) -> QAScoreResult:
    """
    Score a call on five QA dimensions.

    Fetches MCP context (compliance rules + benchmarks) before invoking the LLM.
    The LLM's overall_score is always discarded; the deterministic weighted formula
    recomputes it via QAScoreResult._enforce_computed_score.

    Retries up to 3 times with exponential backoff on transient API errors.
    """
    if llm is None:
        from src.services.llm_factory import get_llm
        llm = get_llm()

    # Fetch MCP context — gracefully degrade if unavailable
    from src.services.mcp_client import get_compliance_rules, get_agent_benchmarks, get_recent_flags
    mcp_rules = get_compliance_rules(call_type)
    mcp_benchmarks = get_agent_benchmarks(call_type)
    mcp_recent_flags = get_recent_flags(call_type)

    context_parts = []
    if mcp_rules:
        context_parts.append("compliance rules")
    if mcp_benchmarks:
        context_parts.append("historical benchmarks")
    if mcp_recent_flags:
        context_parts.append("recent compliance patterns")
    mcp_context_block = (
        f"Use the {' and '.join(context_parts)} below as reference context "
        "when assessing the compliance dimension and calibrating scores."
        if context_parts else ""
    )

    system_content = _SYSTEM_PROMPT.format(mcp_context=mcp_context_block)

    # We need a raw dict from the LLM and build QAScoreResult ourselves so we can
    # enforce that overall_score is always recomputed (not taken from the LLM).
    # Use structured output but strip overall_score before construction.
    structured_llm = llm.with_structured_output(schema=_LLMQAOutput)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=_build_user_message(transcript, summary, mcp_rules, mcp_benchmarks, mcp_recent_flags)),
    ]

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            raw: _LLMQAOutput = structured_llm.invoke(messages)  # type: ignore[assignment]
            result = QAScoreResult(
                call_id=transcript.call_id,
                dimensions=raw.dimensions,
                compliance_flags=raw.compliance_flags,
                # overall_score intentionally omitted — recomputed by model validator
            )
            _logger.info(
                "qa_scoring call_id=%s attempt=%d overall_score=%.4f",
                transcript.call_id,
                attempt,
                result.overall_score,
            )
            return result
        except (OutputParserException, ValueError) as exc:
            raise LLMAnalysisError(
                f"QA scoring output could not be parsed: {exc}",
                context={"call_id": transcript.call_id},
            ) from exc
        except Exception as exc:
            last_exc = exc
            wait = 2**attempt
            _logger.warning(
                "qa_scoring call_id=%s attempt=%d error=%s retrying_in=%ds",
                transcript.call_id,
                attempt,
                exc,
                wait,
            )
            if attempt < 3:
                time.sleep(wait)

    raise LLMAnalysisError(
        f"QA scoring failed after 3 attempts: {last_exc}",
        context={"call_id": transcript.call_id},
    ) from last_exc
