from __future__ import annotations

import time
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException

from src.common.exceptions import LLMAnalysisError
from src.common.logger import get_logger
from src.models.schemas import RedactedTranscript, SummaryResult

_logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an expert call-center quality analyst. Analyze the provided call transcript \
and return a structured JSON summary. Be concise and grounded in the transcript text — \
do not infer or fabricate information that is not explicitly stated.

Return a JSON object with exactly these fields:
- call_purpose: one sentence describing why the customer called
- key_discussion_points: list of 3–7 strings, each a distinct topic discussed
- action_items: list of objects, each with "description" (str), "owner" (str: "Agent" | "Customer" | named party), and optional "deadline" (YYYY-MM-DD string or null)
- resolution_status: one of "resolved" | "unresolved" | "escalated" | "pending"
- sentiment_trajectory: short phrase describing how sentiment evolved, e.g. "frustrated → satisfied"
- named_entities: list of strings — proper nouns mentioned (people, products, departments, companies)

Respond with JSON only. No markdown fences, no explanation.
"""


def _build_user_message(transcript: RedactedTranscript) -> str:
    lines = [f"Call ID: {transcript.call_id}", "", "Transcript:"]
    if transcript.segments:
        for seg in transcript.segments:
            start = f"{int(seg.start_time // 60):02d}:{int(seg.start_time % 60):02d}"
            lines.append(f"[{start}] {seg.speaker}: {seg.text}")
    else:
        lines.append(transcript.full_text)
    return "\n".join(lines)


def run(
    transcript: RedactedTranscript,
    llm: BaseChatModel | None = None,
) -> SummaryResult:
    """
    Summarize a redacted transcript.

    Retries up to 3 times with exponential backoff on transient API errors.
    Raises LLMAnalysisError if all attempts fail or the output cannot be parsed.
    """
    if llm is None:
        from src.services.llm_factory import get_llm
        llm = get_llm()

    structured_llm = llm.with_structured_output(SummaryResult)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(transcript)),
    ]

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            result = cast(SummaryResult, structured_llm.invoke(messages))
            _logger.info(
                "summarization call_id=%s attempt=%d status=ok",
                transcript.call_id,
                attempt,
            )
            return result
        except (OutputParserException, ValueError) as exc:
            # Structured output parsing failure — not retryable
            raise LLMAnalysisError(
                f"Summarization output could not be parsed: {exc}",
                context={"call_id": transcript.call_id},
            ) from exc
        except Exception as exc:
            last_exc = exc
            wait = 2**attempt
            _logger.warning(
                "summarization call_id=%s attempt=%d error=%s retrying_in=%ds",
                transcript.call_id,
                attempt,
                exc,
                wait,
            )
            if attempt < 3:
                time.sleep(wait)

    raise LLMAnalysisError(
        f"Summarization failed after 3 attempts: {last_exc}",
        context={"call_id": transcript.call_id},
    ) from last_exc
