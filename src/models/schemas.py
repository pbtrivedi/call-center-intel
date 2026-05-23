from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Supporting sub-model (not counted among the 14 primary contracts)
# ---------------------------------------------------------------------------


class TranscriptionSegment(BaseModel):
    speaker: str  # 'Agent' or 'Customer'
    text: str
    start_time: float
    end_time: float
    confidence: float  # 0.0–1.0, derived from Whisper avg_logprob + no_speech_prob


# ---------------------------------------------------------------------------
# 1. AudioInput
# ---------------------------------------------------------------------------


class AudioInput(BaseModel):
    file_path: str  # path to the uploaded/temp audio file — never store raw bytes
    filename: str  # original filename for display
    caller_id: str | None = None
    department: str | None = None


# ---------------------------------------------------------------------------
# 2. AudioProperties
# ---------------------------------------------------------------------------


class AudioProperties(BaseModel):
    format: str  # 'wav' | 'mp3' | 'flac' | 'm4a'
    duration_seconds: float
    file_size_bytes: int
    sha256_hash: str  # computed at intake; drives transcription cache lookup
    sample_rate: int | None = None
    channels: int | None = None


# ---------------------------------------------------------------------------
# 3. IntakeResult
# ---------------------------------------------------------------------------


class IntakeResult(BaseModel):
    valid: bool
    call_id: str  # UUID string generated at intake
    audio_properties: AudioProperties | None = None
    temp_file_path: str | None = None
    validation_error: str | None = None


# ---------------------------------------------------------------------------
# 4. TranscriptionResult
# ---------------------------------------------------------------------------


class TranscriptionResult(BaseModel):
    call_id: str
    full_text: str
    segments: list[TranscriptionSegment]
    language: str
    duration_seconds: float
    sha256_hash: str  # matches AudioProperties.sha256_hash; used for cache writes
    from_cache: bool = False


# ---------------------------------------------------------------------------
# 5. InjectionCheckResult
# ---------------------------------------------------------------------------


class InjectionCheckResult(BaseModel):
    matched: bool
    matched_patterns: list[str] = Field(default_factory=list)
    risk_level: str | None = None
    flagged_text: str | None = None


# ---------------------------------------------------------------------------
# 6. RedactedTranscript
# ---------------------------------------------------------------------------


class RedactedTranscript(BaseModel):
    call_id: str
    full_text: str  # PII replaced with typed placeholders
    segments: list[TranscriptionSegment]  # segments also redacted
    redacted_types: list[str] = Field(default_factory=list)  # e.g. ['SSN', 'CREDIT_CARD']
    redaction_count: int = 0


# ---------------------------------------------------------------------------
# 7. ActionItem
# ---------------------------------------------------------------------------


class ActionItem(BaseModel):
    description: str
    owner: str  # 'Agent', 'Customer', or a named party
    deadline: str | None = None  # ISO date string e.g. '2025-06-01'


# ---------------------------------------------------------------------------
# 8. SummaryResult
# ---------------------------------------------------------------------------


class SummaryResult(BaseModel):
    call_purpose: str
    key_discussion_points: list[str]  # 3–7 items
    action_items: list[ActionItem] = Field(default_factory=list)
    resolution_status: Literal["resolved", "unresolved", "escalated", "pending"]
    sentiment_trajectory: str  # e.g. 'frustrated → resolved'
    named_entities: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 9. QADimension
# ---------------------------------------------------------------------------


class QADimension(BaseModel):
    name: Literal["professionalism", "empathy", "problem_resolution", "compliance", "clarity"]
    score: float  # 1.0–5.0
    justification: str
    transcript_references: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def _validate_score(cls, v: float) -> float:
        if not 1.0 <= v <= 5.0:
            raise ValueError(f"score must be between 1.0 and 5.0, got {v}")
        return v


# ---------------------------------------------------------------------------
# 10. ComplianceFlag
# ---------------------------------------------------------------------------


class ComplianceFlag(BaseModel):
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    timestamp: str | None = None  # reference to transcript timestamp
    regulation: str | None = None  # rule or requirement that was violated


# ---------------------------------------------------------------------------
# 11. QAScoreResult
# ---------------------------------------------------------------------------


class QAScoreResult(BaseModel):
    call_id: str
    dimensions: list[QADimension]  # exactly 5 dimensions, one per name
    overall_score: float  # always set by compute_overall(); never taken from the LLM
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)

    # Dimension weights used in the deterministic scoring formula.
    WEIGHTS: ClassVar[dict[str, float]] = {
        "professionalism": 0.15,
        "empathy": 0.20,
        "problem_resolution": 0.30,
        "compliance": 0.20,
        "clarity": 0.15,
    }

    @staticmethod
    def compute_overall(dimensions: list[QADimension]) -> float:
        weights = QAScoreResult.WEIGHTS
        return round(sum(weights[d.name] * d.score for d in dimensions), 4)


# ---------------------------------------------------------------------------
# 12. CallReport
# ---------------------------------------------------------------------------


class CallReport(BaseModel):
    call_id: str
    filename: str
    analyzed_at: datetime
    audio_properties: AudioProperties
    transcription: TranscriptionResult
    summary: SummaryResult
    qa_scores: QAScoreResult
    status: Literal["completed", "failed", "supervisor_review"]
    pdf_bytes: bytes | None = None


# ---------------------------------------------------------------------------
# 13. AuditEvent
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    call_id: str
    action: str  # e.g. 'intake', 'pii_redaction', 'llm_analysis'
    details: str
    timestamp: datetime = Field(default_factory=_utcnow)
    actor: str = "pipeline"


# ---------------------------------------------------------------------------
# 14. TranscriptionCacheEntry
# ---------------------------------------------------------------------------


class TranscriptionCacheEntry(BaseModel):
    sha256_hash: str
    transcription_json: str  # JSON-serialised TranscriptionResult
    created_at: datetime = Field(default_factory=_utcnow)
    hit_count: int = 0
