from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Supporting sub-model (not counted among the 14 primary contracts)
# ---------------------------------------------------------------------------


class TranscriptionSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    speaker: Literal["Agent", "Customer"]
    text: str
    start_time: float
    end_time: float
    confidence: float  # 0.0–1.0, derived from Whisper avg_logprob + no_speech_prob

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_time_order(self) -> "TranscriptionSegment":
        if self.start_time > self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) must be <= end_time ({self.end_time})"
            )
        return self


# ---------------------------------------------------------------------------
# 1. AudioInput
# ---------------------------------------------------------------------------


class AudioInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_path: str  # path to the uploaded/temp audio file — never store raw bytes
    filename: str  # original filename for display
    caller_id: str | None = None
    department: str | None = None


# ---------------------------------------------------------------------------
# 2. AudioProperties
# ---------------------------------------------------------------------------


class AudioProperties(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["wav", "mp3", "flac", "m4a"]
    duration_seconds: float
    file_size_bytes: int
    sha256_hash: str  # computed at intake; drives transcription cache lookup
    sample_rate: int | None = None
    channels: int | None = None

    @field_validator("sha256_hash")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        if not _SHA256_RE.fullmatch(v):
            raise ValueError(f"sha256_hash must be 64 lowercase hex characters, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# 3. IntakeResult
# ---------------------------------------------------------------------------


class IntakeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    call_id: str  # UUID string generated at intake
    audio_properties: AudioProperties | None = None
    temp_file_path: str | None = None
    validation_error: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "IntakeResult":
        if self.valid and (self.audio_properties is None or self.temp_file_path is None):
            raise ValueError("valid=True requires both audio_properties and temp_file_path")
        if not self.valid and self.validation_error is None:
            raise ValueError("valid=False requires validation_error")
        return self


# ---------------------------------------------------------------------------
# 4. TranscriptionResult
# ---------------------------------------------------------------------------


class TranscriptionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

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
    model_config = ConfigDict(frozen=True)

    matched: bool
    matched_patterns: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"] | None = None
    flagged_text: str | None = None

    @model_validator(mode="after")
    def _check_risk_level(self) -> "InjectionCheckResult":
        if self.matched and self.risk_level is None:
            raise ValueError("risk_level is required when matched=True")
        return self


# ---------------------------------------------------------------------------
# 6. RedactedTranscript
# ---------------------------------------------------------------------------


class RedactedTranscript(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    full_text: str  # PII replaced with typed placeholders
    segments: list[TranscriptionSegment]  # segments also redacted
    redacted_types: list[str] = Field(default_factory=list)  # e.g. ['SSN', 'CREDIT_CARD']
    redaction_count: int = 0


# ---------------------------------------------------------------------------
# 7. ActionItem
# ---------------------------------------------------------------------------


class ActionItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: str
    owner: str  # 'Agent', 'Customer', or a named party
    deadline: date | None = None


# ---------------------------------------------------------------------------
# 8. SummaryResult
# ---------------------------------------------------------------------------


class SummaryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

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
    model_config = ConfigDict(frozen=True)

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
    model_config = ConfigDict(frozen=True)

    description: str
    severity: Literal["low", "medium", "high", "critical"]
    timestamp: str | None = None  # reference to transcript timestamp
    regulation: str | None = None  # rule or requirement that was violated


# ---------------------------------------------------------------------------
# 11. QAScoreResult
# ---------------------------------------------------------------------------


class QAScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    dimensions: list[QADimension]  # exactly 5 dimensions, one per name
    overall_score: float = 0.0  # always recomputed by _enforce_computed_score; never taken from the LLM
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)

    # Dimension weights used in the deterministic scoring formula.
    WEIGHTS: ClassVar[dict[str, float]] = {
        "professionalism": 0.15,
        "empathy": 0.20,
        "problem_resolution": 0.30,
        "compliance": 0.20,
        "clarity": 0.15,
    }

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, v: list[QADimension]) -> list[QADimension]:
        names = {d.name for d in v}
        expected = set(cls.WEIGHTS)
        if names != expected:
            raise ValueError(f"dimensions must include exactly {expected}, got {names}")
        return v

    @model_validator(mode="after")
    def _enforce_computed_score(self) -> "QAScoreResult":
        object.__setattr__(self, "overall_score", self.compute_overall(self.dimensions))
        return self

    @staticmethod
    def compute_overall(dimensions: list[QADimension]) -> float:
        weights = QAScoreResult.WEIGHTS
        return round(sum(weights[d.name] * d.score for d in dimensions), 4)


# ---------------------------------------------------------------------------
# 12. CallReport
# ---------------------------------------------------------------------------


class CallReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    filename: str
    analyzed_at: datetime
    audio_properties: AudioProperties
    transcription: TranscriptionResult
    summary: SummaryResult
    qa_scores: QAScoreResult
    status: Literal["completed", "failed", "supervisor_review"]
    pdf_bytes: bytes | None = None

    @field_validator("analyzed_at")
    @classmethod
    def _require_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("analyzed_at must be timezone-aware")
        return v


# ---------------------------------------------------------------------------
# 13. AuditEvent
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    action: str  # e.g. 'intake', 'pii_redaction', 'llm_analysis'
    details: str
    timestamp: datetime = Field(default_factory=_utcnow)
    actor: str = "pipeline"


# ---------------------------------------------------------------------------
# 14. TranscriptionCacheEntry
# ---------------------------------------------------------------------------


class TranscriptionCacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    sha256_hash: str
    transcription_json: str  # JSON-serialised TranscriptionResult
    created_at: datetime = Field(default_factory=_utcnow)
    hit_count: int = 0

    @field_validator("transcription_json")
    @classmethod
    def _validate_json(cls, v: str) -> str:
        try:
            json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"transcription_json is not valid JSON: {e}") from e
        return v
