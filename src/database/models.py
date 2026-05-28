from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CallRecord(Base):
    """One row per analyzed call. Upserted by report_agent on every pipeline run."""

    __tablename__ = "call_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    overall_qa_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_format: Mapped[str | None] = mapped_column(String(8), nullable=True)
    sha256_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"CallRecord(call_id={self.call_id!r}, status={self.status!r})"


class TranscriptionCache(Base):
    """SHA-256-keyed cache of transcription results. INSERT-only — never updated."""

    __tablename__ = "transcription_cache"

    sha256_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    transcription_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return f"TranscriptionCache(sha256={self.sha256_hash[:8]}..., hits={self.hit_count})"


class AuditLog(Base):
    """INSERT-only audit log. No UPDATE or DELETE paths exist for this table."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"AuditLog(id={self.id}, call_id={self.call_id!r}, action={self.action!r})"
