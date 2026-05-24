class CallCenterIntelError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context: dict = context or {}


class AudioValidationError(CallCenterIntelError):
    """Raised by intake agent for bad format, size, or duration."""


class TranscriptionError(CallCenterIntelError):
    """Raised when faster-whisper transcription fails."""


class InjectionDetectedError(CallCenterIntelError):
    """Raised when prompt injection patterns match in a transcript."""


class PIIRedactionError(CallCenterIntelError):
    """Raised when PII redaction fails."""


class LLMAnalysisError(CallCenterIntelError):
    """Raised by summarization or QA scoring agents on LLM failure."""


class ReportGenerationError(CallCenterIntelError):
    """Raised by report agent or PDF generator."""


class PipelineError(CallCenterIntelError):
    """Generic wrapper for unexpected pipeline failures."""
