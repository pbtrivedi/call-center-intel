# Data Schemas

> To be filled in as models are implemented.

## 14 Pydantic Contracts

| Model | Stage | Purpose |
|-------|-------|---------|
| `AudioInput` | Intake in | Raw audio bytes, filename, optional metadata |
| `IntakeResult` | Intake out | Validation outcome, audio properties, PII scan |
| `TranscriptionResult` | Transcription out | Segments, speaker labels, timestamps, confidence |
| `InjectionCheckResult` | Injection detection out | Match status, matched pattern names |
| `RedactedTranscript` | PII redaction out | Cleaned full_text and segments |
| `SummaryResult` | Summarization out | Purpose, key points, action items, sentiment |
| `QAScoreResult` | QA scoring out | Dimension scores, compliance flags, overall score |
| `CallReport` | Report out | Assembled final artifact |
| `ComplianceFlag` | Embedded in QAScoreResult | Violation, severity, timestamp reference |
| `AuditEvent` | Database | Timestamped pipeline event |
| `TranscriptionCacheEntry` | Database | SHA-256 hash → TranscriptionResult |
| `ActionItem` | Embedded in SummaryResult | Owner, description, optional deadline |
| `QADimension` | Embedded in QAScoreResult | Score, justification, dimension name |
| `AudioProperties` | Embedded in IntakeResult | Duration, sample rate, channels |
