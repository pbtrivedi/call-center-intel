# Call Center Intelligence System — Specification

## Overview

A production-grade, multi-agent AI pipeline that processes raw call center audio and produces structured transcripts, quality scores, compliance reports, and downloadable artifacts. Built as a Gradio web application backed by a LangGraph state machine.

**Entry point:** `python app.py` → accessible at `http://localhost:7860`

---

## Architecture: Five-Layer Structure

```
src/
├── ui/           # Gradio interface components
├── services/     # Cross-cutting: LLM factory, audio utils, PDF generation
├── agents/       # Seven pipeline stage implementations
├── graph/        # LangGraph state machine wiring and routing
└── database/     # SQLAlchemy ORM models, session factory, repositories
app.py            # ≤50 lines; loads Whisper singleton, launches Gradio
```

---

## Pipeline: Seven Sequential Stages

Each stage receives a typed `PipelineState` TypedDict and returns a partial update. Failures are isolated — one node failure routes to `error` without crashing others.

| # | Stage | Responsibility | Key Detail |
|---|-------|---------------|------------|
| 1 | **Intake** | Validate format, size, duration; scan metadata for PII | Magic-byte detection (first 12 bytes), not file extension |
| 2 | **Transcription** | Speech-to-text with speaker diarization | SHA-256 cache; identical audio returns instantly |
| 3 | **Injection Detection** | Scan transcript for 22+ regex patterns before any LLM call | Blocks malicious audio from reaching LLM entirely |
| 4 | **PII Redaction** | Replace SSN, credit card, email, phone with labeled placeholders | Applied to full text AND every individual segment |
| 5 | **Summarization** | Extract purpose, key points, action items, sentiment, entities | Structured LLM output validated by Pydantic |
| 6 | **QA Scoring** | Score agent on 5 weighted dimensions (1–5 each) | Overall score recomputed deterministically; LLM score discarded |
| 7 | **Report** | Compile PDF + JSON, persist to SQLite, write audit log | SHA-256 cache updated for future calls |

### Terminal Nodes
- `report` — standard completion
- `supervisor_review` — triggered when a `critical` compliance flag is found
- `error` — intake failure, injection detected, or unhandled exception

### Conditional Routing
- `intake` → `transcribe` or `error`
- `injection_check` → `pii_redaction` or `error`
- `qa_scoring` → `report` or `supervisor_review` (if critical flag)

---

## LangGraph State Machine

```python
class PipelineState(TypedDict):
    audio_input: AudioInput
    intake_result: IntakeResult
    transcription_result: TranscriptionResult
    injection_check_result: InjectionCheckResult
    redacted_transcript: RedactedTranscript
    summary_result: SummaryResult
    qa_score_result: QAScoreResult
    call_report: CallReport
    error: str | None
    routing_decision: str
```

Compiled via `workflow.compile()` and invoked with a single `workflow.invoke(state)` call.

---

## Audio Intake & Validation

- **Format detection:** magic bytes (first 12 bytes), not extension
- **Supported formats:** WAV, MP3, FLAC, M4A
- **Limits:** max 50 MB file size; max 60 minutes duration
- **WAV duration:** read from RIFF header before size gate (so oversized WAV returns "duration exceeds" not "file too large")
- **Metadata PII scan:** caller ID and department fields scanned before storage
- **Output:** validated audio written to temp file; path passed downstream

---

## Speech-to-Text Transcription

- **Engine:** `faster-whisper` with CTranslate2 int8 quantization
- **Model loading:** once at startup via `_get_whisper_model()` module-level singleton; never inside a request handler
- **Device:** auto-detect CUDA → CPU fallback; appropriate compute type per device
- **Settings:** `beam_size=1`, VAD filter enabled, `condition_on_previous_text=False`
- **Diarization:** heuristic (gap-based + content-pattern matching) → labels: `Agent` / `Customer`
- **Artifact cleaning:** remove `BLANK_AUDIO` tags, repeated phrases, YouTube footers, non-speech labels
- **Confidence:** per-segment score from `avg_logprob` + `no_speech_prob`
- **Caching:** SHA-256 hash of audio bytes → `TranscriptionCache` table; cache hit skips Whisper entirely
- **Development model:** `tiny` or `base`; `large-v3` for GPU/CUDA deployments only

---

## Security Layer

### Prompt Injection Detector
- 22+ regex patterns covering: ignore-previous-instructions, role switching, DAN mode, system tag injection, prompt leakage, jailbreak phrases, conversation injection
- Runs on full transcript text before any LLM call
- On match: routes to `error` node; returns matched pattern names

### PII Redactor
- Targets: SSN (`[REDACTED_SSN]`), credit card (`[REDACTED_CREDIT_CARD]`), email (`[REDACTED_EMAIL]`), phone (`[REDACTED_PHONE]`)
- Strategy: collect all matches first, replace right-to-left to preserve position offsets
- Scope: applied to `full_text` and every individual `segment` before LLM sees any text

### Audit Logger
- Append-only table `audit_log` in SQLite
- Records: `timestamp`, `call_id`, `action` (started / completed / failed / flagged), `details`
- Records are never deleted or modified

---

## QA Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Professionalism | 15% | Language quality, greeting/closing, composure, no interruptions |
| Empathy | 20% | Active listening, acknowledging feelings, rapport, personalized responses |
| Problem Resolution | 30% | Root cause ID, solution quality, customer confirmation of understanding |
| Compliance | 20% | Required disclosures, identity verification, hold procedures, data safety |
| Communication Clarity | 15% | Clear explanations, minimal jargon, structured delivery, confirmed comprehension |

**Critical:** `overall_score = Σ(dimension_score × weight)` — computed in Python after LLM response. LLM's `overall_score` is always discarded.

---

## Pydantic Data Models (14 total)

| Model | Purpose |
|-------|---------|
| `AudioInput` | Raw audio bytes, filename, optional metadata |
| `IntakeResult` | Validation outcome, audio properties, PII scan results |
| `TranscriptionResult` | Segments with speaker labels, timestamps, confidence, flagging status |
| `InjectionCheckResult` | Match status, matched pattern names |
| `RedactedTranscript` | Cleaned full_text and segments with PII replaced |
| `SummaryResult` | Call purpose, key points, action items, resolution status, sentiment, entities |
| `QAScoreResult` | Five dimension scores + justifications, compliance flags, weighted overall score |
| `CallReport` | Assembled final artifact with processing metadata |
| `ComplianceFlag` | Violation description, severity (low/medium/high/critical), transcript timestamp |
| `AuditEvent` | Timestamped pipeline event record |
| `TranscriptionCacheEntry` | SHA-256 hash → serialized TranscriptionResult |
| `ActionItem` | Owner, description, optional deadline |
| `QADimension` | Score (1–5), justification, dimension name |
| `AudioProperties` | Duration, sample rate, channels |

---

## LLM Integration

### Provider Factory
Switch active provider via single env var `LLM_PROVIDER`. No code changes required.

| Provider | Model | Cost | Limit |
|----------|-------|------|-------|
| OpenAI | GPT-4o | ~$0.03/call | Pay-per-use |
| Gemini | 2.0 Flash | Free | 1,500 req/day |
| Groq | Llama 3.3 70B | Free | 30 RPM |

### Summarization Agent
- Fields: `call_purpose`, `key_discussion_points` (3–7), `action_items` (owner + optional deadline), `resolution_status` (resolved/unresolved/escalated), `sentiment_trajectory`, `named_entities`
- Retry: exponential backoff, up to 3 attempts

### QA Scoring Agent
- Receives summarization result as context
- Generates justifications with transcript timestamp references
- Identifies compliance flags with severity levels
- Overall score recomputed deterministically after response

---

## Report Generation & Persistence

### SQLite Schema (SQLAlchemy ORM)
```
call_records       — transcript, summary JSON, QA JSON, report JSON, timestamps
transcription_cache — SHA-256 hash → serialized TranscriptionResult
audit_log          — append-only pipeline event records
```

- Connection pooling via cached `sessionmaker` factory; never raw connection per request
- DB file: `data/calls.db` (created at runtime; excluded from git)

### Downloadable Artifacts
- **PDF:** generated via ReportLab (summary, QA scorecard, compliance flags)
- **JSON:** Pydantic model serialization of `CallReport`

---

## Web Interface (Gradio 5.x)

### Tab 1: Analyze Call
- Audio upload (file or microphone recording)
- Optional: caller ID, department fields
- Analyze button with processing status message (includes estimated duration + do-not-refresh warning)
- Output: speaker-labeled transcript, formatted summary markdown, QA scorecard markdown
- PDF and JSON download buttons

### Tab 2: All MP3 History
- Master-detail browser over all analyzed calls
- Searchable/filterable call list

### Tab 3: Observability
- Auto-refreshes when tab is selected + manual refresh button
- Metrics: total calls, completed/failed/flagged counts, success rate %, average QA score, total compliance flags
- Table: 20 most recent audit events (timestamp, call ID, action, details)
- LangSmith integration status (enabled/disabled) with project link when active

---

## Testing Requirements

**Minimum: 100+ tests**, all passing with `pytest tests/ -v`

```
tests/
├── unit/          # Agent functions, security functions, routing logic, Pydantic models, formatters
├── integration/   # End-to-end pipeline (mocked LLM + Whisper), DB persistence, audit log
└── security/      # PII format variants, 22+ injection patterns, edge cases (PII in conversation text)
```

### Key Mocking Requirements
- LLM calls must be mocked in integration tests (no real API calls)
- Whisper model must be mocked in integration tests
- Security tests use real pattern matching (no mocks)

---

## Deployment & Configuration

### Environment Variables (all in `.env.example`)
```bash
LLM_PROVIDER=openai          # openai | gemini | groq
OPENAI_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
LANGSMITH_API_KEY=           # optional
LANGSMITH_PROJECT=           # optional
MAX_FILE_SIZE_MB=50
MAX_DURATION_MINUTES=60
WHISPER_MODEL=base           # tiny | base | large-v3
```

### Makefile Targets
`install`, `test`, `test-integration`, `test-all`, `lint`, `format`, `run`, `clean`

### Dockerfile
- Builds and runs without manual steps from a clean clone
- GPU auto-detection: CUDA when available, CPU int8 fallback

### .gitignore Exclusions
```
.env
data/audio/
data/*.db
```

---

## Dataset

Use Kaggle [`louisteitelbaum/911-recordings`](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings) for test audio files.

---

## Key Constraints & Gotchas

| Constraint | Why It Matters |
|------------|---------------|
| Magic-byte format detection | File extension can be spoofed; `.wav` containing MP3 data must be detected correctly |
| PII redacted before any LLM call | GDPR/CCPA — raw customer data must never reach LLM |
| QA overall score recomputed in Python | LLM scores are non-deterministic; deterministic formula required for audit |
| Whisper loaded once at startup | Model loading = 5–30s; loading per request is a fatal performance issue |
| SHA-256 transcription cache | At 5,000 calls/day, re-processing identical audio is prohibitively expensive |
| `condition_on_previous_text=False` | Prevents Whisper hallucination loops |
| Injection detection before LLM | Adversarial audio can manipulate LLM behavior if not blocked first |
| Temp file cleanup | Disk fills after hundreds of calls; implement rolling cleanup |
| No hardcoded provider/model/key | All must come from env vars for zero-code provider switching |
| Error node three-level fallback | `state['error']` → `intake.validation_error` → generic default |

---

## Evaluation Weights

| Criterion | Weight |
|-----------|--------|
| Pipeline Architecture & Orchestration | 20% |
| LLM Integration & Prompt Engineering | 20% |
| Audio Processing & Transcription | 15% |
| Security Layer | 15% |
| Report Generation & Persistence | 10% |
| Testing | 10% |
| Code Quality & Deployment | 10% |
