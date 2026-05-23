# Architecture

## System Overview

Call Center Intelligence System is a production-grade, multi-agent AI pipeline that processes raw call center audio and produces structured transcripts, quality scores, compliance reports, and downloadable artifacts.

```
Audio Upload → 7-Stage LangGraph Pipeline → PDF/JSON Reports + SQLite persistence
```

The application is a Gradio 5.x web app (`python app.py` → `http://localhost:7860`) backed by a LangGraph state machine. All processing is sequential with shared typed state; no fan-out.

---

## Repository Layout

```
call-center-intel/
├── app.py                    # Entry point ≤50 lines; Whisper singleton + Gradio launch
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── src/
│   ├── agents/               # Seven pipeline stage implementations
│   │   ├── intake_agent.py
│   │   ├── transcription_agent.py
│   │   ├── summarization_agent.py
│   │   ├── qa_scoring_agent.py
│   │   └── report_agent.py
│   ├── security/             # Injection detector, PII redactor, audit logger
│   │   ├── injection_detector.py
│   │   ├── pii_redactor.py
│   │   └── audit_logger.py
│   ├── graph/                # LangGraph wiring: state, nodes, routing edges
│   │   ├── state.py          # PipelineState TypedDict
│   │   ├── pipeline.py       # StateGraph definition + compile()
│   │   └── routing.py        # Conditional edge functions
│   ├── services/             # Cross-cutting, framework-agnostic utilities
│   │   ├── llm_factory.py    # Multi-provider LangChain factory
│   │   ├── whisper_model.py  # Singleton loader + get_whisper_model()
│   │   ├── audio_utils.py    # Magic-byte detection, duration, temp file cleanup
│   │   └── pdf_generator.py  # ReportLab PDF builder
│   ├── database/             # Persistence layer
│   │   ├── models.py         # SQLAlchemy ORM table definitions
│   │   ├── repository.py     # CRUD operations + audit log writer
│   │   └── database.py       # Cached sessionmaker factory
│   ├── ui/                   # Gradio tabs
│   │   ├── interface.py      # build_interface() — assembles all tabs
│   │   ├── analyze_tab.py
│   │   ├── history_tab.py
│   │   └── observability_tab.py
│   ├── models/               # 14 Pydantic contracts (stage I/O types)
│   │   └── schemas.py
│   ├── config/               # YAML configs + env loader
│   │   ├── loader.py
│   │   └── settings.yaml
│   └── common/               # Shared infrastructure (no domain logic)
│       ├── logger.py         # get_logger(name) factory → logs/app.log
│       └── exceptions.py     # CallCenterIntelError hierarchy
│
├── tests/
│   ├── conftest.py           # Shared fixtures (mock LLM, mock Whisper, test DB)
│   ├── unit/                 # Agent functions, security, routing, models, formatters
│   ├── integration/          # End-to-end pipeline, DB persistence
│   └── security/             # PII format variants, 22+ injection adversarial payloads
│
├── docs/
│   ├── ARCHITECTURE.md       # This file
│   ├── DATA_SCHEMAS.md       # 14 Pydantic model field reference
│   ├── ENV_CONFIGURATION.md  # Environment variable reference
│   ├── DECISIONS.md          # Architecture Decision Records
│   └── QUICK_START.md        # Setup and run guide
│
├── infrastructure/
│   └── docker/               # Docker configs (cloud/CDK stacks added later)
│
├── data/
│   └── audio/                # Runtime uploads — gitignored
└── logs/                     # Runtime logs — gitignored
```

---

## Pipeline: Seven Sequential Stages

Each stage is a standalone function that receives the full `PipelineState` and returns a partial dict update. No stage holds shared mutable state. A failure at any stage is caught, written to `state['error']`, and routed to the `error` terminal node — the rest of the pipeline does not run.

```
┌─────────┐    ┌───────────────┐    ┌──────────────────┐    ┌───────────────┐
│  Intake  │───▶│ Transcription │───▶│ Injection Check  │───▶│ PII Redaction │
└─────────┘    └───────────────┘    └──────────────────┘    └───────────────┘
     │                                        │                       │
   error                                    error                     ▼
                                                            ┌──────────────────┐
                                                            │  Summarization   │
                                                            └──────────────────┘
                                                                      │
                                                                      ▼
                                                            ┌──────────────────┐
                                                            │   QA Scoring     │
                                                            └──────────────────┘
                                                                      │
                                                         ┌────────────┴────────────┐
                                                         ▼                         ▼
                                                      report            supervisor_review
                                                   (standard)            (critical flag)
```

| # | Stage | Input | Output | Key Constraint |
|---|-------|-------|--------|---------------|
| 1 | **Intake** | `AudioInput` | `IntakeResult` | Format by magic bytes, not extension; WAV duration from RIFF header |
| 2 | **Transcription** | temp file path | `TranscriptionResult` | SHA-256 cache check first; Whisper singleton never loaded per-request |
| 3 | **Injection Check** | `TranscriptionResult.full_text` | `InjectionCheckResult` | 22+ patterns; runs before any LLM call |
| 4 | **PII Redaction** | `TranscriptionResult` | `RedactedTranscript` | Applied to full_text AND every segment; replace right-to-left |
| 5 | **Summarization** | `RedactedTranscript` | `SummaryResult` | Structured LLM output → Pydantic; exponential backoff, 3 retries |
| 6 | **QA Scoring** | `SummaryResult` + `RedactedTranscript` | `QAScoreResult` | LLM score discarded; overall recomputed from weighted formula |
| 7 | **Report** | all upstream results | `CallReport` | Persist to SQLite, write audit log, generate PDF + JSON |

### Conditional Routing

```python
# graph/routing.py
def route_after_intake(state) -> str:
    return "error" if state.get("error") else "transcription"

def route_after_injection_check(state) -> str:
    return "error" if state["injection_check_result"].matched else "pii_redaction"

def route_after_qa_scoring(state) -> str:
    flags = state["qa_score_result"].compliance_flags
    return "supervisor_review" if any(f.severity == "critical" for f in flags) else "report"
```

### Error Node Fallback Chain

The `error_step` node resolves the displayed message via a three-level fallback:
1. `state['error']` — set by any stage on exception
2. `state['intake_result'].validation_error` — set by intake on format/size/duration rejection
3. Generic default: `"Pipeline failed — check logs for details"`

---

## LangGraph State

```python
# graph/state.py
class PipelineState(TypedDict):
    audio_input:            AudioInput
    intake_result:          IntakeResult
    transcription_result:   TranscriptionResult
    injection_check_result: InjectionCheckResult
    redacted_transcript:    RedactedTranscript
    summary_result:         SummaryResult
    qa_score_result:        QAScoreResult
    call_report:            CallReport
    error:                  str | None
    routing_decision:       str
```

The compiled graph is invoked once per request: `workflow.invoke(initial_state)`. The `workflow` object is a module-level singleton in `graph/pipeline.py`, compiled at import time.

---

## Five-Layer Responsibilities

### `agents/` — Pipeline Stage Implementations
Each agent is a pure function: `def run(state: PipelineState) -> dict`. No agent imports from another agent. Each has its own error handling that sets `state['error']` before returning.

### `security/` — Security Components
Three components kept separate from agents because they are independently testable and reusable:
- **`injection_detector.py`** — compiled regex bank, returns match list
- **`pii_redactor.py`** — stateless redaction; collects matches first, replaces right-to-left
- **`audit_logger.py`** — writes to `audit_log` table; never reads or deletes

### `graph/` — Orchestration
Owns the `StateGraph` definition, all `add_node` / `add_conditional_edges` calls, and the compiled `workflow`. Nothing outside `graph/` should import LangGraph.

### `services/` — Cross-Cutting Utilities
Framework-agnostic. No LangGraph, no Gradio, no SQLAlchemy imports here. Contains:
- `llm_factory.py` — returns a `BaseChatModel` based on `LLM_PROVIDER` env var
- `whisper_model.py` — module-level singleton; `get_whisper_model()` is called once in `app.py`
- `audio_utils.py` — magic-byte detection, `mutagen`-based metadata, temp file cleanup
- `pdf_generator.py` — ReportLab builder; takes a `CallReport` and returns bytes

### `database/` — Persistence
Three SQLAlchemy ORM tables:

```
call_records        id, call_id, audio_filename, transcript_text,
                    summary_json, qa_json, report_json, status,
                    created_at, updated_at

transcription_cache sha256_hash (PK), transcription_json, created_at

audit_log           id, call_id, action, details, created_at
                    (INSERT only — no UPDATE or DELETE)
```

`database.py` exports a cached `get_session()` factory. No code outside `database/` opens a raw SQLAlchemy connection.

### `ui/` — Gradio Interface
Three tabs assembled by `build_interface()` in `interface.py`:
- **Analyze Call** — audio upload/record, optional metadata, Analyze button, results display, PDF/JSON download
- **All MP3 History** — master-detail browser; loads from `call_records`
- **Observability** — auto-refresh on tab select; metrics + 20 most recent audit events + LangSmith status

### `models/` — Pydantic Contracts
All 14 stage I/O types live in `schemas.py`. This is the only place where stage contracts are defined. See [DATA_SCHEMAS.md](DATA_SCHEMAS.md) for field-level detail.

### `config/` — Configuration
`loader.py` reads `.env` via `python-dotenv` and merges with `settings.yaml` defaults. Every configurable value (model name, limits, timeouts, provider) is accessed through here — never via `os.environ` directly in agent code.

---

## Security Design

### Why Injection Detection Runs Before PII Redaction
Injection patterns must be checked against the raw transcript because an attacker could craft instructions that only appear after PII is removed. Running injection detection on the pre-redacted text catches all possible patterns.

### PII Redaction Strategy
Right-to-left replacement preserves character offsets when multiple PII items appear in the same text. Collecting all matches first (before replacing any) prevents one replacement from shifting the position of the next match.

### Audit Log Immutability
`audit_log` has no `UPDATE` or `DELETE` paths in `repository.py`. The SQLAlchemy model has no `updated_at` column. This is enforced in code, not just by convention.

---

## LLM Provider Factory

```python
# services/llm_factory.py
def get_llm() -> BaseChatModel:
    provider = os.environ["LLM_PROVIDER"]   # openai | gemini | groq
    match provider:
        case "openai":  return ChatOpenAI(model="gpt-4o", ...)
        case "gemini":  return ChatGoogleGenerativeAI(model="gemini-2.0-flash", ...)
        case "groq":    return ChatGroq(model="llama-3.3-70b-versatile", ...)
```

Model name, API key, and timeout all come from environment variables. Switching providers requires changing `LLM_PROVIDER` only — no code changes.

| Provider | Model | Tier | Limit |
|----------|-------|------|-------|
| OpenAI | GPT-4o | Paid | ~$0.03/call |
| Gemini | 2.0 Flash | Free | 1,500 req/day |
| Groq | Llama 3.3 70B | Free | 30 RPM |

---

## Transcription Caching

```
Request → SHA-256(audio bytes) → lookup TranscriptionCache table
                                        │
                           ┌────────────┴────────────┐
                           │ cache hit                │ cache miss
                           ▼                          ▼
                  return cached result        run faster-whisper
                                              write result to cache
```

At 5,000 calls/day with any repeat audio (hold music, IVR prompts, test recordings), the cache eliminates the majority of Whisper compute. Cache entries are never invalidated — audio content is deterministic given the same bytes.

---

## Whisper Singleton

```python
# services/whisper_model.py
_model = None

def get_whisper_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _model = WhisperModel(os.environ.get("WHISPER_MODEL", "base"),
                              device=device, compute_type=compute_type)
    return _model
```

`app.py` calls `get_whisper_model()` once at startup. Gradio request handlers call it again — the singleton returns instantly. Loading inside a request handler would add 5–30 seconds to every call.

**Model selection:**
- `tiny` / `base` — development and CPU deployments
- `large-v3` — GPU (CUDA) deployments only; do not use on CPU (25+ min/call)

---

## QA Scoring Formula

The LLM is asked to score five dimensions (1–5 each). After the response arrives, Python discards `overall_score` from the LLM output and recomputes it deterministically:

```python
overall_score = (
    professionalism      * 0.15 +
    empathy              * 0.20 +
    problem_resolution   * 0.30 +
    compliance           * 0.20 +
    communication_clarity * 0.15
)
```

This is a hard guardrail: LLM scores are non-deterministic and cannot survive regulatory audit. The weighted formula is reproducible given the same dimension scores.

---

## Web Interface Layout

```
┌─────────────────────────────────────────────────────┐
│  Call Center Intelligence System                     │
├──────────────┬──────────────────┬───────────────────┤
│ Analyze Call │  All MP3 History │  Observability    │
├──────────────┴──────────────────┴───────────────────┤
│ [Tab 1 - Analyze Call]                               │
│  Audio Upload / Microphone                           │
│  Caller ID (optional)   Department (optional)        │
│  [ Analyze ]                                         │
│  ─────────────────────────────────────────────────  │
│  Transcript (speaker-labeled)                        │
│  Summary (markdown)                                  │
│  QA Scorecard (markdown)                             │
│  [ Download PDF ]  [ Download JSON ]                 │
└─────────────────────────────────────────────────────┘
```

---

## Testing Strategy

```
tests/
├── unit/         Pure functions — no I/O, no network, no DB
├── integration/  Full pipeline with mocked LLM + mocked Whisper; real SQLite in-memory
└── security/     Real regex matching — no mocks; adversarial PII and injection payloads
```

All three suites are independent: `make test-unit`, `make test-integration`, `make test-security` each run in isolation. `make test-all` runs all three. LLM and Whisper are always mocked outside the `security/` suite to keep tests fast and free.

---

## Startup Sequence

```
python app.py
  1. load_dotenv()                          # read .env
  2. get_whisper_model()                    # load Whisper once; 5-30s on first run
  3. build_interface()                      # assemble Gradio tabs
  4. app.launch(server_port=7860)           # start HTTP server
```

On first request, `database.py` creates `data/calls.db` and all tables if they don't exist (SQLAlchemy `create_all`).
