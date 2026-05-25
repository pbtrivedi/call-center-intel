# Development Phases

Nine iterations, each delivering a working, testable slice of the system. Build them in order — each phase depends on the previous one being solid before moving forward.

> **Rule:** Do not start the next phase until every item in the current phase's Definition of Done passes.

---

## Phase 1 — Foundation: Data Contracts, Config, Logger & Exceptions

**Goal:** Establish the typed backbone that every other phase depends on. After this phase, all 14 Pydantic models exist and are validated, the config system reads from `.env` and `settings.yaml`, a structured application logger is wired up, and all custom exception types are defined. Every agent from Phase 2 onwards will import from this layer.

> **Note on two different loggers:** This phase builds the *application logger* (structured log output to `logs/` files). The *audit logger* (compliance event log written to SQLite) is a separate component built in Phase 3 as part of the security layer — it has different concerns (append-only DB writes, surfaced in the Observability tab).

### What to Build
- `src/models/schemas.py` — all 14 Pydantic v2 models:
  `AudioInput`, `IntakeResult`, `AudioProperties`, `TranscriptionResult`, `InjectionCheckResult`, `RedactedTranscript`, `SummaryResult`, `ActionItem`, `QAScoreResult`, `QADimension`, `ComplianceFlag`, `CallReport`, `AuditEvent`, `TranscriptionCacheEntry`
- `src/config/loader.py` — reads `.env` + `settings.yaml`; exposes typed `Settings` object
- `src/config/settings.yaml` — default values for all optional config keys
- `src/common/logger.py` — application logger
  - Structured log output (JSON or key=value) to `logs/app.log` and stdout
  - `get_logger(name) -> Logger` factory used by every module
  - Log level driven by `LOG_LEVEL` env var (default `INFO`)
  - Rotating file handler with configurable max size
- `src/common/exceptions.py` — custom exception hierarchy:
  - `CallCenterIntelError` — base exception for all pipeline errors
  - `AudioValidationError` — raised by intake agent (bad format, size, duration)
  - `TranscriptionError` — raised by transcription agent
  - `InjectionDetectedError` — raised when injection patterns match
  - `PIIRedactionError` — raised on redaction failure
  - `LLMAnalysisError` — raised by summarization or QA scoring agents
  - `ReportGenerationError` — raised by report agent or PDF generator
  - `PipelineError` — generic wrapper for unexpected pipeline failures

### Tests to Write (`tests/unit/`)
- Valid construction of every Pydantic model with required fields
- Validation errors raised for invalid field values (e.g. score outside 1–5, unsupported severity)
- `ComplianceFlag.severity` rejects values outside `low | medium | high | critical`
- Config loader returns correct defaults when env vars are absent
- Config loader overrides defaults when env vars are set
- `get_logger()` returns a logger with the correct name and respects `LOG_LEVEL`
- All custom exceptions are subclasses of `CallCenterIntelError`
- Each exception carries the expected message and optional context fields

### Definition of Done
- [x] All 14 models importable and instantiable
- [x] `pip install -e .` succeeds from a clean clone
- [x] `get_logger(__name__)` callable from any module; writes to `logs/app.log`
- [x] All 8 custom exception classes defined and importable (`ConfigurationError` added as 9th)
- [x] `make test-unit` passes with tests covering all models, logger, and exceptions (118 tests)
- [x] No hardcoded strings or magic values outside `schemas.py` or `settings.yaml`

**Completed:** 2026-05-24 — PR #1 merged (`f7b4699`). Second review follow-up (`a8a0eb1`) added `ConfigurationError`, `frozen=True` on all schemas, cross-field validators, sha256/confidence/speaker/JSON validators, thread-safe `get_settings()`, and env-var error handling.

---

## Phase 2 — Audio Processing: Intake + Transcription + Cache

**Goal:** Given a raw audio file, produce a speaker-labeled transcript. This is the first phase that processes real input and produces real output — a working transcription pipeline end to end.

### What to Build
- `src/agents/intake_agent.py`
  - Magic-byte format detection (WAV/MP3/FLAC/M4A); reject everything else
  - File size gate (50 MB); WAV duration from RIFF header; non-WAV duration via `mutagen`
  - Metadata PII pattern scan (caller ID, department fields)
  - Write validated audio to temp file; return `IntakeResult`
- `src/services/audio_utils.py`
  - `detect_format(bytes) -> str` — magic-byte lookup
  - `get_duration(path) -> float` — RIFF header for WAV, mutagen for others
  - `cleanup_temp_files(dir, max_files)` — rolling cleanup
- `src/services/whisper_model.py`
  - `get_whisper_model()` singleton; auto-detect CUDA → CPU; set compute type
- `src/agents/transcription_agent.py`
  - SHA-256 hash audio bytes → check `TranscriptionCache` table (stub OK in this phase)
  - Run `faster-whisper` with `beam_size=1`, VAD, `condition_on_previous_text=False`
  - Heuristic diarization: gap-based + content-pattern → `Agent` / `Customer` labels
  - Clean artifacts: `BLANK_AUDIO`, repeated phrases, YouTube footers
  - Per-segment confidence from `avg_logprob` + `no_speech_prob`
  - Return `TranscriptionResult`

### Tests to Write (`tests/unit/`)
- Magic-byte detection: WAV, MP3, FLAC, M4A — all return correct format
- Magic-byte detection: `.wav` file containing MP3 bytes → returns `mp3`, not `wav`
- File rejected if size > 50 MB
- WAV file rejected if duration > 60 minutes (RIFF header path)
- Non-WAV file rejected if duration > 60 minutes (mutagen path)
- Unsupported format (OGG, TXT renamed to WAV) → clear error with supported formats listed
- Transcript artifact cleaning: `BLANK_AUDIO` removed, repeated phrases collapsed
- Confidence scoring: segment with high `no_speech_prob` gets low confidence score
- SHA-256 of identical bytes always produces the same hash

### Definition of Done
- [x] Upload a real MP3 → get back a `TranscriptionResult` with speaker-labeled segments
- [x] Same file uploaded twice → second call returns faster (cache path taken)
- [x] Unsupported file format returns a clear rejection message
- [x] `make test-unit` passes all intake and transcription tests (190 tests)
- [x] Whisper model is NOT loaded during test runs (mocked)

**Completed:** 2026-05-24 — PR #2 merged (`3551e7a`). Includes diarization fix (removed bare "I'm calling" from `_CUSTOMER_RE`; added outbound agent patterns) and hallucination suppression (`_MIN_CONFIDENCE=0.05` confidence gate). Tested with real Kaggle call-center audio.

---

## Phase 3 — Security Layer: Injection Detection + PII Redaction + Audit Log

**Goal:** Make the pipeline safe to connect to an LLM. After this phase, no raw PII ever reaches an LLM call, adversarial transcripts are blocked before analysis begins, and every pipeline event is permanently logged.

### What to Build
- `src/security/injection_detector.py`
  - Compiled `re` pattern bank with 22+ patterns covering:
    ignore-previous-instructions, role switching, DAN mode, system tag injection,
    prompt leakage requests, jailbreak phrases, conversation injection
  - `detect_injection(text) -> InjectionCheckResult`
- `src/security/pii_redactor.py`
  - Patterns for: SSN, credit card (major formats), email, phone (US and international)
  - Collect all matches first, replace right-to-left (preserves offsets)
  - Apply to `full_text` and every individual segment
  - `redact(transcript: TranscriptionResult) -> RedactedTranscript`
- `src/security/audit_logger.py`
  - `log_event(call_id, action, details)` — INSERT only; no UPDATE/DELETE paths
  - Stubs to real DB write (DB layer comes in Phase 6; use a no-op stub here)
- `src/database/models.py` — `AuditLog` ORM table stub (columns only, no session yet)

### Tests to Write (`tests/security/`)
- All 22+ injection patterns detected individually with purpose-built adversarial strings
- Clean transcript passes injection check without false positives
- SSN redacted in multiple formats: `123-45-6789`, `123 45 6789`, `123456789`
- Credit card redacted: Visa, Mastercard, Amex formats; with and without spaces/dashes
- Email redacted: standard, subdomain, plus-addressing formats
- Phone redacted: `(555) 123-4567`, `555-123-4567`, `+1 555 123 4567`, international
- PII embedded mid-sentence: `"my card is 4111111111111111 and my email is..."` → both redacted
- Redaction applied to individual segments, not just `full_text`
- `InjectionCheckResult.matched = False` for clean transcript

### Definition of Done
- [ ] Transcript containing a credit card number → LLM never sees the raw digits
- [ ] Transcript with `"Ignore all previous instructions"` → `InjectionCheckResult.matched = True`
- [ ] `make test-security` passes all adversarial payload tests
- [ ] All PII format variants covered in security suite
- [ ] Audit logger `log_event()` callable without errors (stub DB write is fine)

---

## Phase 4 — LLM Analysis: Summarization + QA Scoring + Provider Factory + Component Eval

**Goal:** Given a redacted transcript, produce a structured summary and deterministic QA scorecard. After this phase, the two LLM analysis stages work independently, can be called with any of the three providers, and their *output quality* has been validated against real transcripts — not just functional correctness.

> **Why evaluation starts here, not later:** Prompts and structured output schemas are defined in this phase. Catching quality problems now (hallucinated summaries, poor QA justifications, miscalibrated scores) is far cheaper than discovering them after the full pipeline is wired. Evaluation is a feedback loop on prompt engineering, not just a final QA step.

### What to Build
- `src/services/llm_factory.py`
  - `get_llm() -> BaseChatModel` — reads `LLM_PROVIDER` env var
  - Supports `openai` (GPT-4o), `gemini` (2.0 Flash), `groq` (Llama 3.3 70B)
  - Model name, API key, and timeout all from env vars — nothing hardcoded
- `src/agents/summarization_agent.py`
  - Structured output prompt → `SummaryResult` via Pydantic `.with_structured_output()`
  - Fields: `call_purpose`, `key_discussion_points` (3–7), `action_items` (owner + optional deadline), `resolution_status`, `sentiment_trajectory`, `named_entities`
  - Exponential backoff retry, up to 3 attempts
- `src/agents/qa_scoring_agent.py`
  - Receives `RedactedTranscript` + `SummaryResult` as context
  - Structured output → five `QADimension` scores (1–5) with justifications and transcript timestamps
  - Compliance flags with severity (`low | medium / high | critical`)
  - **After LLM responds:** discard `overall_score` from LLM; recompute deterministically:
    `overall = professionalism×0.15 + empathy×0.20 + problem_resolution×0.30 + compliance×0.20 + clarity×0.15`
- `evals/fixtures/` — 3–5 hand-labeled transcripts with expected outputs
  - Each fixture: a redacted transcript + expected `SummaryResult` fields + expected QA score ranges
  - Source: Kaggle 911-recordings dataset; label manually before running evals
- `evals/eval_summarization.py` — component-level evaluation script
  - Run each fixture through the summarization agent with the real LLM
  - Score output: does `call_purpose` match expected? Are `action_items` present and accurate?
  - Log results to LangSmith (once tracing is added in Phase 5; use `print` for now)
- `evals/eval_qa_scoring.py` — component-level evaluation script
  - Run each fixture through QA scoring with the real LLM
  - Check: are dimension scores within ±1 of expected? Are justifications grounded in transcript text?
  - Check: does a known compliance violation in the fixture get flagged?

### Optional MCP Enhancement

> MCP servers are fetched **before** the LLM call and injected as additional prompt context — not as live tool calls during inference. This preserves determinism and auditability while grounding the LLM in real, current data.

- `src/services/mcp_client.py` — thin wrapper using `langchain-mcp-adapters`
  - `get_compliance_rules(call_type: str) -> str` — fetches rules from MCP Compliance Server
  - `get_agent_benchmarks(call_type: str) -> str` — fetches historical score averages from MCP Stats Server
  - Both functions return formatted strings injected into the LLM prompt; gracefully return empty string if MCP server is unavailable
- `mcp_servers/compliance_rules_server.py` — MCP server exposing compliance rulebook
  - Backed by `config/compliance_rules.yaml` — structured rules per call type (e.g. `credit_dispute`, `account_inquiry`, `billing_issue`)
  - Tool: `get_compliance_rules(call_type)` → returns list of required disclosures and verification steps
- `mcp_servers/historical_stats_server.py` — MCP server wrapping the SQLite read repository
  - Tool: `get_agent_benchmarks(call_type)` → returns avg dimension scores across past calls of that type
  - Tool: `get_recent_flags(call_type)` → returns most common compliance flags for context
- Update `src/agents/qa_scoring_agent.py`:
  - Before calling the LLM: fetch `get_compliance_rules(call_type)` and `get_agent_benchmarks(call_type)`
  - Append results as a `### Reference Context` section in the scoring prompt
  - LLM scores compliance against actual rules, not training memory

**Call flow with MCP:**
```
qa_scoring_agent.run(state)
  1. mcp_client.get_compliance_rules(call_type)   ← MCP Compliance Server
  2. mcp_client.get_agent_benchmarks(call_type)   ← MCP Stats Server
  3. Build prompt: transcript + summary + rules + benchmarks
  4. LLM.invoke(prompt) → structured QAScoreResult
  5. Recompute overall_score deterministically     ← unchanged
```

### Tests to Write (`tests/unit/`)
- LLM factory: returns correct class for each `LLM_PROVIDER` value
- LLM factory: raises `ValueError` for unknown provider
- Summarization agent: parses valid structured LLM response into `SummaryResult`
- Summarization agent: retries on `APIError`; succeeds on third attempt
- QA scoring agent: `overall_score` from LLM is always ignored
- QA scoring agent: weighted formula produces correct result for known dimension values
- QA scoring agent: `critical` compliance flag correctly identified in result
- MCP client: `get_compliance_rules()` returns empty string when MCP server is unavailable
- MCP client: `get_agent_benchmarks()` returns formatted string for known call type
- QA scoring agent: compliance rules injected into prompt when MCP client returns rules
- QA scoring agent: functions correctly with empty MCP context (graceful degradation)

### Definition of Done
- [ ] Call `summarization_agent.run()` with a mock LLM → get back a valid `SummaryResult`
- [ ] Call `qa_scoring_agent.run()` with a mock LLM → `overall_score` matches the formula, not the LLM value
- [ ] `LLM_PROVIDER=groq make run` works without code changes
- [ ] `make test-unit` passes all LLM agent tests (LLM always mocked)
- [ ] `python evals/eval_summarization.py` runs against real LLM and prints pass/fail per fixture
- [ ] `python evals/eval_qa_scoring.py` runs and confirms compliance flags fire on known violations
- [ ] At least 3 eval fixtures created and labeled in `evals/fixtures/`
- [ ] MCP servers start independently: `python mcp_servers/compliance_rules_server.py`
- [ ] QA scoring agent runs correctly with MCP servers both up and down (graceful degradation tested)

---

## Phase 5 — Pipeline Orchestration: LangGraph State Machine + LangSmith

**Goal:** Wire all agents into a single runnable pipeline with full observability. After this phase, one `workflow.invoke(state)` call processes audio end to end with all routing and error isolation working, and every LLM call is traced in LangSmith with token counts and per-node latency.

### What to Build
- `src/graph/state.py` — `PipelineState` TypedDict with all stage result fields
- `src/graph/routing.py` — three conditional edge functions:
  - `route_after_intake`: `error` if `state['error']` else `transcription`
  - `route_after_injection_check`: `error` if matched else `pii_redaction`
  - `route_after_qa_scoring`: `supervisor_review` if any critical flag else `report`
- `src/graph/pipeline.py`
  - Build `StateGraph(PipelineState)`
  - Add all 7 stage nodes + `error` node + `supervisor_review` node
  - Wire conditional edges using routing functions
  - `workflow = graph.compile()` — module-level singleton
- Error node: three-level message fallback (`state['error']` → `intake.validation_error` → generic)
- **LangSmith integration** in `src/config/loader.py` and `app.py`:
  - Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT` from env vars on startup
  - Tracing activates automatically for all LangChain/LangGraph calls when `LANGSMITH_API_KEY` is present
  - If `LANGSMITH_API_KEY` is absent, pipeline runs normally with no tracing (graceful degradation)
  - `get_langsmith_status() -> dict` — returns `{"enabled": bool, "project": str, "url": str | None}`; consumed by the Observability tab in Phase 7
- **Update `evals/eval_*.py`** from Phase 4: replace `print` statements with LangSmith-logged results now that tracing is active

### LangSmith Tracing Coverage
| What Gets Traced | Where |
|-----------------|-------|
| Every LLM call (summarization, QA scoring) | Automatic via LangChain integration |
| Full pipeline run per call (input → output) | LangGraph traces each node execution |
| Token counts and cost estimate per call | Automatic via LangSmith |
| Per-node latency breakdown | Automatic via LangGraph + LangSmith |
| Retry attempts on LLM failures | Traced as child runs |

### Tests to Write (`tests/integration/`)
- Happy path: valid MP3 → all 7 stages run → `CallReport` in final state (LLM + Whisper mocked)
- Intake failure: unsupported format → `error` node reached; other stages not called
- Injection detected: transcript with injection pattern → `error` node; summarization not called
- Critical compliance flag: QA returns `critical` flag → `supervisor_review` node reached
- Non-critical flag: QA returns `high` flag → `report` node reached (not supervisor)
- Each stage's exception is isolated: one stage raising does not corrupt other stage results
- `get_langsmith_status()` returns `{"enabled": False}` when `LANGSMITH_API_KEY` is not set
- `get_langsmith_status()` returns `{"enabled": True, "url": ...}` when key is present

### Definition of Done
- [ ] `workflow.invoke(initial_state)` with a real (tiny) MP3 returns a complete `PipelineState`
- [ ] All 6 routing scenarios tested and passing
- [ ] A failing stage does not crash the process — `state['error']` is set and returned
- [ ] `make test-integration` passes all pipeline routing tests
- [ ] With `LANGSMITH_API_KEY` set: pipeline run appears in LangSmith dashboard with per-node traces
- [ ] Without `LANGSMITH_API_KEY`: pipeline runs normally with no errors or warnings
- [ ] Eval scripts from Phase 4 now log results as LangSmith feedback

---

## Phase 6 — Persistence & Reports: SQLite + PDF + JSON

**Goal:** Every analyzed call is stored, downloadable, and survives a server restart. After this phase, reports can be downloaded and call history can be browsed from the database.

### What to Build
- `src/database/models.py` — complete SQLAlchemy ORM:
  - `CallRecord` — all fields including transcript, summary, QA, report JSON, status, timestamps
  - `TranscriptionCache` — `sha256_hash` PK + `transcription_json` + `created_at`
  - `AuditLog` — INSERT-only; no `updated_at` column
- `src/database/database.py`
  - `get_session()` — cached `sessionmaker` factory; creates DB + tables on first call
  - Never opens raw connections per request
- `src/database/repository.py`
  - `save_call_record(session, report)` — upsert `CallRecord`
  - `get_call_history(session, limit)` — ordered by `created_at DESC`
  - `get_cached_transcription(session, sha256)` → `TranscriptionResult | None`
  - `save_transcription_cache(session, sha256, result)` — INSERT only
  - `log_audit_event(session, call_id, action, details)` — INSERT only
- `src/agents/report_agent.py`
  - Assemble `CallReport` from upstream state
  - Call `save_call_record`, `save_transcription_cache`, `log_audit_event`
- `src/services/pdf_generator.py`
  - ReportLab: summary section, QA scorecard table, compliance flags section
  - Returns `bytes` — no file I/O inside the function

### Tests to Write (`tests/integration/`)
- `save_call_record` → `get_call_history` returns it (in-memory SQLite)
- `save_transcription_cache` → `get_cached_transcription` returns the same result
- `log_audit_event` called 3× → 3 rows in `audit_log`; no UPDATE path exists
- DB persists across session factory calls (same file, new session)
- PDF generator returns non-empty bytes for a valid `CallReport`
- JSON export is valid JSON and round-trips through `CallReport` model

### Definition of Done
- [ ] Full pipeline run → `CallReport` persisted → restart app → call still in DB
- [ ] PDF download bytes open as a valid PDF
- [ ] `audit_log` has one row per pipeline event; rows are never modified
- [ ] SHA-256 cache round-trip: save then retrieve returns identical `TranscriptionResult`
- [ ] `make test-integration` passes all persistence tests

---

## Phase 7 — Web Interface: Gradio Tabs

**Goal:** A working browser UI that covers all six evaluator test scenarios. After this phase, the product is demo-able end to end.

### What to Build
- `src/ui/interface.py` — `build_interface()` assembles all three tabs
- `src/ui/analyze_tab.py`
  - Audio upload component (file + microphone)
  - Optional caller ID and department text inputs
  - Analyze button → calls `workflow.invoke()` → formats results
  - Processing status message with estimated duration and do-not-refresh warning
  - Transcript display (speaker-labeled), summary markdown, QA scorecard markdown
  - PDF download button, JSON download button
- `src/ui/history_tab.py`
  - Master list of all analyzed calls (from `get_call_history`)
  - Select a call → show its transcript, summary, and QA scorecard in detail pane
- `src/ui/observability_tab.py`
  - Auto-refresh on tab select; manual Refresh button
  - Metric cards: total calls, completed / failed / flagged counts, success rate %, average QA score, total compliance flags
  - Table: 20 most recent audit events (timestamp, call ID, action, details)
  - LangSmith status indicator: `Enabled — link` or `Disabled`

### Validation: All Six Evaluator Scenarios
Run through each manually with a real audio file from the Kaggle dataset:

| Scenario | Pass Condition |
|----------|---------------|
| 1 — Valid audio | Speaker-labeled transcript + summary + QA scorecard + working downloads |
| 2 — PII in audio | `[REDACTED_CREDIT_CARD]` visible in transcript; LLM output has no raw digits |
| 3 — Compliance violation | `HIGH` flag shown with timestamp; critical flag routes to supervisor_review |
| 4 — Prompt injection | Error shown with matched pattern names; no summary or QA generated |
| 5 — Invalid file | Clear rejection message listing WAV/MP3/FLAC/M4A |
| 6 — Observability tab | All metrics load; audit table shows 20 most recent events |

### Definition of Done
- [ ] All six evaluator scenarios pass manually
- [ ] App starts with `python app.py` and is reachable at `http://localhost:7860`
- [ ] PDF and JSON downloads produce valid files
- [ ] Observability tab loads without errors after multiple pipeline runs
- [ ] No unhandled exceptions visible in browser UI (errors are shown as user-friendly messages)

---

## Phase 8 — Production Readiness: Tests to 100+, Batch Eval, Docker

**Goal:** The submission is clean, complete, and passes from a cold clone. This phase also closes the evaluation loop opened in Phase 4 — running a batch pipeline eval over the Kaggle dataset to confirm end-to-end quality before submission.

### What to Build / Complete
- Fill any test gaps to reach 100+ total across all three suites
- `Dockerfile` — verify clean build from scratch; includes `ffmpeg` and `libsndfile1`
- `docker-compose.yml` — mounts `data/` and `logs/` volumes; reads `.env`
- `README.md` — architecture overview, setup steps, how to run, how to test, GPU deployment note, sample usage
- `pre-commit` config — `ruff` lint + format checks on every commit
- Verify `.gitignore` excludes `data/audio/`, `data/*.db`, `.env`, model weights
- GPU deployment note in README: set `WHISPER_MODEL=large-v3`; ensure CUDA drivers installed
- `evals/eval_pipeline.py` — end-to-end batch evaluation script
  - Run 5–10 Kaggle audio files through the full pipeline
  - For each call: assert `TranscriptionResult` is populated, `SummaryResult` has all required fields, `QAScoreResult.overall_score` matches the weighted formula, no raw PII appears in the LLM inputs (verified via LangSmith traces)
  - Log aggregate metrics to LangSmith: average QA score, provider latency comparison, token cost per call

### Test Coverage Checklist (reach 100+)
| Suite | Target Count | Covers |
|-------|-------------|--------|
| `tests/unit/` | ~60 | All agent functions, security functions, routing logic, all 14 Pydantic models, display formatters, LLM factory |
| `tests/integration/` | ~25 | End-to-end pipeline (mocked LLM + Whisper), DB persistence, audit log, cache round-trip |
| `tests/security/` | ~20 | 22+ injection patterns, PII format variants (phone, email, SSN, credit card), PII embedded in longer text |

### Evaluation Quality Gates (before marking Done)
| Check | Pass Condition |
|-------|---------------|
| Summarization accuracy | `call_purpose` is meaningful for all 5+ eval fixtures |
| QA score calibration | `overall_score` within expected range (±0.5) for labeled fixtures |
| PII non-leakage | LangSmith traces confirm no raw SSN/CC/email/phone in any LLM input |
| Provider parity | All three providers (OpenAI, Gemini, Groq) produce structurally valid output |
| Score determinism | Same transcript run twice → identical `overall_score` both times |

### Definition of Done
- [ ] `git clone ... && pip install -e ".[dev]" && pytest tests/ -v` → 100+ passing, 0 failing
- [ ] `docker compose up --build` → app reachable at `http://localhost:7860`
- [ ] `make lint` passes with zero errors
- [ ] `python evals/eval_pipeline.py` passes all quality gates above
- [ ] No API keys, `.env`, or audio files committed to the repository
- [ ] `README.md` allows a new reviewer to set up and run the app without asking questions

---

## Phase 9 — AWS Deployment

**Goal:** The application runs as a managed container on AWS, with secrets stored securely, logs shipped to CloudWatch, and the SQLite database persisted on EFS. After this phase, the system is accessible via a public HTTPS endpoint without running anything locally.

> This phase uses AWS CDK (Python) to define infrastructure as code, following the same pattern as the `infrastructure/stacks/` layout from related projects.

### What to Build
- `infrastructure/stacks/network_stack.py` — VPC with public/private subnets, NAT gateway
- `infrastructure/stacks/secrets_stack.py`
  - Store all API keys in AWS Secrets Manager: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `LANGSMITH_API_KEY`
  - ECS task role granted read access to these secrets at runtime
- `infrastructure/stacks/storage_stack.py`
  - EFS file system for persistent SQLite (`data/calls.db`) and logs — survives container restarts
  - ECR repository for the Docker image
- `infrastructure/stacks/compute_stack.py`
  - ECS Fargate service running the app container (CPU-only; `WHISPER_MODEL=base`)
  - Application Load Balancer with HTTPS listener (ACM certificate)
  - Container environment variables wired from Secrets Manager
  - CloudWatch log group for container stdout/stderr
  - Health check on `GET /` → Gradio's root
- `infrastructure/app.py` — CDK app entry point; imports and chains all four stacks
- `Makefile` additions: `cdk-deploy`, `cdk-destroy`, `ecr-push`

### GPU Deployment Option
For `WHISPER_MODEL=large-v3`, swap the Fargate task for an EC2-backed ECS cluster using a `g4dn.xlarge` (NVIDIA T4) instance. Set `ECS_ENABLE_GPU_SUPPORT=true` and use the GPU-optimised ECS AMI. Document this as an optional upgrade path in the README.

### Environment Variable Mapping (local → AWS)
| Local `.env` var | AWS Source |
|-----------------|-----------|
| `OPENAI_API_KEY` | Secrets Manager secret |
| `GEMINI_API_KEY` | Secrets Manager secret |
| `GROQ_API_KEY` | Secrets Manager secret |
| `LANGSMITH_API_KEY` | Secrets Manager secret |
| `DB_PATH` | `/mnt/efs/data/calls.db` (EFS mount) |
| `WHISPER_MODEL` | ECS task definition environment variable |
| `APP_PORT` | `7860` — ALB target group port |

### Definition of Done
- [ ] `make ecr-push` builds the Docker image and pushes to ECR
- [ ] `make cdk-deploy` provisions all four stacks without errors
- [ ] App is reachable at the ALB DNS name (or custom domain) over HTTPS
- [ ] `data/calls.db` survives a container restart (EFS mount persists)
- [ ] API keys are NOT in the task definition as plaintext — all come from Secrets Manager
- [ ] Container logs appear in CloudWatch log group
- [ ] `make cdk-destroy` tears down all resources cleanly

---

## Phase Summary

| Phase | Deliverable | Testable Output |
|-------|------------|-----------------|
| 1 | Data contracts + config + logger + exceptions | All 14 Pydantic models validate; logger writes to file; all exception types importable |
| 2 | Audio intake + transcription + cache | Upload audio → get speaker-labeled transcript |
| 3 | Security layer | PII redacted; injection patterns blocked; audit log writes |
| 4 | LLM analysis + component eval + MCP servers | Structured summary + deterministic QA scores; compliance rules grounded in rulebook; eval fixtures pass quality checks |
| 5 | LangGraph pipeline + LangSmith tracing | `workflow.invoke()` routes through all 7 stages; traces visible in LangSmith |
| 6 | Persistence + reports | PDF/JSON downloadable; call history survives restart |
| 7 | Gradio web UI | All 6 evaluator scenarios pass in the browser |
| 8 | Production readiness + batch eval | 100+ tests pass; Docker build works; pipeline quality gates pass |
| 9 | AWS deployment | App live on HTTPS; secrets in Secrets Manager; DB on EFS; logs in CloudWatch |
