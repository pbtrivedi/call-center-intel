# Architecture

> To be filled in during development.

## Five-Layer Structure

```
src/
├── ui/           Gradio tabs (Analyze Call, History, Observability)
├── services/     LLM factory, Whisper singleton, audio utils, PDF generator
├── agents/       Seven pipeline stage implementations
├── graph/        LangGraph state machine, PipelineState, routing edges
├── database/     SQLAlchemy ORM models, session factory, repositories
├── models/       Pydantic contracts (14 typed models)
├── security/     Injection detector, PII redactor, audit logger
└── config/       YAML configs, env loader
```

## Pipeline Flow

```
intake → transcription → injection_check → pii_redaction → summarization → qa_scoring → report
                                ↓                                                  ↓
                              error                                        supervisor_review
```

## Data Flow

See [DATA_SCHEMAS.md](DATA_SCHEMAS.md) for the 14 Pydantic contracts between stages.
