# Call Center Intelligence System

AI-powered multi-agent pipeline that processes call center audio and produces transcripts, quality scores, compliance reports, and downloadable artifacts.

## Architecture

Seven-stage LangGraph pipeline with PII redaction, prompt injection defense, and deterministic QA scoring. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
intake → transcription → injection_check → pii_redaction → summarization → qa_scoring → report
```

## Quick Start

See [docs/QUICK_START.md](docs/QUICK_START.md).

```bash
make install
cp .env.example .env   # add your LLM API key
make run               # http://localhost:7860
```

## Configuration

See [docs/ENV_CONFIGURATION.md](docs/ENV_CONFIGURATION.md) for all environment variables.

## Testing

```bash
make test-all          # 100+ tests across unit, integration, and security suites
```

## Tech Stack

Python 3.11 · LangGraph 0.4 · faster-whisper · Gradio 5 · SQLite/SQLAlchemy · Pydantic v2 · ReportLab

## Dataset

[911-recordings on Kaggle](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings) for test audio.
