# Environment Configuration

Copy `.env.example` to `.env` and fill in your values.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `openai` | Active LLM: `openai` \| `gemini` \| `groq` |
| `OPENAI_API_KEY` | If provider=openai | — | GPT-4o API key |
| `GEMINI_API_KEY` | If provider=gemini | — | Gemini 2.0 Flash API key |
| `GROQ_API_KEY` | If provider=groq | — | Groq Llama 3.3 70B API key |
| `LANGSMITH_API_KEY` | No | — | Enables LangSmith tracing when set |
| `LANGSMITH_PROJECT` | No | `call-center-intel` | LangSmith project name |
| `WHISPER_MODEL` | No | `base` | Model size: `tiny` \| `base` \| `large-v3` |
| `MAX_FILE_SIZE_MB` | No | `50` | Maximum audio upload size |
| `MAX_DURATION_MINUTES` | No | `60` | Maximum audio duration |
| `DB_PATH` | No | `data/calls.db` | SQLite database file path |
| `APP_PORT` | No | `7860` | Gradio server port |
| `MAX_TEMP_FILES` | No | `100` | Rolling temp file cleanup limit |
