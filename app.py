"""Entry point — loads .env (including LangSmith tracing vars), then launches Gradio UI."""
from dotenv import load_dotenv

# Must be first — sets LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, etc. before any LangChain import
load_dotenv()

from src.services.whisper_model import get_whisper_model  # noqa: E402
from src.config.loader import get_settings  # noqa: E402
from src.database.database import get_session  # noqa: E402
from src.ui.interface import build_interface  # noqa: E402

# Warm up singletons at startup so the first UI interaction is never slow
get_whisper_model()
with get_session() as _warm:  # initialises SQLAlchemy engine + tables once
    pass

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_port=get_settings().app_port)
