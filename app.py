"""Entry point — loads .env (including LangSmith tracing vars), then launches Gradio UI."""
from dotenv import load_dotenv

# Must be first — sets LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, etc. before any LangChain import
load_dotenv()

from src.services.whisper_model import get_whisper_model  # noqa: E402
from src.config.loader import get_settings  # noqa: E402
from src.ui.interface import build_interface  # noqa: E402

# Load Whisper once at startup — never inside a request handler
get_whisper_model()

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_port=get_settings().app_port)
