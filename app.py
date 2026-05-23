"""Entry point — loads Whisper singleton, launches Gradio UI."""
from dotenv import load_dotenv

load_dotenv()

from src.services.whisper_model import get_whisper_model  # noqa: E402
from src.ui.interface import build_interface  # noqa: E402

# Load Whisper once at startup — never inside a request handler
get_whisper_model()

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_port=7860)
