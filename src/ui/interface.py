from __future__ import annotations

import gradio as gr

from src.ui.analyze_tab import build_analyze_tab
from src.ui.history_tab import build_history_tab
from src.ui.observability_tab import build_observability_tab


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Call Center Intelligence") as app:
        gr.Markdown(
            "# Call Center Intelligence\n"
            "AI-powered call analysis — transcription, summarization, QA scoring, and compliance flagging."
        )

        build_analyze_tab()
        build_history_tab()
        build_observability_tab()

    return app
