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

        # Top-level banner — always visible regardless of which tab is active.
        # The Analyze generator writes here so users see pipeline status even
        # when they switch to History or Observability mid-run.
        pipeline_status = gr.Markdown(visible=False)

        build_analyze_tab(pipeline_status)
        build_history_tab()
        build_observability_tab()

    # default_concurrency_limit=4 lets Refresh buttons and other lightweight
    # events run concurrently. The analyze pipeline gets its own named lane
    # (concurrency_id="pipeline") so it never blocks UI events.
    app.queue(default_concurrency_limit=4)
    return app
