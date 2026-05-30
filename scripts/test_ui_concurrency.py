"""
Concurrency smoke-test for the Gradio UI.

Replicates the exact queue/concurrency settings from interface.py but replaces
workflow.invoke() with asyncio.sleep(30) so no audio file is needed.

HOW TO RUN:
    python scripts/test_ui_concurrency.py

WHAT TO DO:
    1. Open http://localhost:7861
    2. Click "Slow Analyze" (simulates the 30-second pipeline)
    3. IMMEDIATELY switch to the "Fast Tab" and click "Fast Refresh"
    4. Fast Refresh should respond within 1-2 seconds even while Slow Analyze runs
"""
from __future__ import annotations

import asyncio

import gradio as gr


# ---------------------------------------------------------------------------
# Simulate the pipeline_status banner (lives outside tabs, like interface.py)
# ---------------------------------------------------------------------------

async def _slow_analyze():
    """Async generator — mirrors _run_analysis in analyze_tab.py."""
    yield (
        gr.update(value="⏳ **Pipeline running** — 30-second simulated analysis.", visible=True),
        "⏳ Analyzing…",
        gr.update(interactive=False),
    )

    await asyncio.sleep(30)  # ← replaces workflow.invoke() in executor

    yield (
        gr.update(value="", visible=False),
        "✅ Analysis complete (simulated).",
        gr.update(interactive=True),
    )


def _fast_refresh():
    """Instant DB-like query — mirrors History/Observability Refresh handlers."""
    return f"Refreshed at {time.strftime('%H:%M:%S')} — responded immediately ✅"


# ---------------------------------------------------------------------------
# Build the test app (mirrors interface.py structure exactly)
# ---------------------------------------------------------------------------

with gr.Blocks(title="Concurrency Test") as app:
    gr.Markdown("## Gradio Concurrency Smoke-Test")
    pipeline_status = gr.Markdown(visible=False)   # same as interface.py

    with gr.Tab("Slow Tab"):
        gr.Markdown("Click the button below then switch to **Fast Tab**.")
        analyze_btn = gr.Button("Slow Analyze (30 s)", variant="primary")
        status_out = gr.Markdown("Idle.")

        analyze_btn.click(
            fn=_slow_analyze,
            inputs=[],
            outputs=[pipeline_status, status_out, analyze_btn],
            concurrency_id="pipeline",   # same as analyze_tab.py
            concurrency_limit=1,
        )

    with gr.Tab("Fast Tab"):
        gr.Markdown("Click Refresh — it should respond in under 2 seconds.")
        refresh_btn = gr.Button("Fast Refresh", variant="secondary")
        refresh_out = gr.Markdown("Not yet refreshed.")

        refresh_btn.click(
            fn=_fast_refresh,
            inputs=[],
            outputs=[refresh_out],
            # no concurrency_id → uses default pool (same as Refresh buttons)
        )

app.queue(default_concurrency_limit=4)   # same as interface.py

if __name__ == "__main__":
    app.launch(server_port=7861)
