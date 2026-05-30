from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from src.models.schemas import AudioInput

_WEIGHTS = {
    "professionalism": 0.15,
    "empathy": 0.20,
    "problem_resolution": 0.30,
    "compliance": 0.20,
    "clarity": 0.15,
}


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------


def _fmt_transcript(transcription) -> str:
    if transcription is None:
        return "_No transcript available._"
    lines = []
    for seg in transcription.segments:
        ts = f"{int(seg.start_time // 60):02d}:{int(seg.start_time % 60):02d}"
        lines.append(f"**[{ts}] {seg.speaker}:** {seg.text}")
    return "\n\n".join(lines) if lines else "_Empty transcript._"


def _fmt_summary(report) -> str:
    s = report.summary
    parts = [
        f"**Purpose:** {s.call_purpose}",
        f"**Resolution:** {s.resolution_status.title()}  |  **Sentiment:** {s.sentiment_trajectory}",
        "",
        "**Key Discussion Points:**",
    ]
    for pt in s.key_discussion_points:
        parts.append(f"- {pt}")
    if s.action_items:
        parts.append("")
        parts.append("**Action Items:**")
        for ai in s.action_items:
            deadline = f" — due {ai.deadline}" if ai.deadline else ""
            parts.append(f"- [{ai.owner}] {ai.description}{deadline}")
    return "\n".join(parts)


def _fmt_qa(report) -> str:
    qa = report.qa_scores
    parts = [
        f"**Overall Score: {qa.overall_score:.2f} / 5.00**",
        "",
        "| Dimension | Score | Weight | Justification |",
        "|-----------|:-----:|:------:|---------------|",
    ]
    for dim in qa.dimensions:
        name = dim.name.replace("_", " ").title()
        weight = _WEIGHTS.get(dim.name, 0)
        just = dim.justification[:110] + ("…" if len(dim.justification) > 110 else "")
        parts.append(f"| {name} | {dim.score:.1f} | {weight:.0%} | {just} |")

    if qa.compliance_flags:
        parts.append("")
        parts.append("**Compliance Flags:**")
        for flag in qa.compliance_flags:
            ts = f" @{flag.timestamp}" if flag.timestamp else ""
            parts.append(f"- **[{flag.severity.upper()}]**{ts} {flag.description}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Pipeline runner — generator so the UI stays responsive during processing
# ---------------------------------------------------------------------------


def _run_analysis(audio_path: str | None, caller_id: str, department: str):
    """
    Yields tuples of (pipeline_status, status, transcript, summary, qa, pdf_btn, json_btn, analyze_btn).

    First yield is immediate so the browser unblocks before the long pipeline call.
    """
    _hidden_banner = gr.update(value="", visible=False)
    _no_downloads = (gr.update(visible=False), gr.update(visible=False))
    _btn_on = gr.update(interactive=True)
    _btn_off = gr.update(interactive=False)

    if audio_path is None:
        yield (
            _hidden_banner,
            "Upload or record audio then click **Analyze**.",
            "", "", "",
            *_no_downloads,
            _btn_on,
        )
        return

    # ── First yield: unblock the UI immediately ──────────────────────────────
    yield (
        gr.update(
            value="⏳ **Pipeline running** — analysis in progress. You can browse other tabs while waiting.",
            visible=True,
        ),
        "⏳ Analyzing… (30–120 s)",
        "", "", "",
        *_no_downloads,
        _btn_off,
    )

    # ── Run the pipeline (long-running) ──────────────────────────────────────
    try:
        from src.graph.pipeline import workflow

        filename = Path(audio_path).name
        audio_input = AudioInput(
            file_path=audio_path,
            filename=filename,
            caller_id=caller_id.strip() or None,
            department=department.strip() or None,
        )

        final_state = workflow.invoke({"audio_input": audio_input})

        if final_state.get("error"):
            yield (
                _hidden_banner,
                f"**Pipeline error:** {final_state['error']}",
                "", "", "",
                *_no_downloads,
                _btn_on,
            )
            return

        report = final_state.get("call_report")
        if report is None:
            yield (
                _hidden_banner,
                "**No report generated.** Check logs for details.",
                "", "", "",
                *_no_downloads,
                _btn_on,
            )
            return

        transcript_md = _fmt_transcript(final_state.get("transcription_result"))
        summary_md = _fmt_summary(report)
        qa_md = _fmt_qa(report)

        from src.services.pdf_generator import generate as generate_pdf

        pdf_bytes = generate_pdf(report)
        pdf_path = os.path.join(tempfile.gettempdir(), f"report_{report.call_id[:8]}.pdf")
        with open(pdf_path, "wb") as fh:
            fh.write(pdf_bytes)

        json_path = os.path.join(tempfile.gettempdir(), f"report_{report.call_id[:8]}.json")
        with open(json_path, "w") as fh:
            fh.write(report.model_dump_json(indent=2))

        label = report.status.replace("_", " ").title()
        status_msg = (
            f"✅ Analysis complete — **{label}** | "
            f"QA score: **{report.qa_scores.overall_score:.2f}/5.00**"
        )

        yield (
            _hidden_banner,
            status_msg,
            transcript_md, summary_md, qa_md,
            gr.update(value=pdf_path, visible=True),
            gr.update(value=json_path, visible=True),
            _btn_on,
        )

    except Exception as exc:
        yield (
            _hidden_banner,
            f"**Error:** {exc}",
            "", "", "",
            *_no_downloads,
            _btn_on,
        )


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_analyze_tab(pipeline_status: gr.Markdown) -> None:
    with gr.Tab("Analyze"):
        gr.Markdown("## Analyze a Call Recording")

        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio File  (WAV / MP3 / FLAC / M4A, max 50 MB)",
                )
                with gr.Row():
                    caller_id = gr.Textbox(
                        label="Caller ID (optional)",
                        placeholder="e.g. 555-0100",
                        scale=1,
                    )
                    department = gr.Textbox(
                        label="Department (optional)",
                        placeholder="e.g. Billing",
                        scale=1,
                    )
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                status = gr.Markdown(
                    "*Upload an audio file and click **Analyze**.*"
                )

        with gr.Row():
            pdf_btn = gr.DownloadButton(
                "Download PDF Report",
                visible=False,
                variant="secondary",
                size="sm",
            )
            json_btn = gr.DownloadButton(
                "Download JSON",
                visible=False,
                variant="secondary",
                size="sm",
            )

        with gr.Tabs():
            with gr.Tab("Transcript"):
                transcript_out = gr.Markdown()
            with gr.Tab("Summary"):
                summary_out = gr.Markdown()
            with gr.Tab("QA Scorecard"):
                qa_out = gr.Markdown()

        analyze_btn.click(
            fn=_run_analysis,
            inputs=[audio, caller_id, department],
            outputs=[
                pipeline_status,   # top-level banner in interface.py
                status,
                transcript_out, summary_out, qa_out,
                pdf_btn, json_btn,
                analyze_btn,       # disabled while running, re-enabled on finish
            ],
        )
