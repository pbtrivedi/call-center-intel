from __future__ import annotations

import json

import gradio as gr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_records() -> tuple[list[list], list[dict]]:
    """Return (table_rows, full_report_json_list) for the 50 most recent calls."""
    try:
        from src.database.database import get_session
        from src.database.repository import get_call_history

        with get_session() as session:
            records = get_call_history(session, limit=50)
    except Exception:
        return [], []

    rows: list[list] = []
    reports: list[dict] = []
    for r in records:
        rows.append([
            r.call_id[:8] + "…",
            r.filename,
            r.status.replace("_", " ").title(),
            f"{r.overall_qa_score:.2f}" if r.overall_qa_score is not None else "—",
            r.analyzed_at.strftime("%Y-%m-%d %H:%M UTC") if r.analyzed_at else "—",
        ])
        reports.append(r.report_json or {})

    return rows, reports


# ---------------------------------------------------------------------------
# Detail formatters  (rebuild from stored report JSON)
# ---------------------------------------------------------------------------


def _fmt_transcript_from_json(data: dict) -> str:
    tx = data.get("transcription", {})
    segments = tx.get("segments", [])
    if not segments:
        return "_No transcript available._"
    lines = []
    for seg in segments:
        start = seg.get("start_time", 0)
        ts = f"{int(start // 60):02d}:{int(start % 60):02d}"
        lines.append(f"**[{ts}] {seg.get('speaker', '?')}:** {seg.get('text', '')}")
    return "\n\n".join(lines)


def _fmt_summary_from_json(data: dict) -> str:
    s = data.get("summary", {})
    if not s:
        return "_No summary available._"
    parts = [
        f"**Purpose:** {s.get('call_purpose', '—')}",
        f"**Resolution:** {s.get('resolution_status', '—').title()}  |  "
        f"**Sentiment:** {s.get('sentiment_trajectory', '—')}",
        "",
        "**Key Discussion Points:**",
    ]
    for pt in s.get("key_discussion_points", []):
        parts.append(f"- {pt}")
    action_items = s.get("action_items", [])
    if action_items:
        parts.append("")
        parts.append("**Action Items:**")
        for ai in action_items:
            deadline = f" — due {ai['deadline']}" if ai.get("deadline") else ""
            parts.append(f"- [{ai.get('owner', '?')}] {ai.get('description', '')}{deadline}")
    return "\n".join(parts)


def _fmt_qa_from_json(data: dict) -> str:
    qa = data.get("qa_scores", {})
    if not qa:
        return "_No QA data available._"

    _weights = {
        "professionalism": 0.15, "empathy": 0.20,
        "problem_resolution": 0.30, "compliance": 0.20, "clarity": 0.15,
    }

    overall = qa.get("overall_score", 0.0)
    parts = [
        f"**Overall Score: {overall:.2f} / 5.00**",
        "",
        "| Dimension | Score | Weight | Justification |",
        "|-----------|:-----:|:------:|---------------|",
    ]
    for dim in qa.get("dimensions", []):
        name = dim.get("name", "").replace("_", " ").title()
        score = dim.get("score", 0.0)
        weight = _weights.get(dim.get("name", ""), 0)
        just = dim.get("justification", "")
        just = just[:110] + ("…" if len(just) > 110 else "")
        parts.append(f"| {name} | {score:.1f} | {weight:.0%} | {just} |")

    flags = qa.get("compliance_flags", [])
    if flags:
        parts.append("")
        parts.append("**Compliance Flags:**")
        for flag in flags:
            ts = f" @{flag['timestamp']}" if flag.get("timestamp") else ""
            sev = flag.get("severity", "").upper()
            parts.append(f"- **[{sev}]**{ts} {flag.get('description', '')}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_history_tab() -> None:
    with gr.Tab("History") as history_tab:
        gr.Markdown("## Call History")

        with gr.Row():
            refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")

        history_state = gr.State([])

        history_df = gr.Dataframe(
            headers=["Call ID", "Filename", "Status", "QA Score", "Analyzed At"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            label="Click Refresh to load calls, then select a row to view details",
        )

        with gr.Column(visible=False) as detail_col:
            gr.Markdown("### Call Detail")
            with gr.Tabs():
                with gr.Tab("Transcript"):
                    detail_transcript = gr.Markdown()
                with gr.Tab("Summary"):
                    detail_summary = gr.Markdown()
                with gr.Tab("QA Scorecard"):
                    detail_qa = gr.Markdown()

        def _on_load():
            rows, reports = _load_records()
            return rows, reports

        def _on_select(evt: gr.SelectData, reports: list[dict]):
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if not reports or row_idx >= len(reports):
                return gr.update(visible=False), "", "", ""
            data = reports[row_idx]
            transcript_md = _fmt_transcript_from_json(data)
            summary_md = _fmt_summary_from_json(data)
            qa_md = _fmt_qa_from_json(data)
            return gr.update(visible=True), transcript_md, summary_md, qa_md

        refresh_btn.click(
            fn=_on_load,
            inputs=[],
            outputs=[history_df, history_state],
        )
        history_df.select(
            fn=_on_select,
            inputs=[history_state],
            outputs=[detail_col, detail_transcript, detail_summary, detail_qa],
        )
