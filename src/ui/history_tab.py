from __future__ import annotations

import gradio as gr
from sqlalchemy import select

from src.database.database import get_session
from src.database.models import CallRecord
from src.database.repository import delete_call_record, get_call_history


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_records() -> tuple[list[list], list[str], str]:
    """Return (table_rows, call_id_list, status_msg) for the 50 most recent calls."""
    try:
        with get_session() as session:
            records = get_call_history(session, limit=50)
    except Exception as exc:
        return [], [], f"_Database error: {exc}_"

    if not records:
        return [], [], "_No call records found. Analyze a call to see results here._"

    rows: list[list] = []
    call_ids: list[str] = []
    for r in records:
        rows.append([
            r.call_id[:8] + "…",
            r.filename,
            r.status.replace("_", " ").title(),
            f"{r.overall_qa_score:.2f}" if r.overall_qa_score is not None else "—",
            r.analyzed_at.strftime("%Y-%m-%d %H:%M UTC") if r.analyzed_at else "—",
        ])
        call_ids.append(r.call_id)

    return rows, call_ids, f"_{len(records)} record(s) loaded._"


def _load_detail(call_id: str) -> tuple[str, str, str]:
    """Fetch transcript/summary/QA markdown for a single call_id from DB."""
    try:
        with get_session() as session:
            record = session.execute(
                select(CallRecord).where(CallRecord.call_id == call_id)
            ).scalar_one_or_none()
    except Exception as exc:
        return f"_DB error: {exc}_", "", ""

    if record is None:
        return "_Record not found._", "", ""

    data = record.report_json or {}
    return (
        _fmt_transcript(data),
        _fmt_summary(data),
        _fmt_qa(data),
    )


# ---------------------------------------------------------------------------
# Markdown formatters (from stored report_json)
# ---------------------------------------------------------------------------


def _fmt_transcript(data: dict) -> str:
    segments = data.get("transcription", {}).get("segments", [])
    if not segments:
        return "_No transcript available._"
    lines = []
    for seg in segments:
        start = seg.get("start_time", 0)
        ts = f"{int(start // 60):02d}:{int(start % 60):02d}"
        lines.append(f"**[{ts}] {seg.get('speaker', '?')}:** {seg.get('text', '')}")
    return "\n\n".join(lines)


def _fmt_summary(data: dict) -> str:
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
    for ai in s.get("action_items", []):
        deadline = f" — due {ai['deadline']}" if ai.get("deadline") else ""
        parts.append(f"- [{ai.get('owner', '?')}] {ai.get('description', '')}{deadline}")
    return "\n".join(parts)


_WEIGHTS = {
    "professionalism": 0.15, "empathy": 0.20,
    "problem_resolution": 0.30, "compliance": 0.20, "clarity": 0.15,
}


def _fmt_qa(data: dict) -> str:
    qa = data.get("qa_scores", {})
    if not qa:
        return "_No QA data available._"
    parts = [
        f"**Overall Score: {qa.get('overall_score', 0):.2f} / 5.00**",
        "",
        "| Dimension | Score | Weight | Justification |",
        "|-----------|:-----:|:------:|---------------|",
    ]
    for dim in qa.get("dimensions", []):
        name = dim.get("name", "").replace("_", " ").title()
        score = dim.get("score", 0.0)
        weight = _WEIGHTS.get(dim.get("name", ""), 0)
        just = dim.get("justification", "")
        just = just[:110] + ("…" if len(just) > 110 else "")
        parts.append(f"| {name} | {score:.1f} | {weight:.0%} | {just} |")
    for flag in qa.get("compliance_flags", []):
        ts = f" @{flag['timestamp']}" if flag.get("timestamp") else ""
        parts.append(f"- **[{flag.get('severity','').upper()}]**{ts} {flag.get('description', '')}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_history_tab() -> None:
    # Pre-populate at build time — session is already warmed up by app.py
    initial_rows, initial_ids, initial_status = _load_records()

    with gr.Tab("History"):
        gr.Markdown("## Call History")

        with gr.Row():
            refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
            delete_btn = gr.Button(
                "Delete Selected", variant="stop", size="sm", interactive=False
            )

        load_status = gr.Markdown(initial_status)

        # State holds only the list of full call_ids — small, no JSON blobs
        call_ids_state = gr.State(initial_ids)
        selected_id_state = gr.State(None)

        history_df = gr.Dataframe(
            value=initial_rows,
            headers=["Call ID", "Filename", "Status", "QA Score", "Analyzed At"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            label="Select a row to view details",
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

        # ── Helpers ──────────────────────────────────────────────────────────

        def _reset_detail():
            return (
                gr.update(visible=False),  # detail_col
                "", "", "",                # detail content
                None,                      # selected_id_state
                gr.update(interactive=False),  # delete_btn
            )

        def _on_load():
            rows, call_ids, status_msg = _load_records()
            return (rows, call_ids, status_msg) + _reset_detail()

        def _on_select(evt: gr.SelectData, call_ids: list[str]):
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if not call_ids or row_idx >= len(call_ids):
                return (gr.update(visible=False), "", "", "", None, gr.update(interactive=False))
            call_id = call_ids[row_idx]
            transcript_md, summary_md, qa_md = _load_detail(call_id)
            return (
                gr.update(visible=True),
                transcript_md, summary_md, qa_md,
                call_id,
                gr.update(interactive=True),
            )

        def _on_delete(call_id: str | None):
            if call_id:
                try:
                    with get_session() as session:
                        delete_call_record(session, call_id)
                except Exception:
                    pass
            rows, call_ids, status_msg = _load_records()
            return (rows, call_ids, status_msg) + _reset_detail()

        # ── Event wiring ─────────────────────────────────────────────────────

        _detail_outputs = [
            detail_col, detail_transcript, detail_summary, detail_qa,
            selected_id_state, delete_btn,
        ]
        _load_outputs = [history_df, call_ids_state, load_status] + _detail_outputs

        refresh_btn.click(fn=_on_load, inputs=[], outputs=_load_outputs)
        history_df.select(
            fn=_on_select,
            inputs=[call_ids_state],
            outputs=_detail_outputs,
        )
        delete_btn.click(
            fn=_on_delete,
            inputs=[selected_id_state],
            outputs=_load_outputs,
        )
