from __future__ import annotations

import gradio as gr


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _load_metrics() -> tuple:
    """Return (summary_md, audit_rows, langsmith_md) for the Observability tab."""
    try:
        from sqlalchemy import func, select

        from src.database.database import get_session
        from src.database.models import AuditLog, CallRecord

        with get_session() as session:
            # --- Call counts ---
            total: int = session.execute(select(func.count(CallRecord.id))).scalar_one()
            completed: int = session.execute(
                select(func.count(CallRecord.id)).where(CallRecord.status == "completed")
            ).scalar_one()
            failed: int = session.execute(
                select(func.count(CallRecord.id)).where(CallRecord.status == "failed")
            ).scalar_one()
            flagged: int = session.execute(
                select(func.count(CallRecord.id)).where(CallRecord.status == "supervisor_review")
            ).scalar_one()

            # --- Average QA score ---
            avg_qa_raw = session.execute(
                select(func.avg(CallRecord.overall_qa_score))
            ).scalar_one()
            avg_qa: float = avg_qa_raw or 0.0

            # --- Compliance flags across all reports ---
            records = session.execute(select(CallRecord)).scalars().all()
            total_flags = sum(
                len((r.report_json or {}).get("qa_scores", {}).get("compliance_flags", []))
                for r in records
            )

            # --- 20 most recent audit events ---
            audit_rows_raw = session.execute(
                select(AuditLog).order_by(AuditLog.created_at.desc()).limit(20)
            ).scalars().all()

    except Exception as exc:
        return (
            f"**DB unavailable:** {exc}",
            [],
            _langsmith_md(),
        )

    success_rate = (completed / total * 100) if total else 0.0

    summary_md = (
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total Calls | **{total}** |\n"
        f"| Completed | **{completed}** |\n"
        f"| Failed | **{failed}** |\n"
        f"| Supervisor Review | **{flagged}** |\n"
        f"| Success Rate | **{success_rate:.1f}%** |\n"
        f"| Avg QA Score | **{avg_qa:.2f} / 5.00** |\n"
        f"| Total Compliance Flags | **{total_flags}** |"
    )

    audit_rows = []
    for row in audit_rows_raw:
        ts = row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else "—"
        details = str(row.details) if row.details else "—"
        audit_rows.append([ts, row.call_id[:8] + "…", row.action, details])

    return summary_md, audit_rows, _langsmith_md()


def _langsmith_md() -> str:
    try:
        from src.config.loader import get_langsmith_status
        status = get_langsmith_status()
    except Exception:
        return "_LangSmith status unavailable._"

    if status["enabled"]:
        return f"**LangSmith Tracing:** Enabled — [View project]({status['url']})"
    return "**LangSmith Tracing:** Disabled (set `LANGSMITH_API_KEY` to enable)"


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_observability_tab() -> None:
    with gr.Tab("Observability"):
        gr.Markdown("## Pipeline Observability")

        with gr.Row():
            refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")

        with gr.Row():
            with gr.Column(scale=1):
                metrics_md = gr.Markdown()
            with gr.Column(scale=2):
                langsmith_md = gr.Markdown()

        gr.Markdown("### Recent Audit Events (last 20)")
        audit_df = gr.Dataframe(
            headers=["Timestamp", "Call ID", "Action", "Details"],
            datatype=["str", "str", "str", "str"],
            interactive=False,
        )

        def _on_load():
            summary, rows, ls_md = _load_metrics()
            return summary, rows, ls_md

        refresh_btn.click(fn=_on_load, inputs=[], outputs=[metrics_md, audit_df, langsmith_md])
