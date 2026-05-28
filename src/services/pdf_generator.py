from __future__ import annotations

import io
from datetime import timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.models.schemas import CallReport

_STYLES = getSampleStyleSheet()

_TITLE = ParagraphStyle("title", parent=_STYLES["Title"], fontSize=18, spaceAfter=6)
_H2 = ParagraphStyle("h2", parent=_STYLES["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=4)
_BODY = ParagraphStyle("body", parent=_STYLES["Normal"], fontSize=10, spaceAfter=4)
_SMALL = ParagraphStyle("small", parent=_STYLES["Normal"], fontSize=9, textColor=colors.grey)

_SEVERITY_COLORS = {
    "low": colors.HexColor("#28a745"),
    "medium": colors.HexColor("#ffc107"),
    "high": colors.HexColor("#fd7e14"),
    "critical": colors.HexColor("#dc3545"),
}


def generate(report: CallReport) -> bytes:
    """
    Generate a PDF report for a CallReport and return the raw bytes.
    No file I/O is performed inside this function.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []
    _add_header(story, report)
    _add_summary_section(story, report)
    _add_qa_scorecard(story, report)
    _add_compliance_flags(story, report)
    _add_transcript_sample(story, report)

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _add_header(story: list, report: CallReport) -> None:
    analyzed = report.analyzed_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph("Call Center Intelligence Report", _TITLE))
    story.append(Paragraph(f"<b>Call ID:</b> {report.call_id}", _BODY))
    story.append(Paragraph(f"<b>File:</b> {report.filename}", _BODY))
    story.append(Paragraph(f"<b>Analyzed:</b> {analyzed}", _BODY))
    story.append(Paragraph(
        f"<b>Status:</b> {report.status.replace('_', ' ').title()}  |  "
        f"<b>Duration:</b> {report.audio_properties.duration_seconds:.0f}s  |  "
        f"<b>Format:</b> {report.audio_properties.format.upper()}",
        _BODY,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=8))


def _add_summary_section(story: list, report: CallReport) -> None:
    s = report.summary
    story.append(Paragraph("Summary", _H2))
    story.append(Paragraph(f"<b>Purpose:</b> {s.call_purpose}", _BODY))
    story.append(Paragraph(
        f"<b>Resolution:</b> {s.resolution_status.title()}  |  "
        f"<b>Sentiment:</b> {s.sentiment_trajectory}",
        _BODY,
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Key Discussion Points</b>", _BODY))
    for pt in s.key_discussion_points:
        story.append(Paragraph(f"• {pt}", _BODY))

    if s.action_items:
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Action Items</b>", _BODY))
        for ai in s.action_items:
            deadline = f" — due {ai.deadline}" if ai.deadline else ""
            story.append(Paragraph(f"• [{ai.owner}] {ai.description}{deadline}", _BODY))


def _add_qa_scorecard(story: list, report: CallReport) -> None:
    qa = report.qa_scores
    story.append(Paragraph("QA Scorecard", _H2))
    story.append(Paragraph(
        f"<b>Overall Score: {qa.overall_score:.2f} / 5.00</b>",
        ParagraphStyle("bold", parent=_BODY, fontSize=11),
    ))
    story.append(Spacer(1, 6))

    weights = {
        "professionalism": 0.15,
        "empathy": 0.20,
        "problem_resolution": 0.30,
        "compliance": 0.20,
        "clarity": 0.15,
    }

    data = [["Dimension", "Score", "Weight", "Justification"]]
    for dim in qa.dimensions:
        data.append([
            dim.name.replace("_", " ").title(),
            f"{dim.score:.1f}",
            f"{weights.get(dim.name, 0):.0%}",
            Paragraph(dim.justification[:120] + ("…" if len(dim.justification) > 120 else ""), _SMALL),
        ])

    table = Table(data, colWidths=[3.5 * cm, 1.5 * cm, 1.5 * cm, 10 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)


def _add_compliance_flags(story: list, report: CallReport) -> None:
    flags = report.qa_scores.compliance_flags
    if not flags:
        return

    story.append(Paragraph("Compliance Flags", _H2))
    data = [["Severity", "Description", "Timestamp"]]
    for flag in flags:
        sev_color = _SEVERITY_COLORS.get(flag.severity, colors.black)
        data.append([
            Paragraph(
                f'<font color="{sev_color.hexval()}">{flag.severity.upper()}</font>',
                ParagraphStyle("sev", parent=_SMALL, fontName="Helvetica-Bold"),
            ),
            Paragraph(flag.description, _SMALL),
            flag.timestamp or "—",
        ])

    table = Table(data, colWidths=[2.5 * cm, 12 * cm, 2 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)


def _add_transcript_sample(story: list, report: CallReport) -> None:
    segments = report.transcription.segments
    if not segments:
        return

    story.append(Paragraph("Transcript Sample (first 10 segments)", _H2))
    for seg in segments[:10]:
        ts = f"{int(seg.start_time // 60):02d}:{int(seg.start_time % 60):02d}"
        story.append(Paragraph(
            f"<b>[{ts}] {seg.speaker}:</b> {seg.text}",
            _SMALL,
        ))
