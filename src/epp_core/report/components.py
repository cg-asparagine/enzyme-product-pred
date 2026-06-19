"""Reusable ReportLab building blocks shared across PDF reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CELL = ParagraphStyle("cell", fontSize=8, leading=10)
# Smaller cell used for wide, many-column grid tables so they fit the page width.
GRID_CELL = ParagraphStyle("grid_cell", fontSize=7, leading=8)
HEADER_BLUE = colors.HexColor("#3b7dd8")


def escape_xml(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted keys; join lists into a string."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, dotted))
    elif isinstance(obj, (list, tuple)):
        out[prefix] = ", ".join(str(x) for x in obj) if obj else "[]"
    else:
        out[prefix] = obj
    return out


def kv_table(data: dict[str, Any], col_widths: tuple[float, float] = (2.3, 4.4)) -> Table:
    """A two-column key/value table (widths in inches)."""
    rows = [
        [Paragraph(f"<b>{escape_xml(k)}</b>", CELL), Paragraph(escape_xml(v), CELL)]
        for k, v in data.items()
    ]
    if not rows:
        rows = [[Paragraph("(none)", CELL), Paragraph("", CELL)]]
    table = Table(rows, colWidths=[w * inch for w in col_widths])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
            ]
        )
    )
    return table


def image_flowable(path: str | Path, max_width_in: float = 5.5) -> Image:
    width_px, height_px = ImageReader(str(path)).getSize()
    width = max_width_in * inch
    height = width * (height_px / width_px)
    return Image(str(path), width=width, height=height)


def render_pdf(
    out_path: str | Path,
    title: str,
    subtitle_lines: Sequence[str],
    sections: Sequence[tuple[str, Sequence[Any]]],
) -> Path:
    """Assemble a simple titled report: subtitle lines, then (heading, flowables) sections."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()

    story: list[Any] = [Paragraph(escape_xml(title), styles["Title"])]
    story += [Paragraph(escape_xml(line), styles["Normal"]) for line in subtitle_lines]
    story.append(Spacer(1, 0.2 * inch))
    for heading, flowables in sections:
        story.append(Paragraph(escape_xml(heading), styles["Heading2"]))
        story.extend(flowables)
        story.append(Spacer(1, 0.15 * inch))

    SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        title=title,
    ).build(story)
    return out_path
