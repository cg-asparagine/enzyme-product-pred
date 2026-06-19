"""Assemble a PDF evaluation report (metadata + metrics + plots) with ReportLab.

Used both by the runner (right after evaluation) and as a CLI to re-render a
report from a run directory's saved ``metadata.json`` / ``metrics.json`` /
``plots/``::

    python -m epp_core.report.generator --run experiments/<run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from epp_core.eval.types import EvalArtifacts, MetricResult
from epp_core.io import read_json
from epp_core.report.components import CELL as _CELL
from epp_core.report.components import GRID_CELL as _GRID_CELL
from epp_core.report.components import HEADER_BLUE as _HEADER_BLUE
from epp_core.report.components import escape_xml as _escape
from epp_core.report.components import flatten as _flatten
from epp_core.report.components import image_flowable as _image_flowable
from epp_core.report.components import kv_table as _kv_table


def _metrics_table(metrics: list[MetricResult]) -> Table:
    rows = [
        [
            Paragraph("<b>Metric</b>", _CELL),
            Paragraph("<b>Value</b>", _CELL),
            Paragraph("<b>Higher is better</b>", _CELL),
        ]
    ]
    for m in metrics:
        value = f"{m.value:.4f}" if isinstance(m.value, float) else _escape(m.value)
        rows.append(
            [
                Paragraph(_escape(m.name), _CELL),
                Paragraph(value, _CELL),
                Paragraph("yes" if m.higher_is_better else "no", _CELL),
            ]
        )
    table = Table(rows, colWidths=[2.7 * inch, 2.0 * inch, 2.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _grid_table(rows: list[list]) -> Table:
    """Render a multi-column grid whose first row is the header.

    Used for wide tables such as a per-EC metrics matrix (rows = each EC class,
    columns = the individual metrics). The label column gets a little more room;
    the remaining columns share the rest evenly.
    """
    ncols = max(len(r) for r in rows)
    cells = [[*(_escape(c) for c in r), *([""] * (ncols - len(r)))] for r in rows]
    header, *body = cells
    table_rows = [[Paragraph(f"<b>{c}</b>", _GRID_CELL) for c in header]]
    table_rows += [[Paragraph(c, _GRID_CELL) for c in row] for row in body]

    usable, first = 7.1 * inch, 1.0 * inch
    rest = (usable - first) / (ncols - 1) if ncols > 1 else usable
    table = Table(table_rows, colWidths=[first, *([rest] * (ncols - 1))], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (0, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def render_report(metadata: dict, artifacts: EvalArtifacts, out_path: str | Path) -> Path:
    """Render a full PDF report and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()

    def heading(text: str) -> Paragraph:
        return Paragraph(_escape(text), styles["Heading2"])

    story: list[Any] = [
        Paragraph(
            _escape(f"Evaluation report — {metadata.get('model_name', '?')}"), styles["Title"]
        ),
        Paragraph(
            _escape(
                f"Task: {metadata.get('task_type', '?')}  •  Run: {metadata.get('run_id', '?')}"
            ),
            styles["Normal"],
        ),
        Paragraph(
            _escape(f"Model timestamp (UTC): {metadata.get('timestamp_utc', '?')}"),
            styles["Normal"],
        ),
        Spacer(1, 0.2 * inch),
    ]

    architecture = metadata.get("architecture", {}) or {}
    run_info = {
        "model_name": metadata.get("model_name", ""),
        "task_type": metadata.get("task_type", ""),
        # Pretrained checkpoint the model was fine-tuned from (e.g. a MolT5 or
        # ReactionT5 hub id), surfaced prominently here and also in Architecture.
        "pretrained_checkpoint": architecture.get("base_checkpoint", "—"),
        "run_id": metadata.get("run_id", ""),
        "dataset_id": metadata.get("dataset_id", ""),
        "dataset_hash": metadata.get("dataset_hash", ""),
        "notes": metadata.get("notes", ""),
    }
    story += [heading("Run information"), _kv_table(run_info), Spacer(1, 0.15 * inch)]

    environment = {
        **_flatten(metadata.get("git", {}), "git"),
        **_flatten(metadata.get("device", {}), "device"),
        **_flatten(metadata.get("libraries", {}), "lib"),
    }
    story += [heading("Environment"), _kv_table(environment), Spacer(1, 0.15 * inch)]

    story += [
        heading("Hyperparameters"),
        _kv_table(_flatten(metadata.get("hyperparameters", {}))),
        Spacer(1, 0.15 * inch),
    ]

    architecture = _flatten(metadata.get("architecture", {}))
    if architecture:
        story += [heading("Architecture"), _kv_table(architecture), Spacer(1, 0.15 * inch)]

    train_config = _flatten(metadata.get("train_config", {}))
    if train_config:
        story += [
            heading("Training configuration"),
            _kv_table(train_config),
            Spacer(1, 0.15 * inch),
        ]

    story += [heading("Metrics"), _metrics_table(artifacts.metrics), Spacer(1, 0.15 * inch)]

    for name, rows in (artifacts.tables or {}).items():
        rows = [r for r in rows if r]
        if not rows:
            continue
        # A 2-column table is a key/value list; anything wider is a grid whose
        # first row is the header.
        if max(len(r) for r in rows) > 2:
            flowable: Any = _grid_table(rows)
        else:
            pairs = {r[0]: r[1] for r in rows if len(r) >= 2}
            if not pairs:
                continue
            flowable = _kv_table(pairs)
        story += [heading(name.replace("_", " ").title()), flowable, Spacer(1, 0.15 * inch)]

    existing_plots = [p for p in artifacts.plot_paths if Path(p).exists()]
    if existing_plots:
        story.append(PageBreak())
        story.append(heading("Plots"))
        for plot_path in existing_plots:
            story.append(_image_flowable(plot_path))
            story.append(Spacer(1, 0.2 * inch))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        title=f"Evaluation report — {metadata.get('model_name', '')}",
    )
    doc.build(story)
    return out_path


def _load_run(run_dir: str | Path) -> tuple[dict, EvalArtifacts]:
    run_dir = Path(run_dir)
    metadata = read_json(run_dir / "metadata.json")
    metrics = [MetricResult(**m) for m in read_json(run_dir / "metrics.json")]
    plot_paths = sorted(str(p) for p in (run_dir / "plots").glob("*.png"))
    tables_path = run_dir / "tables.json"
    tables = read_json(tables_path) if tables_path.exists() else {}
    return metadata, EvalArtifacts(metrics=metrics, plot_paths=plot_paths, tables=tables)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-render a run's PDF report from saved JSON.")
    parser.add_argument("--run", required=True, help="Path to an experiments/<run_id> directory")
    args = parser.parse_args()
    metadata, artifacts = _load_run(args.run)
    out = render_report(metadata, artifacts, Path(args.run) / "report.pdf")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
