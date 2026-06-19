from pathlib import Path

from epp_core.eval.types import EvalArtifacts, MetricResult
from epp_core.report.generator import render_report

_METADATA = {
    "model_name": "demo-model",
    "task_type": "generative",
    "run_id": "20260615T000000Z__demo__abc1234",
    "dataset_id": "enzymemap-brenda2023-v1",
    "dataset_hash": "deadbeef",
    "notes": "",
    "hyperparameters": {"num_beams": 5, "max_length": 200},
    "architecture": {"base_checkpoint": "reaction-t5", "num_parameters": 220_000_000},
    "train_config": {"epochs": 5},
    "git": {"commit": "abc1234", "branch": "main", "dirty": False},
    "device": {"platform": "macOS", "device_name": "cpu"},
    "libraries": {"torch": "2.2.0", "transformers": "4.44.0"},
    "timestamp_utc": "2026-06-15T00:00:00+00:00",
}


def test_render_report_creates_valid_pdf(tmp_path):
    artifacts = EvalArtifacts(
        metrics=[
            MetricResult("top_1_accuracy", 0.62),
            MetricResult("exact_set_match_top_1", 0.41),
        ]
    )
    out = render_report(_METADATA, artifacts, tmp_path / "report.pdf")
    data = Path(out).read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 1000


def test_render_report_with_grid_table(tmp_path):
    # A wider-than-2-column table renders as a grid (first row = header).
    artifacts = EvalArtifacts(
        metrics=[MetricResult("top_1_accuracy", 0.6)],
        tables={
            "per_ec_metrics": [
                ["EC class", "N", "Top-1", "Exact-set"],
                ["1 (oxidoreductases)", "1200", "0.640", "0.420"],
                ["2 (transferases)", "980", "0.610", "0.400"],
            ]
        },
    )
    out = render_report(_METADATA, artifacts, tmp_path / "report.pdf")
    assert Path(out).read_bytes()[:5] == b"%PDF-"


def test_render_report_with_plot(tmp_path):
    # Create a tiny PNG via the plot helpers and confirm it embeds without error.
    from epp_core.plots import save_figure
    from epp_core.plots.generative import plot_top_k_accuracy

    plot_path = save_figure(
        plot_top_k_accuracy((1, 5, 10), [0.4, 0.6, 0.7]),
        tmp_path / "plots" / "top_k_accuracy.png",
    )
    artifacts = EvalArtifacts(metrics=[MetricResult("top_1_accuracy", 0.4)], plot_paths=[plot_path])
    out = render_report(_METADATA, artifacts, tmp_path / "report.pdf")
    assert Path(out).read_bytes()[:5] == b"%PDF-"
