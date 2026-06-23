"""End-to-end pipeline proof: synthetic inputs -> metrics -> plots -> PDF report.

No model is involved; this validates the full epp_core.runner spine that every
model's evaluate.py will call.
"""

from epp_core.eval.types import GenerativeEvalInputs, TaskType
from epp_core.io import read_json
from epp_core.metadata.capture import ExperimentMetadata
from epp_core.runner import evaluate_model


def test_generative_run_produces_artifacts(tmp_path):
    inputs = GenerativeEvalInputs(
        references=[["CCO"], ["c1ccccc1"]],
        predictions=[["CCO", "CCC"], ["CCC", "c1ccccc1"]],
        train_smiles={"CCO"},
    )
    meta = ExperimentMetadata(
        model_name="gen-smoke",
        task_type=TaskType.GENERATIVE,
        hyperparameters={"num_beams": 5},
        dataset_id="enzymemap-brenda2023-v1",
    )
    run_dir = evaluate_model(inputs=inputs, metadata=meta, run_root=tmp_path)

    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "report.pdf").read_bytes()[:5] == b"%PDF-"
    assert list((run_dir / "plots").glob("*.png"))

    # metrics.json round-trips and carries the scalar (non-per-k) metrics.
    metric_names = {m["name"] for m in read_json(run_dir / "metrics.json")}
    assert {"validity", "per_molecule_validity", "uniqueness", "mean_tanimoto_top1"} <= metric_names

    # The per-k family (accuracy / exact-set-match / precision / sensitivity / F1)
    # is recorded as a matrix table, one row per metric across the k columns.
    tables = read_json(run_dir / "tables.json")
    assert "top_k_metrics" in tables
    row_labels = {row[0] for row in tables["top_k_metrics"][1:]}
    assert {"Precision", "F1", "Sensitivity (recall)"} <= row_labels

    # metadata.json records the run id and resolved task type.
    saved = read_json(run_dir / "metadata.json")
    assert saved["run_id"] == run_dir.name
    assert saved["task_type"] == "generative"
