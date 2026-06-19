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

    # metrics.json round-trips and includes the reaction-specific metrics.
    metric_names = {m["name"] for m in read_json(run_dir / "metrics.json")}
    assert {"top_1_accuracy", "exact_set_match_top_1", "per_molecule_validity"} <= metric_names

    # metadata.json records the run id and resolved task type.
    saved = read_json(run_dir / "metadata.json")
    assert saved["run_id"] == run_dir.name
    assert saved["task_type"] == "generative"
