import json

from epp_core.eval.types import TaskType
from epp_core.metadata.capture import ExperimentMetadata, capture_libraries


def test_metadata_serializes_to_json():
    meta = ExperimentMetadata(
        model_name="test-model",
        task_type=TaskType.GENERATIVE,
        hyperparameters={"lr": 2e-5, "batch_size": 32},
        dataset_id="enzymemap-brenda2023-v1",
    )
    d = meta.to_dict()
    assert d["task_type"] == "generative"
    assert d["model_name"] == "test-model"
    assert "commit" in d["git"]
    assert "platform" in d["device"]
    # The whole dict must be JSON-serializable.
    restored = json.loads(json.dumps(d))
    assert restored["hyperparameters"]["batch_size"] == 32


def test_capture_libraries_reports_presence():
    libs = capture_libraries(["pytest", "definitely-not-a-real-package-xyz"])
    assert libs["pytest"] is not None
    assert libs["definitely-not-a-real-package-xyz"] is None
