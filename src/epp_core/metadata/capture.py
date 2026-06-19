"""Experiment metadata: the reproducibility record attached to every eval run.

:class:`ExperimentMetadata` records what was run (model, task, hyperparameters,
architecture, dataset) and the environment it ran in (git commit, library
versions, device). It serializes to ``metadata.json`` and feeds the PDF report.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from epp_core.eval.types import TaskType

_DEFAULT_TRACKED_LIBS: tuple[str, ...] = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "rdkit",
    "scikit-learn",
    "numpy",
    "pandas",
    "matplotlib",
    "reportlab",
)


def _run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def capture_git() -> dict:
    """Capture the current git commit, branch, and whether the tree is dirty."""
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run_git(["status", "--porcelain"])),
    }


def capture_libraries(packages: Sequence[str] = _DEFAULT_TRACKED_LIBS) -> dict:
    """Map each package name to its installed version (``None`` if absent)."""
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def capture_device() -> dict:
    """Capture platform / Python / accelerator info, tolerant of missing torch."""
    info: dict = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch

        cuda = torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(mps_backend is not None and mps_backend.is_available())
        info["torch"] = torch.__version__
        info["cuda_available"] = cuda
        info["mps_available"] = mps
        if cuda:
            info["device_name"] = torch.cuda.get_device_name(0)
        else:
            info["device_name"] = "mps" if mps else "cpu"
    except Exception:  # noqa: BLE001 - torch optional / import may fail
        info["device_name"] = "cpu"
    return info


def hf_architecture(model: object, tokenizer: object, base_checkpoint: str) -> dict:
    """Capture a HuggingFace seq2seq model's architecture for the metadata record.

    Duck-typed over ``model.config`` so it works for any encoder-decoder model
    (MolT5, ReactionT5, ...) without importing transformers here.
    """
    cfg = getattr(model, "config", None)
    return {
        "base_checkpoint": base_checkpoint,
        "model_type": getattr(cfg, "model_type", None),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),  # type: ignore[attr-defined]
        "d_model": getattr(cfg, "d_model", None),
        "num_layers": getattr(cfg, "num_layers", None),
        "num_decoder_layers": getattr(cfg, "num_decoder_layers", None),
        "num_heads": getattr(cfg, "num_heads", None),
        "vocab_size": len(tokenizer),  # type: ignore[arg-type]
    }


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ExperimentMetadata:
    """Full, serializable record of one model evaluation run."""

    model_name: str
    task_type: TaskType
    hyperparameters: dict = field(default_factory=dict)
    architecture: dict = field(default_factory=dict)
    train_config: dict = field(default_factory=dict)
    dataset_id: str = ""
    dataset_hash: str = ""
    notes: str = ""
    run_id: str = ""
    git: dict = field(default_factory=capture_git)
    libraries: dict = field(default_factory=capture_libraries)
    device: dict = field(default_factory=capture_device)
    timestamp_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["task_type"] = str(self.task_type)
        return data
