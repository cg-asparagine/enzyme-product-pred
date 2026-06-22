from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.slow
@pytest.mark.network
def test_train_eval_smoke(tmp_path):
    """End-to-end on a tiny synthetic dataset (CPU): embed -> train -> generate -> PDF.

    Downloads ESM-2 650M + ReactionT5; deselected from `just check`, run with
    `just test-slow`.
    """
    from esm2_reactiont5.config import SMOKE_CONFIG
    from esm2_reactiont5.pipeline import evaluate_run, train_model

    def rows(split: str) -> list[dict]:
        return [
            {
                "reaction_id": i,
                "reactant_smiles": "CCO" if i % 2 == 0 else "CCCO",
                "product_smiles": "CC=O" if i % 2 == 0 else "CCC=O",
                "uniprot_id": f"P{i % 2}",
                "sequence": "MKTAYIAKQR" if i % 2 == 0 else "MAAAAGGGTT",
                "split": split,
            }
            for i in range(6)
        ]

    df = pd.DataFrame(rows("train") + rows("valid") + rows("test"))
    processed = tmp_path / "processed"
    processed.mkdir()
    df.to_parquet(processed / "reactions.parquet", index=False)

    config = replace(
        SMOKE_CONFIG,
        dataset_dir=str(processed),
        embedding_cache=str(tmp_path / "emb.npz"),
        output_dir=str(tmp_path / "ckpt"),
        max_train_samples=6,
        max_eval_samples=4,
    )
    train_model(config)
    run_dir = evaluate_run(config, run_root=str(tmp_path / "experiments"))

    assert (Path(run_dir) / "report.pdf").read_bytes()[:5] == b"%PDF-"
    assert (Path(run_dir) / "metrics.json").exists()
    assert (Path(run_dir) / "metadata.json").exists()
