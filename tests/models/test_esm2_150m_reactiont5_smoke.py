from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.slow
@pytest.mark.network
def test_train_eval_smoke(tmp_path):
    """End-to-end on a tiny synthetic dataset (CPU): train (ESM in the graph) ->
    generate -> PDF. Uses the 8M ESM + ReactionT5 so it stays small; deselected from
    `just check`, run with `just test-slow`.
    """
    from esm2_150m_reactiont5.config import SMOKE_CONFIG
    from esm2_150m_reactiont5.pipeline import evaluate_run, train_model

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
        output_dir=str(tmp_path / "ckpt"),
        max_train_samples=6,
        max_eval_samples=4,
    )
    train_model(config)
    run_dir = evaluate_run(config, run_root=str(tmp_path / "experiments"))

    assert (Path(run_dir) / "report.pdf").read_bytes()[:5] == b"%PDF-"
    assert (Path(run_dir) / "metrics.json").exists()
    assert (Path(run_dir) / "metadata.json").exists()
    # The saved dir is self-contained: ESM + its tokenizer live under esm/.
    assert (Path(config.output_dir) / "esm" / "config.json").exists()


@pytest.mark.slow
@pytest.mark.network
def test_gradient_checkpointing_path_trains_esm():
    """The real-config default (`gradient_checkpointing=True`) is the headline memory
    lever: confirm it activates and that ESM still receives gradients through the
    checkpointed forward. Uses the 8M ESM so it stays tiny.
    """
    import torch
    from esm2_150m_reactiont5.model import ReactionT5WithTrainableProtein

    model = ReactionT5WithTrainableProtein.from_base(
        "sagawa/ReactionT5v2-forward",
        "facebook/esm2_t6_8M_UR50D",
        esm_dim=320,
        gradient_checkpointing=True,
    )
    model.train()
    assert model.esm.is_gradient_checkpointing
    assert model.esm.config.use_cache is False  # incompatible with checkpointing

    b, length = 2, 6
    out = model(
        input_ids=torch.randint(5, 200, (b, length)),
        attention_mask=torch.ones(b, length, dtype=torch.long),
        esm_input_ids=torch.randint(4, 30, (b, 8)),
        esm_attention_mask=torch.ones(b, 8, dtype=torch.long),
        labels=torch.randint(5, 200, (b, 4)),
    )
    out.loss.backward()
    esm_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.esm.parameters()
    )
    assert esm_has_grad  # ESM is genuinely fine-tuned, not just along for the ride
    assert model.protein_proj.weight.grad is not None
