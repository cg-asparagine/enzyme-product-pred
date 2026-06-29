"""Evaluate ESM2-150M-ReactionT5 -> experiments/<run_id>/report.pdf.

Usage: `just evaluate ESM2-150M-ReactionT5 [--smoke] [--plus] [--model-dir <path>]`.

To evaluate the trained model against a *different* benchmark (not the training
set), pass `--dataset-dir <processed-dir> [--dataset-id <id>]`; add
`--shuffle-enzymes` for a control run that permutes the enzyme conditioning across
test rows (to test whether the correct enzyme actually helps).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataclasses import replace

    from esm2_150m_reactiont5.config import SMOKE_CONFIG, TrainConfig, with_plus_dataset
    from esm2_150m_reactiont5.pipeline import evaluate_run

    parser = argparse.ArgumentParser(description="Evaluate ESM2-150M-ReactionT5.")
    parser.add_argument("--smoke", action="store_true", help="tiny config for a quick CPU/MPS run")
    parser.add_argument(
        "--plus",
        action="store_true",
        help="evaluate the EnzymeMap_with_seq_plus model (checkpoints-plus, plus test split)",
    )
    parser.add_argument("--model-dir", default=None, help="trained model dir (defaults to config)")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="evaluate against this processed dataset dir instead of the training set",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="dataset id recorded in metadata (defaults to the dataset folder name)",
    )
    parser.add_argument(
        "--shuffle-enzymes",
        action="store_true",
        help="control: permute enzyme conditioning across test rows (fixed seed)",
    )
    args = parser.parse_args()

    config = SMOKE_CONFIG if args.smoke else TrainConfig()
    if args.plus:
        config = with_plus_dataset(config)
    if args.dataset_dir is not None:
        dataset_id = args.dataset_id or Path(args.dataset_dir).resolve().parent.name
        config = replace(config, dataset_dir=args.dataset_dir, dataset_id=dataset_id)
    elif args.dataset_id is not None:
        config = replace(config, dataset_id=args.dataset_id)

    run_dir = evaluate_run(config, model_dir=args.model_dir, shuffle_enzymes=args.shuffle_enzymes)
    print(f"Wrote report to {run_dir}")


if __name__ == "__main__":
    main()
