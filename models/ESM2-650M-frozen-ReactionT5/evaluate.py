"""Evaluate ESM2-650M-frozen-ReactionT5 -> experiments/<run_id>/report.pdf.

Usage: `just evaluate ESM2-650M-frozen-ReactionT5 [--smoke] [--model-dir <path>]`.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from esm2_reactiont5.config import SMOKE_CONFIG, TrainConfig
    from esm2_reactiont5.pipeline import evaluate_run

    parser = argparse.ArgumentParser(description="Evaluate ESM2-650M-frozen-ReactionT5.")
    parser.add_argument("--smoke", action="store_true", help="tiny config for a quick CPU/MPS run")
    parser.add_argument("--model-dir", default=None, help="trained model dir (defaults to config)")
    args = parser.parse_args()

    config = SMOKE_CONFIG if args.smoke else TrainConfig()
    run_dir = evaluate_run(config, model_dir=args.model_dir)
    print(f"Wrote report to {run_dir}")


if __name__ == "__main__":
    main()
