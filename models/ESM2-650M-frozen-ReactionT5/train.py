"""Train ESM2-650M-frozen-ReactionT5. Usage: `just train ESM2-650M-frozen-ReactionT5 [--smoke]`."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from esm2_reactiont5.config import SMOKE_CONFIG, TrainConfig
    from esm2_reactiont5.pipeline import train_model

    parser = argparse.ArgumentParser(description="Train ESM2-650M-frozen-ReactionT5.")
    parser.add_argument("--smoke", action="store_true", help="tiny config for a quick CPU/MPS run")
    args = parser.parse_args()

    config = SMOKE_CONFIG if args.smoke else TrainConfig()
    out = train_model(config)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
