#!/usr/bin/env python3
"""Verify that a baseline checkpoint directory completed its planned rounds."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-rounds", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_rounds <= 0:
        raise ValueError("--expected-rounds must be positive")

    best_checkpoint = args.checkpoint_dir / "best_checkpoint_supernet.pt"
    latest_checkpoint = args.checkpoint_dir / "latest_round_model.pt"
    if not best_checkpoint.is_file() or not latest_checkpoint.is_file():
        raise FileNotFoundError(
            "Predictor collection requires both best_checkpoint_supernet.pt "
            f"and latest_round_model.pt in {args.checkpoint_dir}"
        )

    latest = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
    best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    next_round = latest.get("next_round")
    completed_round = latest.get("completed_round")
    if next_round != args.expected_rounds:
        raise RuntimeError(
            f"Training is not complete: {latest_checkpoint} records "
            f"next_round={next_round}, expected {args.expected_rounds}"
        )
    if completed_round is not None and completed_round != args.expected_rounds - 1:
        raise RuntimeError(
            f"Inconsistent completion state: completed_round={completed_round}, "
            f"expected {args.expected_rounds - 1}"
        )
    for field in ("checkpoint_format_version", "model_class_name", "resume_config_digest"):
        latest_value = latest.get(field)
        best_value = best.get(field)
        if latest_value is not None and best_value is not None and latest_value != best_value:
            raise RuntimeError(
                f"Best/latest checkpoint mismatch for {field}: "
                f"best={best_value!r}, latest={latest_value!r}"
            )

    print(
        f"Training complete: {args.checkpoint_dir} "
        f"(next_round={next_round}, completed_round={completed_round})"
    )


if __name__ == "__main__":
    main()
