#!/usr/bin/env python3
"""Run the locked CINIC-10 MixAug architecture comparison search.

The shared search implementation does not load image data. DeepFedNAS uses
its structural objective, while Range-matched Random uses the predictor trained on
the official CINIC-10 validation directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "scripts/search"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints/cinic10_alpha100_c0.4"
PREDICTOR_DIR = REPO_ROOT / "outputs/cinic10/predictor"
OUTPUT_DIR = REPO_ROOT / "outputs/cinic10/search"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        default=CHECKPOINT_DIR / "range_matched_random/seed0/best_checkpoint_supernet.pt",
    )
    parser.add_argument(
        "--deepfednas-checkpoint", type=Path,
        default=CHECKPOINT_DIR / "deepfednas/seed0/best_checkpoint_supernet.pt",
    )
    parser.add_argument("--predictor-dir", type=Path, default=PREDICTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--baseline-method-label",
        choices=("Range-matched Random",),
        default="Range-matched Random",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_checkpoint = args.baseline_checkpoint.resolve()
    deepfednas_checkpoint = args.deepfednas_checkpoint.resolve()
    predictor_dir = args.predictor_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = [
        baseline_checkpoint,
        deepfednas_checkpoint,
        predictor_dir / "predictor_model.pt",
        predictor_dir / "predictor_dataset.csv",
        predictor_dir / "predictor_provenance.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    sys.path.insert(0, str(EVALUATION_DIR))
    import locked_search_impl as implementation

    implementation.BASELINE_CHECKPOINT = baseline_checkpoint
    implementation.DEEP_CHECKPOINT = deepfednas_checkpoint
    implementation.BASELINE_PREDICTOR = predictor_dir / "predictor_model.pt"
    implementation.BASELINE_PREDICTOR_DATASET = predictor_dir / "predictor_dataset.csv"
    implementation.REPO_ROOT = REPO_ROOT

    sys.argv = [
        "run_cinic10_locked_search.py",
        "--output-dir", str(output_dir),
        "--seeds", "42", "43", "44", "45", "46",
        "--methods", "DeepFedNAS", args.baseline_method_label,
        "--deep-stem-index", "9",
        "--population-size", "256",
        "--generations", "512",
        "--mutation-probability", "0.3",
        "--parent-ratio", "0.25",
    ]
    implementation.main()

    provenance_path = output_dir / "search_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    predictor_provenance = predictor_dir / "predictor_provenance.json"
    provenance["dataset"] = "CINIC-10"
    provenance["selection_split"] = (
        f"Official CINIC-10 validation directory for the {args.baseline_method_label} "
        "predictor; no image data loaded by search"
    )
    provenance["baseline_predictor"]["split_audit"] = (
        "Predictor trained from a fixed 10,000-image subset of the official "
        "CINIC-10 validation directory with deterministic transforms; test "
        "data was not loaded."
    )
    provenance.update({
        "baseline_predictor_provenance": str(predictor_provenance.resolve()),
        "baseline_predictor_provenance_sha256": sha256_file(predictor_provenance),
        "search_driver": str(Path(__file__).resolve()),
        "search_driver_sha256": sha256_file(Path(__file__).resolve()),
    })
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"CINIC-10 locked architecture manifest: {output_dir / 'locked_architectures.csv'}")


if __name__ == "__main__":
    main()
