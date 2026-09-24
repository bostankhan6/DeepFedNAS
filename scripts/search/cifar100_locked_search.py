#!/usr/bin/env python3
"""Run the multi-seed CIFAR-100 architecture search.

The search phase loads no CIFAR-100 image or label data. DeepFedNAS uses the
structural fitness objective, while the baseline uses the predictor built from
the fixed CIFAR-100 validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "scripts/search"
BASELINE_CHECKPOINT = REPO_ROOT / "checkpoints/cifar100_alpha100_c0.4/range_matched_random/seed0/best_checkpoint_supernet.pt"
DEEP_CHECKPOINT = REPO_ROOT / "checkpoints/cifar100_alpha100_c0.4/deepfednas/seed0/best_checkpoint_supernet.pt"
PREDICTOR_DIR = REPO_ROOT / "outputs/cifar100/predictor"
OUTPUT_DIR = REPO_ROOT / "outputs/cifar100/search"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_wrapper_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument("--deepfednas-checkpoint", type=Path, default=DEEP_CHECKPOINT)
    parser.add_argument("--predictor-dir", type=Path, default=PREDICTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--baseline-method-label",
        choices=("Range-matched Random",),
        default="Range-matched Random",
        help="Name of the range-matched random control.",
    )
    parser.add_argument("-h", "--help", action="store_true")
    args, search_args = parser.parse_known_args()
    if args.help:
        parser.print_help()
        print("\nAdditional search options are forwarded to the shared search implementation.")
        return args, None
    return args, search_args


def main() -> None:
    wrapper_args, search_args = parse_wrapper_args()
    if wrapper_args.help:
        return
    baseline_checkpoint = wrapper_args.baseline_checkpoint.resolve()
    deepfednas_checkpoint = wrapper_args.deepfednas_checkpoint.resolve()
    predictor_dir = wrapper_args.predictor_dir.resolve()
    output_dir = wrapper_args.output_dir.resolve()
    if not (predictor_dir / "predictor_model.pt").is_file():
        raise FileNotFoundError(
            "Build the CIFAR-100 validation predictor first with "
            "scripts/search/collect_predictor.sh"
        )
    if not baseline_checkpoint.is_file() or not deepfednas_checkpoint.is_file():
        raise FileNotFoundError(
            "Both CIFAR-100 clean checkpoints are required. "
            "Complete the alpha=100 training runs first."
        )

    sys.path.insert(0, str(EVALUATION_DIR))
    import locked_search_impl as implementation

    implementation.BASELINE_CHECKPOINT = baseline_checkpoint
    implementation.DEEP_CHECKPOINT = deepfednas_checkpoint
    implementation.BASELINE_PREDICTOR = predictor_dir / "predictor_model.pt"
    implementation.BASELINE_PREDICTOR_DATASET = predictor_dir / "predictor_dataset.csv"
    # The classifier contributes to MAC accounting, so CIFAR-100 has slightly
    # different exact operational endpoints from the 10-class datasets.
    implementation.MAC_BINS = (
        ("0.458-0.95B", 458_330_112.0, 0.95e9),
        ("0.95-1.45B", 0.95e9, 1.45e9),
        ("1.45-2.45B", 1.45e9, 2.45e9),
        ("2.45-3.403B", 2.45e9, 3_403_554_816.0),
    )

    original_loader = implementation.load_arch_params

    def load_arch_params_with_defaults(path):
        arch_params = original_loader(path)
        if arch_params.get("supernet_rho0_constraint") is None:
            arch_params["supernet_rho0_constraint"] = 2.0
        if arch_params.get("supernet_effectiveness_fitness_weight") is None:
            arch_params["supernet_effectiveness_fitness_weight"] = 100.0
        return arch_params

    implementation.load_arch_params = load_arch_params_with_defaults
    implementation.REPO_ROOT = REPO_ROOT

    argv = [
        "run_locked_search.py",
        "--output-dir", str(output_dir),
        "--seeds", "42", "43", "44", "45", "46",
        "--methods", "DeepFedNAS", wrapper_args.baseline_method_label,
        "--deep-stem-index", "9",
        "--population-size", "256",
        "--generations", "512",
        "--mutation-probability", "0.3",
        "--parent-ratio", "0.25",
    ]
    argv.extend(search_args)
    sys.argv = argv
    implementation.main()

    provenance_path = output_dir / "search_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    split_manifest = REPO_ROOT / "configs/splits/cifar100/split_manifest_seed0.json"
    val_pickle = REPO_ROOT / "data/cifar100/val.pkl"
    baseline_predictor = provenance.get("baseline_predictor")
    if baseline_predictor is not None:
        baseline_predictor["split_audit"] = (
            "CIFAR-100 predictor trained from all 5,000 samples in the fixed "
            "data/cifar100/val.pkl artifact; predictor held-out rows are an "
            "internal architecture-level split and official test data was "
            "not loaded."
        )
    provenance.update({
        "dataset": "CIFAR-100",
        "selection_split": "Fixed data/cifar100/val.pkl for the baseline predictor; no image data loaded by search",
        "test_data_loaded": False,
        "clean_split_manifest": str(split_manifest.resolve()),
        "clean_split_manifest_sha256": sha256_file(split_manifest),
        "validation_pickle": str(val_pickle.resolve()),
        "validation_pickle_sha256": sha256_file(val_pickle),
        "baseline_predictor_provenance": str((predictor_dir / "predictor_provenance.json").resolve()),
        "baseline_predictor_provenance_sha256": sha256_file(predictor_dir / "predictor_provenance.json"),
        "search_driver": str(Path(__file__).resolve()),
        "search_driver_sha256": sha256_file(Path(__file__).resolve()),
    })
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"CIFAR-100 locked architecture manifest: {output_dir / 'locked_architectures.csv'}")


if __name__ == "__main__":
    main()
