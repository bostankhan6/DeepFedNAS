#!/usr/bin/env python3
"""Build the baseline accuracy predictor from official CINIC-10 validation data.

This script deliberately has no test-data path or test loader.  It samples
architectures from the baseline checkpoint, evaluates them on a fixed
stratified subset of ``data/cinic10/val``, and trains an accuracy predictor.
The architecture rows, validation subset manifest, model, and metrics are
written to one provenance directory for later predictor-guided search.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from deepfednas.Server.generic_server_model import GenericServerOFA
from deepfednas.utils.subnet_cost import subnet_macs


CHECKPOINT = REPO_ROOT / "checkpoints/cinic10_alpha100_c0.4/range_matched_random/seed0/best_checkpoint_supernet.pt"
DATA_DIR = REPO_ROOT / "data/cinic10"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/cinic10/predictor"

CINIC10_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC10_STD = [0.24205776, 0.23828046, 0.25874835]

MAC_BINS = (
    ("0.458-0.95B", 458_237_952, 950_000_000),
    ("0.95-1.45B", 950_000_000, 1_450_000_000),
    ("1.45-2.45B", 1_450_000_000, 2_450_000_000),
    ("2.45-3.403B", 2_450_000_000, 3_403_370_496),
)


class AccuracyPredictor(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_features, 400), nn.ReLU(inplace=True),
            nn.Linear(400, 400), nn.ReLU(inplace=True),
            nn.Linear(400, 1),
        )

    def forward(self, inputs):
        return self.layer(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-architectures", type=int, default=10000)
    parser.add_argument(
        "--initial-dataset",
        type=Path,
        default=None,
        help=(
            "Reuse rows from an existing predictor_dataset.csv or partial CSV. "
            "Existing architectures are validated, deduplicated, and counted "
            "toward --num-architectures. The source file is never modified."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("uniform", "restricted-bin-stratified"),
        default="restricted-bin-stratified",
        help=(
            "Sample uniformly from the full architecture space, or allocate "
            "equal validation evaluations to each reported restricted MAC bin."
        ),
    )
    parser.add_argument("--validation-subset-size", type=int, default=10000)
    parser.add_argument("--subset-seed", type=int, default=20260805)
    parser.add_argument("--architecture-seed", type=int, default=20260806)
    parser.add_argument("--predictor-split-seed", type=int, default=20260807)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--predictor-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(temporary_path, index=False, lineterminator="\n")
    os.replace(temporary_path, path)


def architecture_features(arch: dict, arch_params: dict) -> np.ndarray:
    exp_choices = list(arch_params["expansion_ratio_choices"])
    exp_one_hot = np.zeros(len(arch["e"]) * len(exp_choices), dtype=np.float32)
    for index, value in enumerate(arch["e"]):
        exp_one_hot[index * len(exp_choices) + exp_choices.index(value)] = 1.0
    return np.concatenate([
        np.asarray(arch["d"], dtype=np.float32),
        exp_one_hot,
        np.asarray(arch["w_indices"], dtype=np.float32),
    ])


def load_server(checkpoint_path: Path) -> tuple[GenericServerOFA, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch_params = copy.deepcopy(checkpoint["arch_params"])
    arch_params["original_stage_base_channels"] = np.asarray(arch_params["original_stage_base_channels"])
    arch_params["expansion_ratio_choices"] = list(arch_params["expansion_ratio_choices"])
    arch_params["width_multiplier_choices"] = list(arch_params["width_multiplier_choices"])
    server = GenericServerOFA(arch_params=arch_params, sampling_method="all_random", num_cli_total=1)
    server.set_model_params(checkpoint["params"])
    del checkpoint
    return server, arch_params


def stratified_subset_indices(dataset: datasets.ImageFolder, subset_size: int, seed: int) -> list[int]:
    if subset_size <= 0 or subset_size > len(dataset):
        raise ValueError(f"Validation subset size must be in [1, {len(dataset)}]")
    labels = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == label)
        count = subset_size // len(np.unique(labels))
        if label < subset_size % len(np.unique(labels)):
            count += 1
        selected.extend(rng.choice(class_indices, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return [int(index) for index in selected]


def build_validation_loader(data_dir: Path, subset_size: int, subset_seed: int, batch_size: int, workers: int):
    val_dir = data_dir / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Official CINIC-10 validation directory not found: {val_dir}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
    ])
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    indices = stratified_subset_indices(dataset, subset_size, subset_seed)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    return loader, dataset, indices


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += int((model(images).argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    model.cpu()
    torch.cuda.empty_cache()
    return 100.0 * correct / total


def train_predictor(
    features: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    sampling_bins=None,
):
    tensors = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
    )
    train_size = int(0.9 * len(tensors))
    val_size = len(tensors) - train_size
    generator = torch.Generator().manual_seed(args.predictor_split_seed)
    train_set, heldout_set = random_split(tensors, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=args.predictor_batch_size, shuffle=True)
    heldout_loader = DataLoader(heldout_set, batch_size=args.predictor_batch_size, shuffle=False)
    model = AccuracyPredictor(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    for _ in tqdm(range(args.epochs), desc="Training validation predictor"):
        model.train()
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    actual = []
    with torch.inference_mode():
        for batch_features, batch_targets in heldout_loader:
            predictions.extend(model(batch_features.to(device)).squeeze(1).cpu().tolist())
            actual.extend(batch_targets.squeeze(1).tolist())
    predictions = np.asarray(predictions)
    actual = np.asarray(actual)
    errors = predictions - actual
    ranks_pred = np.argsort(np.argsort(predictions))
    ranks_actual = np.argsort(np.argsort(actual))
    rank_corr = float(np.corrcoef(ranks_pred, ranks_actual)[0, 1])
    metrics = {
        "train_rows": len(train_set),
        "heldout_rows": len(heldout_set),
        "heldout_mae_percent": float(np.mean(np.abs(errors))),
        "heldout_rmse_percent": float(np.sqrt(np.mean(errors ** 2))),
        "heldout_rank_correlation": rank_corr,
        "heldout_actual_min_percent": float(actual.min()),
        "heldout_actual_max_percent": float(actual.max()),
        "heldout_prediction_min_percent": float(predictions.min()),
        "heldout_prediction_max_percent": float(predictions.max()),
    }
    if sampling_bins is not None:
        heldout_groups = np.asarray(sampling_bins)[heldout_set.indices]
        metrics["heldout_by_sampling_bin"] = {}
        for group in sorted(set(heldout_groups)):
            mask = heldout_groups == group
            group_predictions = predictions[mask]
            group_actual = actual[mask]
            group_errors = group_predictions - group_actual
            metrics["heldout_by_sampling_bin"][str(group)] = {
                "rows": int(mask.sum()),
                "mae_percent": float(np.mean(np.abs(group_errors))),
                "rmse_percent": float(np.sqrt(np.mean(group_errors ** 2))),
                "rank_correlation": float(np.corrcoef(
                    np.argsort(np.argsort(group_predictions)),
                    np.argsort(np.argsort(group_actual)),
                )[0, 1]),
            }
    return model.cpu(), metrics


def width_floor_for_bin(lower: int, num_width_choices: int) -> int:
    if lower >= 2_450_000_000:
        return num_width_choices - 1
    if lower >= 1_450_000_000:
        return max(0, num_width_choices - 2)
    if lower >= 950_000_000:
        return num_width_choices // 2
    return 0


def load_initial_rows(
    path: Path,
    args: argparse.Namespace,
    sampling_plan: list[tuple],
) -> tuple[list[dict], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Initial predictor dataset does not exist: {path}")
    dataframe = pd.read_csv(path)
    required_columns = {
        "sample_index", "sampling_bin", "accuracy", "actual_macs",
        "num_parameters", "arch_d", "arch_e", "arch_w_indices",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Initial predictor dataset is missing columns: {sorted(missing_columns)}")
    feature_columns = [column for column in dataframe.columns if column.startswith("feature_")]
    if not feature_columns:
        raise ValueError("Initial predictor dataset does not contain architecture features")

    target_counts = {name: target_count for name, _, _, target_count, _ in sampling_plan}
    actual_counts = dataframe["sampling_bin"].value_counts().to_dict()
    unexpected_bins = set(actual_counts).difference(target_counts)
    if unexpected_bins:
        raise ValueError(f"Initial predictor dataset contains unexpected bins: {sorted(unexpected_bins)}")
    for bin_name, count in actual_counts.items():
        if count > target_counts[bin_name]:
            raise ValueError(
                f"Initial bin {bin_name} has {count} rows, exceeding target {target_counts[bin_name]}"
            )

    rows = dataframe.to_dict(orient="records")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        arch = {
            "d": json.loads(row["arch_d"]),
            "e": json.loads(row["arch_e"]),
            "w_indices": json.loads(row["arch_w_indices"]),
        }
        key = json.dumps(arch, sort_keys=True)
        if key in seen:
            raise ValueError(f"Initial predictor dataset has a duplicate architecture at row {index}")
        seen.add(key)
        row["sample_index"] = index

    provenance_path = path.parent / "predictor_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        expected_checkpoint_hash = provenance.get("checkpoint_sha256")
        actual_checkpoint_hash = sha256_file(args.checkpoint)
        if expected_checkpoint_hash and expected_checkpoint_hash != actual_checkpoint_hash:
            raise ValueError("Initial predictor dataset was generated from a different checkpoint")
        expected_strategy = provenance.get("sampling_strategy")
        if expected_strategy and expected_strategy != args.sampling_strategy:
            raise ValueError(
                f"Initial sampling strategy is {expected_strategy}, requested {args.sampling_strategy}"
            )
        expected_subset_seed = provenance.get("subset_seed")
        if expected_subset_seed is not None and expected_subset_seed != args.subset_seed:
            raise ValueError(
                f"Initial subset seed is {expected_subset_seed}, requested {args.subset_seed}"
            )
        expected_subset_size = provenance.get("validation_subset_sample_count")
        if expected_subset_size is not None and expected_subset_size != args.validation_subset_size:
            raise ValueError(
                f"Initial validation subset has {expected_subset_size} samples, "
                f"requested {args.validation_subset_size}"
            )
    return rows, seen


def main() -> None:
    args = parse_args()
    if args.num_architectures <= 0 or args.epochs <= 0:
        raise ValueError("--num-architectures and --epochs must be positive")
    if args.sampling_strategy == "restricted-bin-stratified" and args.num_architectures % len(MAC_BINS):
        raise ValueError("Stratified --num-architectures must be divisible by the number of MAC bins")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this workflow is intended for the GPU evaluation environment")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.initial_dataset:
        initial_dataset = args.initial_dataset.resolve()
        output_datasets = {
            (args.output_dir / "predictor_dataset.csv").resolve(),
            (args.output_dir / "predictor_dataset.partial.csv").resolve(),
        }
        if initial_dataset in output_datasets:
            raise ValueError(
                "--initial-dataset must be outside --output-dir so the source "
                "dataset cannot be overwritten"
            )
    random.seed(args.architecture_seed)
    np.random.seed(args.architecture_seed)
    torch.manual_seed(args.architecture_seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    server, arch_params = load_server(args.checkpoint)
    loader, val_dataset, subset_indices = build_validation_loader(
        args.data_dir, args.validation_subset_size, args.subset_seed,
        args.batch_size, args.workers,
    )
    if len(val_dataset) != 90_000:
        raise RuntimeError(f"Expected 90,000 official CINIC-10 validation images, found {len(val_dataset)}")

    if args.sampling_strategy == "uniform":
        sampling_plan = [("full-space-uniform", None, None, args.num_architectures, 0)]
    else:
        per_bin = args.num_architectures // len(MAC_BINS)
        sampling_plan = [
            (name, lower, upper, per_bin, width_floor_for_bin(lower, len(arch_params["width_multiplier_choices"])))
            for name, lower, upper in MAC_BINS
        ]
    rows, seen = (
        load_initial_rows(args.initial_dataset, args, sampling_plan)
        if args.initial_dataset else ([], set())
    )
    initial_architecture_count = len(rows)
    if initial_architecture_count > args.num_architectures:
        raise ValueError(
            f"Initial predictor dataset has {initial_architecture_count} rows, "
            f"exceeding target {args.num_architectures}"
        )
    started = time.time()
    proposal_draws = {}
    with tqdm(
        total=args.num_architectures,
        initial=initial_architecture_count,
        desc="Evaluating CINIC-10 validation architectures",
    ) as progress:
        for bin_name, lower, upper, target_count, width_floor in sampling_plan:
            accepted = sum(row["sampling_bin"] == bin_name for row in rows)
            draws = 0
            while accepted < target_count:
                arch = server.random_subnet_arch()
                if lower is not None:
                    arch["w_indices"] = [
                        random.randint(width_floor, len(arch_params["width_multiplier_choices"]) - 1)
                        for _ in arch["w_indices"]
                    ]
                draws += 1
                key = json.dumps(arch, sort_keys=True)
                if key in seen:
                    continue
                macs, params = subnet_macs(
                    arch["d"], arch["e"], arch["w_indices"],
                    arch_params["width_multiplier_choices"], arch_params,
                )
                if lower is not None and not lower <= macs <= upper:
                    continue
                seen.add(key)
                client = server.get_subnet(**arch, preserve_weight=True)
                accuracy = evaluate(client.model, loader, device)
                features = architecture_features(arch, arch_params)
                row = {
                    f"feature_{feature_index}": float(value)
                    for feature_index, value in enumerate(features)
                }
                row.update({
                    "sample_index": len(rows),
                    "sampling_bin": bin_name,
                    "proposal_width_floor_index": width_floor,
                    "accuracy": accuracy,
                    "actual_macs": int(macs),
                    "num_parameters": int(params),
                    "arch_d": json.dumps(arch["d"]),
                    "arch_e": json.dumps([float(value) for value in arch["e"]]),
                    "arch_w_indices": json.dumps(arch["w_indices"]),
                })
                rows.append(row)
                accepted += 1
                progress.update(1)
                del client
                if len(rows) % 100 == 0:
                    write_csv_atomic(
                        pd.DataFrame(rows),
                        args.output_dir / "predictor_dataset.partial.csv",
                    )
            proposal_draws[bin_name] = draws

    dataframe = pd.DataFrame(rows)
    feature_columns = [column for column in dataframe.columns if column.startswith("feature_")]
    model, metrics = train_predictor(
        dataframe[feature_columns].to_numpy(dtype=np.float32),
        dataframe["accuracy"].to_numpy(dtype=np.float32),
        args, device, dataframe["sampling_bin"].to_numpy(),
    )
    dataset_path = args.output_dir / "predictor_dataset.csv"
    model_path = args.output_dir / "predictor_model.pt"
    metrics_path = args.output_dir / "predictor_metrics.json"
    subset_path = args.output_dir / "validation_subset_indices.json"
    provenance_path = args.output_dir / "predictor_provenance.json"
    write_csv_atomic(dataframe, dataset_path)
    torch.save(model.state_dict(), model_path)
    subset_path.write_text(json.dumps({
        "dataset": "CINIC-10",
        "split": "val",
        "directory": str((args.data_dir / "val").resolve()),
        "full_sample_count": len(val_dataset),
        "selected_sample_count": len(subset_indices),
        "subset_seed": args.subset_seed,
        "indices": subset_indices,
    }, indent=2) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    provenance = {
        "completed": True,
        "dataset": "CINIC-10",
        "split": "official validation directory",
        "test_data_loaded": False,
        "validation_directory": str((args.data_dir / "val").resolve()),
        "validation_full_sample_count": len(val_dataset),
        "validation_subset_sample_count": len(subset_indices),
        "validation_transform": "ToTensor(); Normalize(CINIC10_MEAN, CINIC10_STD); no augmentation",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture_count": len(dataframe),
        "initial_architecture_count": initial_architecture_count,
        "new_architecture_count": len(dataframe) - initial_architecture_count,
        "initial_dataset_path": str(args.initial_dataset.resolve()) if args.initial_dataset else None,
        "initial_dataset_sha256": sha256_file(args.initial_dataset) if args.initial_dataset else None,
        "continuation_sampling": (
            "RNG restarted from architecture_seed; existing architecture keys "
            "were rejected before evaluation. The combined dataset is "
            "deterministic but is not an uninterrupted one-pass sample."
            if args.initial_dataset else None
        ),
        "sampling_strategy": args.sampling_strategy,
        "sampling_bins": [
            {"name": name, "lower": lower, "upper": upper}
            for name, lower, upper in MAC_BINS
        ] if args.sampling_strategy == "restricted-bin-stratified" else None,
        "sampling_bin_counts": dataframe["sampling_bin"].value_counts().sort_index().to_dict(),
        "proposal_draws_by_bin": proposal_draws,
        "architecture_seed": args.architecture_seed,
        "subset_seed": args.subset_seed,
        "predictor_split_seed": args.predictor_split_seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "architecture_batch_size": args.batch_size,
        "predictor_batch_size": args.predictor_batch_size,
        "workers": args.workers,
        "device": torch.cuda.get_device_name(device),
        "python_executable": str(Path(__import__("sys").executable).resolve()),
        "torch_version": torch.__version__,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "dataset_sha256": sha256_file(dataset_path),
        "model_sha256": sha256_file(model_path),
        "subset_manifest_sha256": sha256_file(subset_path),
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "elapsed_scope": "new architecture evaluations, final predictor fitting, and artifact writing",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output_dir / "predictor_dataset.partial.csv").unlink(missing_ok=True)
    print(f"Validation predictor dataset: {dataset_path}")
    print(f"Validation predictor model: {model_path}")
    print(f"Held-out predictor metrics: {metrics}")


if __name__ == "__main__":
    main()
