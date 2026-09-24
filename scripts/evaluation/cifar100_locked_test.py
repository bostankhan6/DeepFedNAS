#!/usr/bin/env python3
"""Evaluate a completed CIFAR-100 architecture-search manifest on test data.

This is a separate final phase. It refuses incomplete or hash-mismatched
manifests and performs no ranking, tuning, or architecture replacement.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from deepfednas.data.cifar100.datasets import CIFAR100_truncated
from deepfednas.Server.generic_server_model import GenericServerOFA


CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=REPO_ROOT / "outputs/cifar100/search")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/cifar100")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing test-result artifacts (disabled by default).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def architecture_key(row: dict) -> str:
    canonical = json.dumps({
        "d": json.loads(row["arch_d"]),
        "e": json.loads(row["arch_e"]),
        "w_indices": json.loads(row["arch_w_indices"]),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_manifest(search_dir: Path):
    provenance_path = search_dir / "search_provenance.json"
    manifest_path = search_dir / "locked_architectures.csv"
    provenance = json.loads(provenance_path.read_text())
    if not provenance.get("completed") or provenance.get("test_data_loaded") is not False:
        raise RuntimeError("Refusing test evaluation: search is incomplete or not test-isolated")
    if sha256_file(manifest_path) != provenance.get("locked_architectures_sha256"):
        raise RuntimeError("Refusing test evaluation: locked architecture hash mismatch")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != provenance.get("expected_rows"):
        raise RuntimeError("Locked manifest row count mismatch")
    for row in rows:
        if not int(row["macs_lower"]) <= int(row["actual_macs"]) <= int(row["macs_upper"]):
            raise RuntimeError(f"Architecture outside locked MAC bin: {row}")
        if row["checkpoint_sha256"] != provenance["checkpoints"][row["method"]]["sha256"]:
            raise RuntimeError("Checkpoint hash mismatch inside locked manifest")
    return rows, provenance, manifest_path


def load_server(path: Path) -> GenericServerOFA:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arch_params = copy.deepcopy(checkpoint["arch_params"])
    arch_params["original_stage_base_channels"] = np.asarray(arch_params["original_stage_base_channels"])
    arch_params["expansion_ratio_choices"] = list(arch_params["expansion_ratio_choices"])
    arch_params["width_multiplier_choices"] = list(arch_params["width_multiplier_choices"])
    server = GenericServerOFA(arch_params=arch_params, sampling_method="all_random", num_cli_total=1)
    server.set_model_params(checkpoint["params"])
    del checkpoint
    gc.collect()
    return server


@torch.inference_mode()
def evaluate(model, loader, device, description):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    started = time.time()
    for images, labels in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += int((model(images).argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    torch.cuda.synchronize(device)
    return correct, total, time.time() - started


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["macs_bin"])].append(row)
    summary = []
    for (method, macs_bin), values in grouped.items():
        values.sort(key=lambda row: int(row["search_seed"]))
        accuracies = [float(row["test_accuracy_percent"]) for row in values]
        macs = [int(row["actual_macs"]) for row in values]
        summary.append({
            "method": method,
            "macs_bin": macs_bin,
            "search_seeds": json.dumps([int(row["search_seed"]) for row in values]),
            "num_search_seeds": len(values),
            "num_unique_architectures": len({row["architecture_sha256"] for row in values}),
            "mean_test_accuracy_percent": statistics.mean(accuracies),
            "sample_std_test_accuracy_percent": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "min_test_accuracy_percent": min(accuracies),
            "max_test_accuracy_percent": max(accuracies),
            "mean_actual_macs": statistics.mean(macs),
        })
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    result_paths = (
        args.search_dir / "test_results_by_search_seed.csv",
        args.search_dir / "test_summary_by_search_seed.csv",
        args.search_dir / "test_provenance.json",
    )
    if not args.overwrite and any(path.exists() for path in result_paths):
        raise RuntimeError(
            "Refusing to replace an existing test evaluation. Use --overwrite "
            "only for a documented correction."
        )
    rows, search_provenance, manifest_path = load_manifest(args.search_dir)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_dataset = CIFAR100_truncated(str(args.data_dir), train=False, transform=test_transform, download=False)
    if len(test_dataset) != 10000:
        raise RuntimeError(f"Expected 10,000 CIFAR-100 test images, found {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )

    result_rows = []
    cache = {}
    rows_by_method = defaultdict(list)
    for row in rows:
        rows_by_method[row["method"]].append(row)
    for method, method_rows in rows_by_method.items():
        checkpoint_path = Path(method_rows[0]["checkpoint_path"])
        if sha256_file(checkpoint_path) != method_rows[0]["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed since search: {checkpoint_path}")
        server = load_server(checkpoint_path)
        for row in method_rows:
            key = (method, architecture_key(row))
            if key not in cache:
                client = server.get_subnet(
                    d=json.loads(row["arch_d"]), e=json.loads(row["arch_e"]),
                    w_indices=json.loads(row["arch_w_indices"]), preserve_weight=True,
                )
                correct, total, elapsed = evaluate(
                    client.model, test_loader, device,
                    f"{method} {row['macs_bin']} seed {row['search_seed']}",
                )
                cache[key] = (correct, total, elapsed)
                client.model.cpu()
                del client
                torch.cuda.empty_cache()
                gc.collect()
            correct, total, elapsed = cache[key]
            result = dict(row)
            result.update({
                "architecture_sha256": key[1],
                "test_correct": correct,
                "test_total": total,
                "test_accuracy_percent": 100.0 * correct / total,
                "evaluation_seconds": elapsed,
            })
            result_rows.append(result)
        del server
        gc.collect()

    results_path = args.search_dir / "test_results_by_search_seed.csv"
    summary_path = args.search_dir / "test_summary_by_search_seed.csv"
    provenance_path = args.search_dir / "test_provenance.json"
    write_csv(results_path, result_rows)
    write_csv(summary_path, summarize(result_rows))
    provenance = {
        "completed": True,
        "dataset": "CIFAR-100",
        "split": "official test set",
        "full_test_sample_count": len(test_dataset),
        "transform": "ToTensor(); Normalize(CIFAR100_MEAN, CIFAR100_STD); no augmentation",
        "test_used_for_selection": False,
        "uncertainty_definition": "Sample standard deviation across search seeds conditional on one fixed trained checkpoint per method",
        "search_provenance_sha256": sha256_file(args.search_dir / "search_provenance.json"),
        "locked_architectures_sha256": sha256_file(manifest_path),
        "evaluation_script": str(Path(__file__).resolve()),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "device": torch.cuda.get_device_name(device),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "result_rows": len(result_rows),
        "unique_architectures_evaluated": len(cache),
        "test_results_sha256": sha256_file(results_path),
        "test_summary_sha256": sha256_file(summary_path),
        "search_methods": search_provenance["methods"],
        "search_seeds": search_provenance["seeds"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Detailed results: {results_path}")
    print(f"Search-seed summary: {summary_path}")
    print(f"Test provenance: {provenance_path}")


if __name__ == "__main__":
    main()
