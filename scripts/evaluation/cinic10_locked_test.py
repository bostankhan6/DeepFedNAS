#!/usr/bin/env python3
"""Evaluate a completed, hash-locked CINIC-10 multi-seed search manifest."""

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
from torchvision import datasets, transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from deepfednas.Server.generic_server_model import GenericServerOFA  # noqa: E402


CINIC10_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC10_STD = [0.24205776, 0.23828046, 0.25874835]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=REPO_ROOT / "outputs/cinic10/search")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/cinic10")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Smoke-test only: evaluate the first N test images")
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


def load_locked_manifest(search_dir: Path) -> tuple[list[dict], dict, Path]:
    provenance_path = search_dir / "search_provenance.json"
    manifest_path = search_dir / "locked_architectures.csv"
    if not provenance_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Completed search provenance/manifest not found")
    provenance = json.loads(provenance_path.read_text())
    if not provenance.get("completed"):
        raise RuntimeError("Refusing test evaluation: search manifest is incomplete")
    if provenance.get("test_data_loaded") is not False:
        raise RuntimeError("Search provenance does not certify test-data isolation")
    actual_hash = sha256_file(manifest_path)
    if actual_hash != provenance.get("locked_architectures_sha256"):
        raise RuntimeError("Refusing test evaluation: locked architecture hash mismatch")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != provenance.get("expected_rows") or len(rows) != provenance.get("result_rows"):
        raise RuntimeError("Refusing test evaluation: locked manifest row count mismatch")
    for row in rows:
        macs = int(row["actual_macs"])
        if not int(row["macs_lower"]) <= macs <= int(row["macs_upper"]):
            raise RuntimeError(f"Architecture outside locked MAC bin: {row}")
        expected_checkpoint_hash = provenance["checkpoints"][row["method"]]["sha256"]
        if row["checkpoint_sha256"] != expected_checkpoint_hash:
            raise RuntimeError("Checkpoint hash mismatch inside locked manifest")
    return rows, provenance, manifest_path


def build_test_loader(args: argparse.Namespace):
    test_dir = args.data_dir / "test"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
    ])
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    full_count = len(dataset)
    if full_count != 90_000:
        raise RuntimeError(f"Expected 90,000 CINIC-10 test images, found {full_count}")
    if args.limit_samples is not None:
        if args.limit_samples <= 0:
            raise ValueError("--limit-samples must be positive")
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit_samples, full_count)))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return loader, full_count


def load_server(checkpoint_path: Path) -> GenericServerOFA:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, description: str):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    started = time.time()
    for images, labels in tqdm(
        loader, desc=description, unit="batch", dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return correct, total, time.time() - started


def summarize(rows: list[dict]) -> list[dict]:
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
            "min_actual_macs": min(macs),
            "max_actual_macs": max(macs),
        })
    return summary


def main() -> None:
    args = parse_args()
    locked_rows, search_provenance, manifest_path = load_locked_manifest(args.search_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    loader, full_count = build_test_loader(args)

    results = []
    accuracy_cache = {}
    rows_by_method = defaultdict(list)
    for row in locked_rows:
        rows_by_method[row["method"]].append(row)

    for method, method_rows in rows_by_method.items():
        checkpoint_path = Path(method_rows[0]["checkpoint_path"])
        if sha256_file(checkpoint_path) != method_rows[0]["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed since search: {checkpoint_path}")
        print(f"\nLoading {method}: {checkpoint_path.name}")
        server = load_server(checkpoint_path)
        for row in method_rows:
            key = (method, architecture_key(row))
            if key not in accuracy_cache:
                client_model = server.get_subnet(
                    d=json.loads(row["arch_d"]),
                    e=json.loads(row["arch_e"]),
                    w_indices=json.loads(row["arch_w_indices"]),
                    preserve_weight=True,
                )
                model = client_model.model
                correct, total, elapsed = evaluate(
                    model, loader, device,
                    f"{method} {row['macs_bin']} seed {row['search_seed']}",
                )
                accuracy_cache[key] = (correct, total, elapsed)
                model.cpu()
                del model, client_model
                torch.cuda.empty_cache()
                gc.collect()
            correct, total, elapsed = accuracy_cache[key]
            result = dict(row)
            result.update({
                "architecture_sha256": key[1],
                "test_correct": correct,
                "test_total": total,
                "test_accuracy_percent": 100.0 * correct / total,
                "evaluation_seconds": elapsed,
            })
            results.append(result)
            print(
                f"{method} {row['macs_bin']} seed {row['search_seed']}: "
                f"{100.0 * correct / total:.4f}% ({correct:,}/{total:,})"
            )
        del server
        gc.collect()

    results_path = args.search_dir / "test_results_by_search_seed.csv"
    summary_path = args.search_dir / "test_summary_by_search_seed.csv"
    provenance_path = args.search_dir / "test_provenance.json"
    summary_rows = summarize(results)
    write_csv(results_path, results)
    write_csv(summary_path, summary_rows)
    provenance = {
        "completed": True,
        "dataset": "CINIC-10",
        "split": "test",
        "full_test_sample_count": full_count,
        "evaluated_sample_count": len(loader.dataset),
        "transform": "ToTensor(); Normalize(CINIC10_MEAN, CINIC10_STD); no augmentation",
        "test_used_for_selection": False,
        "uncertainty_definition": "Sample standard deviation across search seeds conditional on one fixed trained checkpoint per method",
        "search_provenance_sha256": sha256_file(args.search_dir / "search_provenance.json"),
        "locked_architectures_sha256": sha256_file(manifest_path),
        "evaluation_script": str(Path(__file__).resolve()),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "device": torch.cuda.get_device_name(device),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_module_path": torch.__file__,
        "torchvision_version": __import__("torchvision").__version__,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "result_rows": len(results),
        "unique_architectures_evaluated": len(accuracy_cache),
        "test_results_sha256": sha256_file(results_path),
        "test_summary_sha256": sha256_file(summary_path),
        "search_methods": search_provenance["methods"],
        "search_seeds": search_provenance["seeds"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nDetailed results: {results_path}")
    print(f"Search-seed summary: {summary_path}")
    print(f"Test provenance: {provenance_path}")


if __name__ == "__main__":
    main()
