#!/usr/bin/env python3
"""Select architectures in strict MAC bins without loading test data.

Five search seeds (42--46) produce a locked manifest before final test
inference. The optional --upper-bound-only mode is exploratory and does not
implement the strict-bin evaluation workflow.
"""

from __future__ import annotations

import argparse
import copy
import csv
import functools
import hashlib
import json
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from deepfednas.nas.parallel_fitness_search import (  # noqa: E402
    calculate_fitness as deepfednas_fitness,
    decode_chromosome,
)
from deepfednas.utils.subnet_cost import subnet_macs  # noqa: E402


DEEP_CHECKPOINT = REPO_ROOT / "checkpoints/cinic10_alpha100_c0.4/deepfednas/seed0/best_checkpoint_supernet.pt"
BASELINE_CHECKPOINT = REPO_ROOT / "checkpoints/cinic10_alpha100_c0.4/range_matched_random/seed0/best_checkpoint_supernet.pt"
BASELINE_PREDICTOR = REPO_ROOT / "outputs/cinic10/predictor/predictor_model.pt"
BASELINE_PREDICTOR_DATASET = REPO_ROOT / "outputs/cinic10/predictor/predictor_dataset.csv"

MAC_BINS = (
    ("0.458-0.95B", 458_237_952.0, 0.95e9),
    ("0.95-1.45B", 0.95e9, 1.45e9),
    ("1.45-2.45B", 1.45e9, 2.45e9),
    ("2.45-3.403B", 2.45e9, 3_403_370_496.0),
)

# The legacy binned scripts displayed these ranges but enforced only the
# upper edge.  The final upper target was 3.75B even though the current
# checkpoints' largest supported architecture is below that value.
UPPER_BOUND_BINS = (
    ("<=0.95B", 0.0, 0.95e9),
    ("<=1.45B", 0.0, 1.45e9),
    ("<=2.45B", 0.0, 2.45e9),
    ("<=3.75B", 0.0, 3.75e9),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/cinic10/search")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "DeepFedNAS", "Range-matched Random",
        ),
        default=["DeepFedNAS", "Range-matched Random"],
    )
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--generations", type=int, default=512)
    parser.add_argument("--mutation-probability", type=float, default=0.3)
    parser.add_argument("--parent-ratio", type=float, default=0.25)
    parser.add_argument("--search-workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument(
        "--deep-stem-index",
        type=int,
        default=None,
        help=(
            "Fix the first DeepFedNAS width gene to this width-choice index. "
            "Use this when the trained supernet only supports a restricted stem "
            "width; the default preserves the unrestricted search."
        ),
    )
    parser.add_argument(
        "--upper-bound-only",
        action="store_true",
        help=(
            "Use the legacy protocol: constrain only macs <= each upper "
            "target, without enforcing a lower MAC-bin bound."
        ),
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


def load_arch_params(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arch_params = copy.deepcopy(checkpoint["arch_params"])
    del checkpoint
    arch_params["original_stage_base_channels"] = np.asarray(arch_params["original_stage_base_channels"])
    arch_params["expansion_ratio_choices"] = list(arch_params["expansion_ratio_choices"])
    arch_params["width_multiplier_choices"] = list(arch_params["width_multiplier_choices"])
    return arch_params


def macs_and_params(arch: dict, arch_params: dict) -> tuple[int, int]:
    macs, params = subnet_macs(
        arch["d"], arch["e"], arch["w_indices"],
        arch_params["width_multiplier_choices"], arch_params,
    )
    return int(macs), int(params)


def _deep_worker(fixed_args, chromosome):
    lower, upper, rho_limit, width_choices, exp_choices, n_depth, n_exp, n_width, arch_params, effectiveness_weight = fixed_args
    arch = decode_chromosome(chromosome, n_depth, n_exp, n_width, exp_choices)
    macs, _ = macs_and_params(arch, arch_params)
    if macs < lower or macs > upper:
        return -1e9
    return deepfednas_fitness(
        chromosome, upper, rho_limit, width_choices, exp_choices,
        n_depth, n_exp, n_width, arch_params, effectiveness_weight,
    )


def search_deepfednas(
    arch_params: dict,
    lower: float,
    upper: float,
    seed: int,
    population_size: int,
    generations: int,
    mutation_probability: float,
    workers: int,
    stem_index: int | None = None,
    upper_bound_only: bool = False,
) -> tuple[dict, float]:
    n_stages = arch_params["num_stages"]
    blocks_per_stage = arch_params["max_extra_blocks_per_stage"] + 1
    n_depth = n_stages
    n_exp = n_stages * blocks_per_stage
    n_width = n_stages + 1
    chromosome_length = n_depth + n_exp + n_width
    stem_gene = n_depth + n_exp
    depth_choices = np.arange(blocks_per_stage)
    exp_choices = np.asarray(arch_params["expansion_ratio_choices"])
    width_choices = arch_params["width_multiplier_choices"]
    if stem_index is not None and not 0 <= stem_index < len(width_choices):
        raise ValueError(
            f"DeepFedNAS stem index {stem_index} is outside the width-choice range "
            f"[0, {len(width_choices) - 1}]"
        )
    rho_limit = arch_params.get("supernet_rho0_constraint", 2.0)
    effectiveness_weight = arch_params.get("supernet_effectiveness_fitness_weight", 100)
    rng = np.random.default_rng(seed)
    # The legacy upper-only search used unconstrained random initialization;
    # the strict-bin protocol uses a lower-bound-aware width floor to make
    # feasible initialization practical for the highest strict bin.
    if upper_bound_only:
        initial_width_floor = 0
    elif lower >= 2.45e9:
        initial_width_floor = len(width_choices) - 1
    elif lower >= 1.45e9:
        initial_width_floor = max(0, len(width_choices) - 2)
    elif lower >= 0.95e9:
        initial_width_floor = len(width_choices) // 2
    else:
        initial_width_floor = 0
    fixed_args = (
        lower, upper, rho_limit, width_choices, exp_choices,
        n_depth, n_exp, n_width, arch_params, effectiveness_weight,
    )
    worker_fn = functools.partial(_deep_worker, fixed_args)

    with multiprocessing.Pool(processes=workers) as pool:
        population = []
        attempts = 0
        max_attempts = population_size * 2000
        while len(population) < population_size and attempts < max_attempts:
            batch_size = max(64, (population_size - len(population)) * 5)
            candidates = []
            for _ in range(batch_size):
                attempts += 1
                candidate = np.concatenate([
                    rng.choice(depth_choices, n_depth),
                    rng.integers(0, len(exp_choices), n_exp),
                    rng.integers(initial_width_floor, len(width_choices), n_width),
                ])
                if stem_index is not None:
                    candidate[stem_gene] = stem_index
                candidates.append(candidate)
            scores = pool.map(worker_fn, candidates)
            population.extend(c for c, score in zip(candidates, scores) if score > -1e7)
            population = population[:population_size]
        if len(population) != population_size:
            raise RuntimeError(
                f"DeepFedNAS seed {seed}: only {len(population)} valid initial candidates "
                f"for [{lower}, {upper}] after {attempts} attempts"
            )

        population = np.stack(population)
        best_chromosome = None
        best_score = -float("inf")
        for _ in tqdm(
            range(generations), desc=f"DeepFedNAS {seed} {upper/1e9:.3f}B",
            leave=False, disable=not sys.stderr.isatty(),
        ):
            scores = np.asarray(pool.map(worker_fn, population))
            best_index = int(np.argmax(scores))
            if scores[best_index] > best_score:
                best_score = float(scores[best_index])
                best_chromosome = population[best_index].copy()

            parents = []
            for _ in range(len(population)):
                competitors = rng.choice(len(population), 3, replace=False)
                parents.append(population[competitors[np.argmax(scores[competitors])]])
            children = []
            for index in range(0, len(parents), 2):
                if index + 1 == len(parents):
                    children.append(parents[index].copy())
                    continue
                crossover = int(rng.integers(1, chromosome_length))
                first, second = parents[index], parents[index + 1]
                children.extend([
                    np.concatenate((first[:crossover], second[crossover:])),
                    np.concatenate((second[:crossover], first[crossover:])),
                ])
            population = np.asarray(children[:population_size])
            for chromosome in population:
                if rng.random() < mutation_probability:
                    gene = int(rng.integers(chromosome_length))
                    if gene < n_depth:
                        chromosome[gene] = rng.choice(depth_choices)
                    elif gene < n_depth + n_exp:
                        chromosome[gene] = rng.integers(len(exp_choices))
                    elif stem_index is None or gene != stem_gene:
                        # An unrestricted search must mutate the stem like every
                        # other width gene.  Skip it only when a fixed stem was
                        # explicitly requested for a controlled search.
                        chromosome[gene] = rng.integers(len(width_choices))
                if stem_index is not None:
                    chromosome[stem_gene] = stem_index

            child_scores = np.asarray(pool.map(worker_fn, population))
            worst_index = int(np.argmin(child_scores))
            if best_chromosome is not None and best_score > child_scores[worst_index]:
                population[worst_index] = best_chromosome.copy()

    if best_chromosome is None:
        raise RuntimeError(f"DeepFedNAS seed {seed} failed to select an architecture")
    arch = decode_chromosome(best_chromosome, n_depth, n_exp, n_width, exp_choices)
    return arch, best_score


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


def arch_features(arch: dict, arch_params: dict) -> np.ndarray:
    exp_choices = arch_params["expansion_ratio_choices"]
    exp_one_hot = np.zeros(len(arch["e"]) * len(exp_choices))
    for index, value in enumerate(arch["e"]):
        exp_one_hot[index * len(exp_choices) + exp_choices.index(value)] = 1
    return np.concatenate([
        np.asarray(arch["d"]), exp_one_hot, np.asarray(arch["w_indices"]),
    ])


def random_architecture(
    arch_params: dict,
    rng: random.Random,
    width_floor_index: int = 0,
) -> dict:
    n_stages = arch_params["num_stages"]
    max_extra = arch_params["max_extra_blocks_per_stage"]
    return {
        "d": [rng.randint(0, max_extra) for _ in range(n_stages)],
        "e": [rng.choice(arch_params["expansion_ratio_choices"]) for _ in range(n_stages * (max_extra + 1))],
        "w_indices": [
            rng.randrange(width_floor_index, len(arch_params["width_multiplier_choices"]))
            for _ in range(n_stages + 1)
        ],
    }


def mutate_architecture(arch: dict, arch_params: dict, rng: random.Random) -> dict:
    child = copy.deepcopy(arch)
    gene_type = rng.choice(("d", "e", "w"))
    if gene_type == "d":
        index = rng.randrange(len(child["d"]))
        child["d"][index] = rng.randint(0, arch_params["max_extra_blocks_per_stage"])
    elif gene_type == "e":
        index = rng.randrange(len(child["e"]))
        child["e"][index] = rng.choice(arch_params["expansion_ratio_choices"])
    else:
        index = rng.randrange(len(child["w_indices"]))
        child["w_indices"][index] = rng.randrange(len(arch_params["width_multiplier_choices"]))
    return child


def crossover_architectures(first: dict, second: dict, rng: random.Random) -> dict:
    child = copy.deepcopy(first)
    for key in child:
        child[key] = [rng.choice((a, b)) for a, b in zip(first[key], second[key])]
    return child


def search_baseline(
    arch_params: dict,
    predictor: AccuracyPredictor,
    lower: float,
    upper: float,
    seed: int,
    population_size: int,
    generations: int,
    mutation_ratio: float,
    parent_ratio: float,
) -> tuple[dict, float]:
    rng = random.Random(seed)
    if lower >= 2.45e9:
        initial_width_floor = len(arch_params["width_multiplier_choices"]) - 1
    elif lower >= 1.45e9:
        initial_width_floor = max(0, len(arch_params["width_multiplier_choices"]) - 2)
    elif lower >= 0.95e9:
        initial_width_floor = len(arch_params["width_multiplier_choices"]) // 2
    else:
        initial_width_floor = 0

    def valid(arch):
        macs, _ = macs_and_params(arch, arch_params)
        return lower <= macs <= upper

    def score_many(architectures):
        features = torch.tensor(
            np.stack([arch_features(arch, arch_params) for arch in architectures]),
            dtype=torch.float32,
        )
        with torch.inference_mode():
            return [float(value) for value in predictor(features).squeeze(1).tolist()]

    initial_architectures = []
    attempts = 0
    while len(initial_architectures) < population_size and attempts < population_size * 4000:
        candidate = random_architecture(arch_params, rng, initial_width_floor)
        attempts += 1
        if valid(candidate):
            initial_architectures.append(candidate)
    if len(initial_architectures) != population_size:
        raise RuntimeError(
            f"Baseline seed {seed}: only {len(initial_architectures)} valid initial candidates "
            f"for [{lower}, {upper}] after {attempts} attempts"
        )
    population = list(zip(score_many(initial_architectures), initial_architectures))

    parent_size = max(2, int(population_size * parent_ratio))
    for _ in tqdm(
        range(generations), desc=f"Baseline {seed} {upper/1e9:.3f}B",
        leave=False, disable=not sys.stderr.isatty(),
    ):
        parents = sorted(population, key=lambda item: item[0], reverse=True)[:parent_size]
        children = []
        attempts = 0
        while len(parents) + len(children) < population_size and attempts < population_size * 20:
            first = rng.choice(parents)[1]
            if rng.random() < mutation_ratio:
                child = mutate_architecture(first, arch_params, rng)
            else:
                child = crossover_architectures(first, rng.choice(parents)[1], rng)
            if valid(child):
                children.append(child)
            attempts += 1
        if not children:
            raise RuntimeError(f"Baseline seed {seed}: population collapsed")
        population = list(parents) + list(zip(score_many(children), children))
    return max(population, key=lambda item: item[0])[1], max(population, key=lambda item: item[0])[0]


def load_predictor(arch_params: dict) -> AccuracyPredictor:
    sample = random_architecture(arch_params, random.Random(0))
    predictor = AccuracyPredictor(len(arch_features(sample, arch_params)))
    state = torch.load(BASELINE_PREDICTOR, map_location="cpu", weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    return predictor


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Search seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_csv = args.output_dir / "locked_architectures.partial.csv"
    final_csv = args.output_dir / "locked_architectures.csv"
    provenance_path = args.output_dir / "search_provenance.json"
    rows = []
    if partial_csv.is_file():
        with partial_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(f"Resuming from {len(rows)} completed selections in {partial_csv}")

    baseline_spec = (BASELINE_CHECKPOINT, load_arch_params(BASELINE_CHECKPOINT))
    specs = {
        "DeepFedNAS": (DEEP_CHECKPOINT, load_arch_params(DEEP_CHECKPOINT)),
        "Range-matched Random": baseline_spec,
    }
    baseline_methods = {
        "Range-matched Random",
    }
    active_baseline_method = next(
        (method for method in args.methods if method in baseline_methods), None
    )
    predictor = (
        load_predictor(specs[active_baseline_method][1])
        if active_baseline_method is not None
        else None
    )
    active_bins = UPPER_BOUND_BINS if args.upper_bound_only else MAC_BINS
    expected_rows = len(args.methods) * len(active_bins) * len(args.seeds)
    provenance = {
        "completed": False,
        "test_data_loaded": False,
        "selection_split": "No dataset loaded by this script; DeepFedNAS uses structural fitness and baseline uses a pre-existing accuracy predictor",
        "methods": args.methods,
        "seeds": args.seeds,
        "bin_protocol": "upper-bound-only" if args.upper_bound_only else "restricted-inclusive",
        "mac_bins": [{"name": name, "lower": lower, "upper": upper} for name, lower, upper in active_bins],
        "population_size": args.population_size,
        "generations": args.generations,
        "mutation_probability": args.mutation_probability,
        "baseline_parent_ratio": args.parent_ratio,
        "baseline_initialization": (
            "Strict-bin searches use the same lower-bound-aware width-floor "
            "proposal as DeepFedNAS to obtain feasible initial populations; "
            "fitness ranking, crossover, mutation, and bin constraints are unchanged"
        ),
        "search_workers": args.search_workers,
        "deepfednas_initialization": (
            "Unconstrained random initialization for legacy upper-bound-only mode; "
            "strict mode uses bin-feasible width-biased initialization. Fitness, "
            "tournament selection, crossover, mutation, and elitism follow the paper search"
        ),
        "deepfednas_fixed_stem_width_index": args.deep_stem_index,
        "expected_rows": expected_rows,
        "resumed_rows": len(rows),
        "search_script": str(Path(__file__).resolve()),
        "search_script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoints": {
            method: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for method, (path, _) in specs.items() if method in args.methods
        },
        "baseline_predictor": {
            "path": str(BASELINE_PREDICTOR.resolve()),
            "sha256": sha256_file(BASELINE_PREDICTOR),
            "source_dataset_path": str(BASELINE_PREDICTOR_DATASET.resolve()),
            "source_dataset_sha256": sha256_file(BASELINE_PREDICTOR_DATASET),
            "split_audit": "The artifact has no embedded split metadata. Its matching repository generation path random-splits train_loader_global.dataset rather than loading data/cinic10/val; rebuild on official validation data before final resubmission reporting.",
        } if active_baseline_method is not None else None,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    completed_keys = {
        (row["method"], row["macs_bin"], int(row["search_seed"])) for row in rows
    }
    if len(completed_keys) != len(rows):
        raise RuntimeError("Partial search manifest contains duplicate method/bin/seed rows")

    for method in args.methods:
        checkpoint, arch_params = specs[method]
        for bin_name, lower, upper in active_bins:
            for seed in args.seeds:
                if (method, bin_name, seed) in completed_keys:
                    continue
                print(f"\nSelecting {method}, bin={bin_name}, seed={seed}")
                started = time.time()
                if method == "DeepFedNAS":
                    arch, selection_score = search_deepfednas(
                        arch_params, lower, upper, seed, args.population_size,
                        args.generations, args.mutation_probability, args.search_workers,
                        args.deep_stem_index,
                        args.upper_bound_only,
                    )
                    score_name = "structural_fitness"
                else:
                    arch, selection_score = search_baseline(
                        arch_params, predictor, lower, upper, seed,
                        args.population_size, args.generations,
                        args.mutation_probability, args.parent_ratio,
                    )
                    score_name = "predicted_validation_accuracy"
                macs, params = macs_and_params(arch, arch_params)
                if not lower <= macs <= upper:
                    raise AssertionError(f"Selected architecture outside {bin_name}: {macs}")
                rows.append({
                    "method": method,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": provenance["checkpoints"][method]["sha256"],
                    "macs_bin": bin_name,
                    "macs_lower": int(lower),
                    "macs_upper": int(upper),
                    "search_seed": seed,
                    "selection_score_name": score_name,
                    "selection_score": selection_score,
                    "actual_macs": macs,
                    "num_parameters": params,
                    "search_seconds": time.time() - started,
                    "arch_d": json.dumps(arch["d"]),
                    "arch_e": json.dumps([float(value) for value in arch["e"]]),
                    "arch_w_indices": json.dumps(arch["w_indices"]),
                })
                write_csv(partial_csv, rows)
                print(f"Locked {method} {bin_name} seed {seed}: {macs/1e9:.4f}B MACs")

    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(rows)}")
    write_csv(final_csv, rows)
    partial_csv.unlink(missing_ok=True)
    provenance["completed"] = True
    provenance["result_rows"] = len(rows)
    provenance["locked_architectures_csv"] = str(final_csv.resolve())
    provenance["locked_architectures_sha256"] = sha256_file(final_csv)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nCompleted locked search manifest: {final_csv}")
    print(f"Search provenance: {provenance_path}")


if __name__ == "__main__":
    main()
