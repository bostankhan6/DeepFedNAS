#!/usr/bin/env python3
"""Download CIFAR and materialize the fixed train/validation split."""
import argparse
import pickle
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("cifar10", "cifar100"))
    args = parser.parse_args()
    data_dir = ROOT / "data" / args.dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_class = CIFAR10 if args.dataset == "cifar10" else CIFAR100
    official_train = dataset_class(str(data_dir), train=True, download=True)
    dataset_class(str(data_dir), train=False, download=True)
    indices_dir = ROOT / "configs" / "splits" / args.dataset
    train_indices = np.load(indices_dir / "train_indices_seed0.npy")
    val_indices = np.load(indices_dir / "val_indices_seed0.npy")
    assert len(train_indices) == 45000 and len(val_indices) == 5000
    assert np.array_equal(np.sort(np.r_[train_indices, val_indices]), np.arange(50000))
    images = official_train.data
    labels = np.asarray(official_train.targets)
    for name, indices in (("train", train_indices), ("val", val_indices)):
        with (data_dir / f"{name}.pkl").open("wb") as output:
            pickle.dump((images[indices], labels[indices]), output)
    print(f"Prepared {args.dataset}: 45,000 training and 5,000 validation images")

if __name__ == "__main__":
    main()
