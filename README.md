# DeepFedNAS

**DeepFedNAS** is a federated neural architecture search framework for finding image classifiers at different compute budgets. It trains one elastic ResNet-style supernet across clients, then selects subnetworks that fit a target number of multiply–accumulate operations (MACs). This lets one trained model serve devices with different compute limits.

Training every candidate architecture separately would be expensive in a federated setting. DeepFedNAS instead prepares a path of 60 high-fitness subnetworks across the supported MAC range. During supernet training, it samples architectures from that path so the shared weights see subnetworks with different costs. After training, it searches for architectures using a structural fitness function rather than training an accuracy predictor for DeepFedNAS.

This repository also includes **SuperFedNAS** for comparison. In these experiments, SuperFedNAS samples subnetworks randomly within the same MAC range used by DeepFedNAS; the data, client schedule, augmentation, and supernet are otherwise the same. For post-training search, SuperFedNAS uses an accuracy predictor trained on validation data. Both methods are searched at the same compute budgets and evaluated on the test set only after their architectures are selected.

## What is included

| Location | Purpose |
| --- | --- |
| `src/deepfednas/` | Elastic supernet, federated training, subnet sampling, fitness, MAC calculation, and data loaders |
| `subnet_caches/4_stage_cache_60_subnets.csv` | Ready-to-use 60-subnet training path |
| `configs/` | Supernet configuration and fixed CIFAR split and partition files |
| `experiments/02_deepfednas/` | DeepFedNAS training launchers |
| `experiments/01_baseline/` | SuperFedNAS training launchers (range-matched random sampling) |
| `scripts/data_setup/` | Dataset download and preparation |
| `scripts/search/` | Validation-only predictor collection and architecture search |
| `scripts/evaluation/` | Final evaluation of locked architectures on official test data |

The repository contains code, scripts, the subnet cache, and split/partition files. It does not include dataset images or trained supernet checkpoints. Reproducing a full training run requires substantial GPU time.

## Results at a glance

At the smallest shared compute budget (0.458–0.95 billion MACs), DeepFedNAS achieves higher official-test accuracy on all three datasets:

| Dataset | SuperFedNAS accuracy (%) | DeepFedNAS accuracy (%) | Gain |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 93.86 ± 0.21 | **94.43 ± 0.23** | **+0.6 points** |
| CIFAR-100 | 71.21 ± 0.41 | **73.09 ± 0.41** | **+1.9 points** |
| CINIC-10 | 78.74 ± 0.30 | **81.81 ± 0.38** | **+3.1 points** |

Both methods use the same expanded supernet, training seed, data partitions, and compute intervals. The values are means and standard deviations across five architecture-search seeds (42–46), using one trained checkpoint per method. DeepFedNAS also leads SuperFedNAS in the other three tested MAC ranges on each dataset. With highly uneven CIFAR-10 client data (Dirichlet alpha `0.1`), it reaches 75.72% versus 71.17% for SuperFedNAS in the smallest range. On CIFAR-100, its smallest-range model reaches 73.09% with 18.88 million parameters; SuperFedNAS's highest mean accuracy across the four ranges is 72.49% with 55.70 million parameters.

DeepFedNAS also avoids the accuracy-predictor preparation needed by SuperFedNAS. In the CIFAR-10 benchmark, that preparation evaluates 10,000 subnetworks on all 5,000 validation images: **50 million image-level forward evaluations**, measured at about **3.9 hours on an NVIDIA RTX A5000**. A DeepFedNAS search for one compute budget took about **20 seconds on a CPU**. Its 60-subnet cache is prepared once before training; that separate CPU step took about **20 minutes**.

## Quick start: CIFAR-10

Use Python 3.10 or newer. Clone the repository and create an environment:

```bash
git clone https://github.com/bostankhan6/DeepFedNAS.git
cd DeepFedNAS
python3 -m venv .venv
source .venv/bin/activate
```

Install a PyTorch and torchvision build compatible with your hardware using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install the project and prepare CIFAR-10:

```bash
pip install -r requirements.txt
bash scripts/data_setup/download_cifar10.sh
```

You can inspect the training commands before starting a run:

```bash
DRY_RUN=1 bash experiments/02_deepfednas/cifar10.sh
DRY_RUN=1 bash experiments/01_baseline/cifar10.sh
```

Remove `DRY_RUN=1` to train the two methods:

```bash
bash experiments/02_deepfednas/cifar10.sh
bash experiments/01_baseline/cifar10.sh
```

These jobs use the same seed (`0`), fixed CIFAR-10 client partition, 20 total clients, 8 clients per round, MixAug augmentation, and 1,500 communication rounds. The difference is how they select subnetworks during training. Checkpoints are written under `checkpoints/cifar10_alpha100_c0.4/`: DeepFedNAS uses the `deepfednas/seed0/` folder, and SuperFedNAS uses `range_matched_random/seed0/`.

## Reproduce the experiment workflow

### 1. Prepare the data

Run the preparation script for each dataset you intend to use:

```bash
bash scripts/data_setup/download_cifar10.sh
bash scripts/data_setup/download_cifar100.sh
bash scripts/data_setup/download_cinic10.sh
```

The CIFAR scripts download the official data and apply the supplied seed-0 indices to create 45,000-image training and 5,000-image validation files. CIFAR-10 also uses the supplied client partition manifests. CINIC-10 uses its `train`, `val`, and `test` folders. Training and checkpoint selection use validation data; official test data is reserved for the final evaluation.

### 2. Train both supernets

Each method folder contains matching launchers for these conditions:

| Launcher | Condition |
| --- | --- |
| `cifar10.sh`, `cifar100.sh`, `cinic10.sh` | Main dataset runs, participation fraction `C=0.4` |
| `cifar10_alpha1.sh`, `cifar10_alpha0p1.sh` | CIFAR-10 client heterogeneity with partition alpha `1` or `0.1` |
| `cifar10_c01.sh`, `cifar10_c02.sh`, `cifar10_c06.sh` | CIFAR-10 participation fraction `C=0.1`, `0.2`, or `0.6` |

For any condition, run the same filename from `experiments/02_deepfednas/` and `experiments/01_baseline/`. CIFAR-10 and CINIC-10 use 1,500 rounds; CIFAR-100 uses 2,000 rounds. All supplied launchers use one training seed, `0`.

Set `GPU_ID` to choose a GPU, `PYTHON_BIN` to choose Python, and `WANDB_MODE` to change logging mode; W&B defaults to offline. Set `RESUME_TRAINING=1` when continuing from the run's latest checkpoint. The launcher checks that both the latest and best checkpoint files are present before resuming.

The included cache is the active training cache. SuperFedNAS reads it only to use the same MAC range and boundary subnetworks as DeepFedNAS; it samples the other subnetworks randomly within that range. You do not need to regenerate the cache to run the supplied experiments.

### 3. Build the SuperFedNAS validation predictor

After the SuperFedNAS CIFAR-10 checkpoint is ready, collect validation accuracy measurements and train its predictor:

```bash
DATASET=cifar10 \
CHECKPOINT=checkpoints/cifar10_alpha100_c0.4/range_matched_random/seed0/best_checkpoint_supernet.pt \
OUTPUT_DIR=outputs/cifar10/predictor \
bash scripts/search/collect_predictor.sh
```

The default collection measures 10,000 unique architectures, with 2,500 in each of four strict MAC intervals. It uses validation data, not the official test set. For CIFAR-100 or CINIC-10, change `DATASET`, `CHECKPOINT`, and `OUTPUT_DIR` accordingly.

### 4. Search, lock, and evaluate

Once both trained checkpoints and the SuperFedNAS predictor are ready, select subnetworks and then evaluate them:

```bash
python scripts/search/cifar10_locked_search.py
python scripts/evaluation/cifar10_locked_test.py
```

DeepFedNAS uses structural fitness during search; SuperFedNAS uses its validation-trained accuracy predictor. The search command selects architectures in four MAC ranges using seeds `42–46` and saves the selections. The evaluation command then measures those saved architectures on the official CIFAR-10 test set. Test images are never used to choose architectures. Results are saved under `outputs/cifar10/`.

For CIFAR-100 or CINIC-10, run the matching search and evaluation scripts (`cifar100_locked_search.py` and `cifar100_locked_test.py`, or `cinic10_locked_search.py` and `cinic10_locked_test.py`). For a different CIFAR-10 setting, point the search script to that setting's checkpoints and predictor, then give its search output folder to the evaluation script. Each script's `--help` shows how to change those paths.

## Optional: generate another subnet cache

The supplied 60-subnet cache is ready to use. If you want to generate a separate cache from the supernet configuration, run:

```bash
bash scripts/cache_generation/run_subnet_cache_generation.sh
```

This writes `subnet_caches/generated_60_subnets.csv` and leaves the supplied training cache unchanged. Cache generation is computationally expensive.

## License

This project is licensed under Apache-2.0. See [LICENSE](LICENSE).
