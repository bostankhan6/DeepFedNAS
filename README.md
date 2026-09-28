# DeepFedNAS

**Federated neural architecture search across compute budgets.** DeepFedNAS trains one elastic supernet using a 60-subnet curriculum, then selects architectures for a target multiply–accumulate operation (MAC) budget without training an accuracy predictor.

**Paper:** [DeepFedNAS on arXiv (v4)](https://arxiv.org/abs/2601.15127v4)

## Contributions

- **Fitness-guided architecture selection:** A structural fitness function combines network information measures with architectural constraints to rank subnetworks without evaluating them on validation images during search.
- **Pareto-guided federated training:** A redesigned elastic ResNet supernet is trained with architectures from a precomputed 60-subnet path across the operating MAC range.
- **Predictor-free deployment search:** A genetic search selects a subnet for a new MAC budget without retraining the supernet or constructing an accuracy predictor.

The repository includes a matched **SuperFedNAS** comparison. Both methods train the same supernet over the same operating MAC range; SuperFedNAS samples architectures within that range and uses a validation-trained accuracy predictor for search.

## Results

At the smallest shared budget (**0.458–0.95 billion MACs**), DeepFedNAS improves official-test accuracy on all three datasets:

| Dataset | SuperFedNAS | DeepFedNAS |
| --- | ---: | ---: |
| CIFAR-10 | 93.86 ± 0.21% | **94.43 ± 0.23%** |
| CIFAR-100 | 71.21 ± 0.41% | **73.09 ± 0.41%** |
| CINIC-10 | 78.74 ± 0.30% | **81.81 ± 0.38%** |

The gain in mean test accuracy holds across **all four shared MAC ranges** (percentage points):

| MAC range (billions) | CIFAR-10 | CIFAR-100 | CINIC-10 |
| --- | ---: | ---: | ---: |
| 0.458–0.95 | +0.57 | +1.88 | +3.07 |
| 0.95–1.45 | +0.52 | +1.63 | +3.59 |
| 1.45–2.45 | +0.72 | +2.36 | +4.22 |
| 2.45–3.403 | +1.04 | +2.60 | +4.80 |
| **Mean** | **+0.71** | **+2.12** | **+3.92** |

Accuracy values are mean ± standard deviation over five architecture-search seeds (42–46) with partition α=100 and participation rate 0.4. At α=0.1 on CIFAR-10, the gains span 3.40–4.55 points. At the smallest CIFAR-100 budget, DeepFedNAS reaches 73.09% with 18.88M parameters, exceeding SuperFedNAS's best mean accuracy across the four ranges (72.49% with 55.70M parameters).

For CIFAR-10, building the SuperFedNAS accuracy predictor requires 10,000 full sweeps of the 5,000-image validation split (50 million image-level forward evaluations) and took about 3.9 hours on an NVIDIA RTX A5000. DeepFedNAS needs no accuracy predictor and searches a target budget in about 20 seconds on a CPU. Its 60-subnet cache is prepared once before training, taking about 20 minutes on a CPU.

## Get started

The package requires Python 3.12 or newer. Clone the repository and activate a virtual environment:

```bash
git clone https://github.com/bostankhan6/DeepFedNAS.git
cd DeepFedNAS
python3.12 -m venv .venv
source .venv/bin/activate
```

Install PyTorch and torchvision with the command for your hardware from the [official selector](https://pytorch.org/get-started/locally/). Then install the project and prepare CIFAR-10:

```bash
pip install -r requirements.txt
bash scripts/data_setup/download_cifar10.sh
```

Train both CIFAR-10 supernets (these runs require substantial GPU time):

```bash
bash experiments/02_deepfednas/cifar10.sh
bash experiments/01_baseline/cifar10.sh
```

To preview either training command, prefix it with `DRY_RUN=1`. The supplied [`60-subnet cache`](subnet_caches/4_stage_cache_60_subnets.csv) and [CIFAR-10 client partitions](configs/partitions/cifar10/) are ready to use; dataset images and trained checkpoints are not included. Each CIFAR-10 launcher selects the matching partition automatically. A real training run stops if that file is missing; it does not regenerate it.

## Reproduce search and test results

After training, build the SuperFedNAS predictor from its checkpoint and validation data:

```bash
DATASET=cifar10 \
CHECKPOINT=checkpoints/cifar10_alpha100_c0.4/range_matched_random/seed0/best_checkpoint_supernet.pt \
OUTPUT_DIR=outputs/cifar10/predictor \
bash scripts/search/collect_predictor.sh
```

This creates `predictor_model.pt` (the trained accuracy predictor), `predictor_dataset.csv` (the validation measurements used to train it), and provenance files in `outputs/cifar10/predictor/`. DeepFedNAS does not need a predictor.

The search script reads **both** trained checkpoints and the SuperFedNAS predictor from their default locations. It searches four MAC ranges with seeds 42–46: DeepFedNAS ranks candidates by structural fitness, while SuperFedNAS ranks them by predicted validation accuracy. It saves the chosen architectures and input hashes in `outputs/cifar10/search/`. Run search first, then test:

```bash
python scripts/search/cifar10_locked_search.py
python scripts/evaluation/cifar10_locked_test.py
```

The test script loads the saved architectures and their matching checkpoints, checks the search manifest, and writes test results to the same search folder. Test images are used only at this final step. For a different run, give the search script `--baseline-checkpoint`, `--deepfednas-checkpoint`, `--predictor-dir`, and `--output-dir`; then pass that output folder to the test script with `--search-dir`.

For CIFAR-100 or CINIC-10, use the matching data script, training launchers, predictor settings, and `cifar100_locked_*` or `cinic10_locked_*` scripts. Both training folders also include CIFAR-10 launchers for other client data and participation settings. Run a script with `--help` for its path options.

## License

Apache-2.0. See [LICENSE](LICENSE).
