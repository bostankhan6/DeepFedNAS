# DeepFedNAS

**Federated neural architecture search for image classifiers at different compute budgets.** DeepFedNAS trains one elastic supernet and uses a 60-subnet cache to guide training across a range of model sizes. After training, structural fitness selects subnetworks for a target multiply–accumulate operation (MAC) budget.

This repository includes a matched **SuperFedNAS** comparison. Both methods use the same supernet and training setup; SuperFedNAS samples within the same MAC range and trains a validation-based accuracy predictor for architecture search. DeepFedNAS uses the supplied subnet cache and searches without that predictor.

## Results

Official-test accuracy at the smallest shared budget (**0.458–0.95 billion MACs**):

| Dataset | SuperFedNAS | DeepFedNAS |
| --- | ---: | ---: |
| CIFAR-10 | 93.86 ± 0.21% | **94.43 ± 0.23%** |
| CIFAR-100 | 71.21 ± 0.41% | **73.09 ± 0.41%** |
| CINIC-10 | 78.74 ± 0.30% | **81.81 ± 0.38%** |

Values are mean ± standard deviation over five architecture-search seeds (42–46), using one trained checkpoint per method. DeepFedNAS also leads in the other three tested MAC ranges on each dataset. For CIFAR-10, preparing the SuperFedNAS predictor took about 3.9 hours on an NVIDIA RTX A5000; one DeepFedNAS budget search took about 20 seconds on a CPU. The DeepFedNAS cache is built once before training (about 20 minutes on a CPU).

## Get started

Python 3.12.12 is recommended; it matches the environment used for these experiments. The package requires Python 3.10 or newer. Clone the repository and activate a virtual environment:

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
