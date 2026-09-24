# DeepFedNAS

DeepFedNAS trains one elastic federated supernet and selects subnetworks for different compute budgets. A precomputed path of 60 structurally strong subnetworks guides training. Architecture search uses structural fitness; the matched random control uses a validation-trained accuracy predictor for its post-training search.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/deepfednas/` | Supernet, federated trainers, sampling, fitness and cost functions, data loaders |
| `configs/` | Supernet configuration and fixed split/partition manifests |
| `subnet_caches/` | The 60-subnet path used by training |
| `experiments/01_baseline/` | Range-matched random training entry points |
| `experiments/02_deepfednas/` | DeepFedNAS training entry points |
| `scripts/data_setup/` | Dataset preparation |
| `scripts/cache_generation/` | Optional cache generation |
| `scripts/search/` | Validation predictor collection and locked architecture search |
| `scripts/evaluation/` | Final evaluation of locked architectures |

## Install and prepare data

Use Python 3.10 or newer and install a PyTorch/torchvision build suitable for your CPU or CUDA setup. Then, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/data_setup/download_cifar10.sh
bash scripts/data_setup/download_cifar100.sh
bash scripts/data_setup/download_cinic10.sh
```

The CIFAR scripts download the official datasets and create fixed 45,000-image training and 5,000-image validation files from `configs/splits/`. CINIC-10 uses its official `train`, `val`, and `test` folders. Training and checkpoint selection use validation data; the official test sets are for final evaluation only. Dataset images and trained supernet weights are not included in this code repository.

## Train supernets

Run each method from the root. The two commands below use the same expanded supernet, fixed client partition, seed 0, MixAug pipeline, optimizer and schedule. The training sampler is the controlled difference.

```bash
bash experiments/02_deepfednas/cifar10.sh
bash experiments/01_baseline/cifar10.sh
```

`01_baseline` is range-matched random sampling. It uses the cache only to define the operational MAC interval and forced endpoints; interior architectures are sampled randomly inside that interval. `02_deepfednas` samples its interior architectures from the 60-subnet path. The active cache is `subnet_caches/4_stage_cache_60_subnets.csv`. Its last row's stored cost/fitness values are historical metadata; the architecture genes are the all-maximum subnet and runtime bounds are recomputed from those genes.

The same folders contain `cifar100.sh`, `cinic10.sh`, `cifar10_alpha1.sh`, `cifar10_alpha0p1.sh`, and `cifar10_c01.sh`, `cifar10_c02.sh`, `cifar10_c06.sh`. These cover the three main datasets, two non-IID conditions, and three additional client-participation conditions. Every script uses one training seed, 0. Set `GPU_ID`, `PYTHON_BIN`, or `WANDB_MODE` in your environment as needed; logging defaults to offline mode. Set `DRY_RUN=1` to print a training command without running it. Set `RESUME_TRAINING=1` to continue a run with its best/latest checkpoint pair.

For training, MixAug means random crop, horizontal flip, RandAugment (2 operations, magnitude 6), and alternating Mixup (alpha 0.4) or CutMix (alpha 1.0). CIFAR-10 and CINIC-10 train for 1,500 rounds; CIFAR-100 trains for 2,000 rounds. The default `C=0.4` uses 8 of 20 CIFAR clients or 40 of 100 CINIC clients per round.

## Search and evaluate

After both CIFAR-10 training jobs complete, collect the random control's validation accuracy predictor. This evaluates 10,000 unique architectures, 2,500 in each strict MAC interval, using validation data only:

```bash
DATASET=cifar10 \
CHECKPOINT=checkpoints/cifar10_alpha100_c0.4/range_matched_random/seed0/best_checkpoint_supernet.pt \
OUTPUT_DIR=outputs/cifar10/predictor \
bash scripts/search/collect_predictor.sh
```

Then search with seeds 42–46 and freeze the architecture manifest before final test inference:

```bash
python scripts/search/cifar10_locked_search.py
python scripts/evaluation/cifar10_locked_test.py
```

Replace `cifar10` with `cifar100` or `cinic10` in the three script names and output paths for those datasets. For non-IID and participation conditions, use the same dataset scripts with `--baseline-checkpoint`, `--deepfednas-checkpoint`, `--predictor-dir`, and `--output-dir` pointing to that condition's training and output directories. The search uses four strict MAC intervals, population 256, 512 generations, and seeds 42–46. The reported variation across these five searches is conditional on one fixed trained checkpoint per method.

The scripts write new artifacts under `outputs/`, including predictor datasets/models, locked architecture manifests, provenance, and full official-test results. Run `--help` on an individual Python script for its path and runtime options. Full retraining is GPU-intensive; the included cache and split/partition manifests avoid regenerating those inputs. To generate another cache, run `scripts/cache_generation/run_subnet_cache_generation.sh`; it writes a separate CSV and does not overwrite the supplied training cache.

## License

Apache-2.0; see [LICENSE](LICENSE).
