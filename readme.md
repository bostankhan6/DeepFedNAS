
# DeepFedNAS: A Unified Framework for Principled Supernet Training and Predictor-Free Federated Architecture Search

**DeepFedNAS** is a novel Federated Neural Architecture Search (FedNAS) framework designed to overcome the inefficiencies of unguided supernet training and the high costs of post-training subnet discovery.

By integrating a **principled, multi-objective fitness function** inspired by mathematical network design, DeepFedNAS introduces two core innovations: **Federated Pareto Optimal Supernet Training** and a **Predictor-Free Search Method**. This framework achieves state-of-the-art accuracy (e.g., +1.21% on CIFAR-100) and delivers a **\~61x speedup** in the architecture search pipeline compared to existing baselines like SuperFedNAS.

-----

## 🚀 Key Features

  * **Principled Multi-Objective Fitness Function ($\mathcal{F}(\mathcal{A})$):** Synthesizes network entropy, effectiveness, and architectural heuristics (depth uniformity, channel monotonicity) into a single metric to guide optimization.
  * **Federated Pareto Optimal Supernet Training:** Replaces random "sandwich" sampling with a curriculum of elite, high-fitness architectures (the "Pareto Path"), ensuring the supernet weights are conditioned for optimal performance.
  * **Predictor-Free Search:** Eliminates the costly data collection and training of accuracy predictors. DeepFedNAS uses the principled fitness function as a zero-cost proxy for accuracy, enabling on-demand subnet discovery in seconds.
  * **Re-Engineered Generic Supernet:** A flexible ResNet-based supernet design that significantly expands the search space ($~1.98 \times 10^{15}$ architectures) to support fine-grained optimization.

-----

## 📂 Repository Structure

  * **`configs/`**: JSON configuration files defining the supernet search spaces (e.g., `4-stage-supernet-deepfednas.json`).
  * **`data/`**: Scripts and storage for datasets (CIFAR-10, CIFAR-100, CINIC-10).
  * **`experiments/`**: Shell scripts serving as entry points for running training experiments.
      * `01_baseline/`: Entry points for the baseline SuperFedNAS method.
      * `02_deepfednas/`: Entry points for the proposed DeepFedNAS method.
  * **`misc_utils/`**: Helper scripts for bounds calculation and subnet set generation for validation.
  * **`scripts/`**: Scripts for different purposes
      * `cache_generation/`: Scripts to generate the Pareto-optimal subnet cache.
      * `data_setup/`: Scripts to download and extract datasets.
      * `evaluation/`: Scripts for post-training search and analysis.
      * `search/`: (Legacy) Predictor-based search scripts for baseline comparison.
  * **`src/deepfednas/`**: Core source code.
      * `Client/`: Client-side local training logic.
      * `Server/`: Server-side aggregation (MaxNet) and sampling strategies.
      * `elastic_nn/`: Dynamic supernet model definitions (`GenericOFAResNet`).
      * `nas/`: Implementation of the fitness function and genetic algorithms.
      * `utils/`: Contains the code for cost calculation

-----

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/DeepFedNAS.git
    cd DeepFedNAS
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Datasets:**
    Use the scripts in `scripts/data_setup/` to download the required datasets:

    ```bash
    bash scripts/data_setup/download_cifar10.sh
    # Also available: download_cifar100.sh, download_cinic10.sh
    ```

-----

## ⚡ Usage Workflow

The DeepFedNAS workflow consists of three distinct phases, mirroring the methodology described in the paper.

### Phase 1: Offline Pareto Optimal Subnet Search (Cache Generation)

Before training, generate the "Pareto Path" cache. This uses the principled fitness function to find a set of elite architectures across the computational budget.

```bash
# Generates the optimal path cache (e.g., 60 subnets)
bash scripts/cache_generation/run_subnet_cache_generation.sh
```

  * **Output:** A CSV file (e.g., `subnet_caches/4_stage_cache_60_subnets.csv`) containing the optimal architectures used to guide supernet training.
  * **Note:** A csv file is already provided by the authors in the repository so you can **skip** this step.

### Phase 2: Federated Supernet Training

Train the supernet using the generated cache as a curriculum.

**Run DeepFedNAS (Proposed Method):**

```bash
# Uses --subnet_dist_type TS_optimal_path and the generated cache
bash experiments/02_deepfednas/cifar10.sh
```

  * *Key Arguments:* `TS_optimal_path` enables the Pareto path sampler. The script points to the cache generated in Phase 1.

**Run SuperFedNAS (Baseline):**

```bash
# Uses --subnet_dist_type TS_all_random (Standard Sandwich Rule)
bash experiments/01_baseline/cifar10.sh
```

### Phase 3: Predictor-Free Deployment Search

After training, find the optimal subnet for a specific hardware constraint (e.g., MACs limit) using the zero-cost fitness proxy. This replaces the multi-hour predictor training pipeline.

```bash
python scripts/evaluation/find_subnet_for_macs.py \
    --arch_config_path configs/supernets/4-stage-supernet-deepfednas.json \
    --target_macs_m 500.0 \
    --population_size 256 \
    --generations 100
```

  * This script runs a fast Genetic Algorithm maximizing $\mathcal{F}(\mathcal{A})$ subject to the MACs constraint, returning the optimal subnet configuration in seconds.

### Phase 4: Comprehensive Benchmarking (Pareto Frontier Generation)

To reproduce the full experimental results (e.g., the Pareto curves in the paper) and validate the supernet across the entire computational spectrum, use the `deepfednas_search.py` script.

**Functionality:**

  * Performs a batch search across a range of MACs targets (e.g., 0.4B to 3.7B).
  * Evaluates discovered subnets on the Test Set.
  * Measures **True Latency** on the running hardware.
  * Generates detailed CSV reports for analysis.

**Configuration:**
Open `scripts/search/deepfednas_search.py` and modify the **CONFIGURATION SECTION** at the top of the file to match your setup:

```python
# --- Example Configuration in deepfednas_search.py ---
MODEL_PATHS = ["trained_models/4-stage_continued_cached_60_subnets.pt"] # Path to your trained supernet
DATASET_NAME = 'cifar10'
TARGET_DEVICE_FOR_SEARCH = 'cuda' # 'cuda' or 'cpu'
LPM_GPU_MODEL_PATH = "path/to/your/lpm_model.pth" # Optional: For latency-constrained search
```

**Run the Benchmark:**

```bash
python scripts/search/deepfednas_search.py
```

**Output:**
The script saves results to `evaluation/latency_prediction/`, including:

  * `deepfednas_subnet_details_...csv`: Detailed metrics for every subnet found (Architecture, MACs, Params, Test Acc, True Latency).
  * `deepfednas_summary_results_...csv`: Aggregated statistics for each target budget.

-----

## 📊 Results

DeepFedNAS demonstrates significant improvements over the baseline SuperFedNAS framework across multiple metrics, including search efficiency, model accuracy, and robustness to non-IID data.

### 1\. Search Efficiency & Speedup

DeepFedNAS eliminates the need for expensive accuracy predictor training, reducing the total search pipeline time by **\~61x**.

| Search Pipeline Stage | Baseline (SuperFedNAS) | **DeepFedNAS (Ours)** |
| :--- | :--- | :--- |
| Cache Generation Stage | N/A | \~20 Minutes |
| Predictor Data Generation | \~20.65 Hours | **N/A** |
| Total Pipeline Time | \~20.65 Hours | **\~20 Minutes** |
| **Search Method** | Accuracy Predictor | **Fitness Function Proxy** |

### 2\. Accuracy vs. Computational Budget

DeepFedNAS consistently outperforms the baseline across various MACs constraints on benchmark datasets[cite: 881].

| Dataset | MACs Budget (B) | Baseline Acc (%) | **DeepFedNAS Acc (%)** | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **CIFAR-100** | 0.95 - 1.45 | 61.66% | **62.87%** | **+1.21%** |
| **CIFAR-100** | 2.45 - 3.75 | 62.30% | **63.20%** | **+0.90%** |
| **CIFAR-10** | 0.45 - 0.95 | 93.47% | **94.16%** | **+0.69%** |
| **CINIC-10** | 1.45 - 2.45 | 77.09% | **77.80%** | **+0.71%** |

### 3\. Robustness to Non-IID Data

DeepFedNAS is significantly more robust to statistical heterogeneity (Non-IID data) compared to the SuperFedNAS baseline.

| Non-IID Degree ($\alpha$) | Baseline Acc (%) | **DeepFedNAS Acc (%)** |
| :--- | :--- | :--- |
| **$\alpha = 100$ (Low)** | 93.72% | **94.51%** |
| **$\alpha = 1.0$ (Med)** | 92.63% | **93.33%** |
| **$\alpha = 0.1$ (High)** | 86.00% | **86.83%** |

### 4\. Parameter Efficiency

DeepFedNAS discovers subnets that achieve higher accuracy with significantly fewer parameters.

  * **CIFAR-100 Example:** DeepFedNAS achieves **62.60%** accuracy with only **19.43M** parameters, whereas the baseline requires **55.03M** parameters to achieve a lower accuracy of **62.22%**.

-----

## 📜 Citation

If you use this code or framework in your research, please cite our paper:

```bibtex
@misc{khan2026deepfednasunifiedframeworkprincipled,
      title={DeepFedNAS: A Unified Framework for Principled, Hardware-Aware, and Predictor-Free Federated Neural Architecture Search}, 
      author={Bostan Khan and Masoud Daneshtalab},
      year={2026},
      eprint={2601.15127},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.15127}, 
}
```
