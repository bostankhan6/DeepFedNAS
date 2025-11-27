import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ast
import json

# --- Import Project's Modules ---
try:
    # Import the original data loaders to get their transformation pipelines
    from deepfednas.data.cifar10.data_loader import _data_transforms_cifar10
    from deepfednas.data.cifar100.data_loader import _data_transforms_cifar100
    from deepfednas.data.cinic10.data_loader import _data_transforms_cinic10
    from deepfednas.Server.generic_server_model import GenericServerOFA
    from deepfednas.utils.subnet_cost import subnet_macs
    from deepfednas.Client.subnet_trainer import SubnetTrainer
    from deepfednas.nas.deepfednas_fitness_maximizer import (
        calculate_L_and_avg_log_w,
        calculate_effectiveness_rho,
        calculate_entropy_objective,
    )
    # Import base components for our custom loader
    from torchvision import datasets
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Please check your PYTHONPATH and file structure.")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Helper Functions ---

# Custom Dataset class for the cached data
class CachedImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def load_supernet_from_checkpoint(path, device, num_classes):
    """Loads a GenericServerOFA model from a saved checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
    
    print(f"Loading supernet from: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if "arch_params" not in checkpoint:
        raise ValueError(f"Checkpoint '{path}' is missing the required 'arch_params' dictionary.")
    
    arch_params = checkpoint["arch_params"]
    arch_params['n_classes'] = num_classes
    
    model = GenericServerOFA(
        arch_params=arch_params,
        sampling_method='TS_all_random',
        num_cli_total=1,
    )
    
    model.set_model_params(checkpoint["params"])
    model.to(device)
    model.eval()
    print(f"Supernet loaded and configured for {num_classes} classes.")
    return model

def get_test_loader(args):
    """
    Loads the correct test data loader for the specified dataset,
    using a caching mechanism for large ImageFolder datasets like CINIC-10.
    """
    print(f"Preparing {args.dataset.upper()} test data loader...")
    
    # Define a cache path for image folder datasets
    cache_path = os.path.join(args.data_dir, f"{args.dataset}_test_cache.pt")

    if args.dataset == 'cifar100':
        _, transform = _data_transforms_cifar100()
        dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        _, transform = _data_transforms_cifar10()
        dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'cinic10':
        _, transform = _data_transforms_cinic10()
        if os.path.exists(cache_path):
            print(f"Loading {args.dataset.upper()} from cache: {cache_path}")
            samples = torch.load(cache_path)
            dataset = CachedImageDataset(samples, transform=transform)
        else:
            print("Scanning CINIC-10 directory for the first time (this may take a few minutes)...")
            test_dir = os.path.join(args.data_dir, 'test')
            if not os.path.isdir(test_dir):
                 raise FileNotFoundError(f"CINIC-10 test directory not found at {test_dir}.")
            image_folder_dataset = datasets.ImageFolder(root=test_dir)
            print(f"Found {len(image_folder_dataset.samples)} images. Saving to cache...")
            torch.save(image_folder_dataset.samples, cache_path)
            print("Cache saved successfully.")
            dataset = CachedImageDataset(image_folder_dataset.samples, transform=transform)
    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported.")

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print("Test loader created successfully.")
    return test_loader

# ... (The rest of your helper functions: evaluate_subnet, decode_arch_from_row, get_pareto_frontier, remain the same) ...

def evaluate_subnet(subnet_model, test_loader, device, args):
    """Evaluates a single subnet on the test dataset."""
    eval_args = argparse.Namespace(
        dataset=args.dataset,
        data_dir=args.data_dir, batch_size=args.batch_size, device=device,
        use_bn=True, feddyn_alpha=0.0, model='generic', reset_bn_stats=False,
        reset_bn_stats_test=False, verbose_test=False, ci=0, kd_ratio=0.0,
        n_words=0, n_chars=0, max_norm=10.0, verbose=False, validseqlen=40, seq_len=80
    )
    eval_trainer = SubnetTrainer(model=subnet_model, device=device, args=eval_args)
    eval_trainer.update_local_dataset(0, None, test_loader, 0)
    metrics = eval_trainer.local_test(True)
    accuracy = metrics["test_correct"] / metrics["test_total"]
    return accuracy

def decode_arch_from_row(row):
    """Decodes the architecture from a row of the CSV file."""
    try:
        d_vec = ast.literal_eval(row['d'])
        e_vec = ast.literal_eval(row['e'])
        w_indices = ast.literal_eval(row['w_indices'])
        return {'d': d_vec, 'e': e_vec, 'w_indices': w_indices}
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing architecture from row {row.name}: {e}")
        return None

def get_pareto_frontier(df, cost_col, acc_col):
    """Calculates the Pareto frontier from a dataframe."""
    df_sorted = df.sort_values(by=cost_col).copy()
    pareto_points = []
    max_acc_so_far = -1.0
    
    for _, row in df_sorted.iterrows():
        if row[acc_col] > max_acc_so_far:
            pareto_points.append(row)
            max_acc_so_far = row[acc_col]
            
    return pd.DataFrame(pareto_points)

def main(args):
    """Main function to run the single supernet analysis."""
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 100 if args.dataset == 'cifar100' else 10
    supernet = load_supernet_from_checkpoint(args.model_path, device, num_classes)
    
    print(f"Loading architectures from {args.arch_csv_path}")
    arch_df = pd.read_csv(args.arch_csv_path)
    
    if args.num_subnets >= len(arch_df):
        sampled_arch_df = arch_df
    else:
        indices = np.linspace(0, len(arch_df) - 1, args.num_subnets, dtype=int)
        sampled_arch_df = arch_df.iloc[indices]
    print(f"Selected {len(sampled_arch_df)} subnets for evaluation.")
    
    test_loader = get_test_loader(args)

    results = []
    
    ref_arch_params = supernet.arch_params
    width_mult_options = ref_arch_params['width_multiplier_choices']

    for _, row in tqdm(sampled_arch_df.iterrows(), total=len(sampled_arch_df), desc="Evaluating Subnets"):
        arch_config = decode_arch_from_row(row)
        if arch_config is None:
            continue
            
        subnet_model = supernet.get_subnet(**arch_config, preserve_weight=True)
        accuracy = evaluate_subnet(subnet_model, test_loader, device, args)

        macs, params = subnet_macs(arch_config['d'], arch_config['e'], arch_config['w_indices'], width_mult_options=width_mult_options, arch_config_params=ref_arch_params)
        
        L, avg_log_w = calculate_L_and_avg_log_w(arch_config['d'], arch_config['e'], arch_config['w_indices'], width_mult_options=width_mult_options, arch_config_params=ref_arch_params)
        rho = calculate_effectiveness_rho(L, avg_log_w)
        entropy, _ = calculate_entropy_objective(arch_config['d'], arch_config['e'], arch_config['w_indices'], width_mult_options=width_mult_options, arch_config_params=ref_arch_params)
        
        results.append({
            'arch_id': row.name,
            'accuracy': accuracy,
            'macs_M': macs / 1e6,
            'params_M': params / 1e6,
            'rho': rho,
            'entropy': entropy
        })

    results_df = pd.DataFrame(results)
    
    print("\n--- Analysis Results ---")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"All analysis artifacts will be saved to: {args.output_dir}")
    
    plt.figure(figsize=(12, 8))
    pareto_frontier = get_pareto_frontier(results_df, 'macs_M', 'accuracy')
    sns.scatterplot(data=results_df, x='macs_M', y='accuracy', label="All Evaluated Subnets", alpha=0.5, color='gray')
    plt.plot(pareto_frontier['macs_M'], pareto_frontier['accuracy'], marker='o', linestyle='-', color='red', label="Pareto Frontier")
    plt.xlabel("Computational Cost (MACs in Millions)")
    plt.ylabel("Subnet Accuracy")
    plt.title(f"Accuracy vs. MACs for {args.model_name} on {args.dataset.upper()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{args.dataset}_accuracy_vs_macs_pareto.png'))
    plt.close()
    print(f"Saved accuracy vs. MACs plot with Pareto frontier.")

    print("\nPerformance Summary:")
    print(results_df[['accuracy', 'macs_M', 'params_M', 'rho', 'entropy']].describe())

    results_csv_path = os.path.join(args.output_dir, f'{args.dataset}_analysis_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved detailed analysis results to: {results_csv_path}")


if __name__ == "__main__":
    # =================================================================================
    # --- CONFIGURATION SECTION: Edit the values below for your specific experiment ---
    # =================================================================================
    
    # --- Choose the dataset ---
    DATASET = "cinic10" # Options: "cifar10", "cifar100", "cinic10"

    MODEL_PATH = "results/cinic10_cached_60_subnets.pt"
    MODEL_NAME = "cinic10_cached_60_subnets"
    ARCH_CSV_PATH = "results/4_stage_cache_60_subnets.csv"
    OUTPUT_DIR = f"results/single_analysis_results/{MODEL_NAME}"
    NUM_SUBNETS_TO_EVAL = 60
    DATA_DIR = "/data" # Root data directory
    BATCH_SIZE = 2*2048
    GPU_ID = 1
    # =================================================================================
    
    # Create a configuration object from the hardcoded values
    args = argparse.Namespace(
        dataset=DATASET,
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        arch_csv_path=ARCH_CSV_PATH,
        num_subnets=NUM_SUBNETS_TO_EVAL,
        data_dir=os.path.join(DATA_DIR, DATASET), # Automatically create dataset-specific path
        batch_size=BATCH_SIZE,
        gpu=GPU_ID,
        output_dir=OUTPUT_DIR,
    )

    main(args)