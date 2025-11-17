import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr, ttest_rel
import ast # Import ast for literal_eval

# --- Path Setup ---
# Ensure the script can find your custom modules by adjusting the path as needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# --- Import Your Project's Modules ---
try:
    from deepfednas.data.cifar10.data_loader import load_partition_data_cifar10
    from deepfednas.Server.generic_server_model import GenericServerOFA
    from deepfednas.utils.subnet_cost import subnet_macs
    from deepfednas.Client.subnet_trainer import SubnetTrainer
    from deepfednas.nas.deepfednas_fitness_maximizer import (
        calculate_L_and_avg_log_w,
        calculate_effectiveness_rho,
        calculate_entropy_objective,
    )
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Please check your PYTHONPATH and file structure.")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Helper Functions ---

def load_supernet_from_checkpoint(path, device):
    """Loads a GenericServerOFA model from a saved checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
    
    print(f"Loading supernet from: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if "arch_params" not in checkpoint:
        raise ValueError(f"Checkpoint '{path}' is missing the required 'arch_params' dictionary.")
    
    arch_params = checkpoint["arch_params"]
    
    model = GenericServerOFA(
        arch_params=arch_params,
        sampling_method='TS_all_random',
        num_cli_total=1,
    )
    
    model.set_model_params(checkpoint["params"])
    model.to(device)
    model.eval()
    print("Supernet loaded successfully.")
    return model

def get_cifar10_test_loader(args):
    """Loads the CIFAR-10 test data loader."""
    _, _, _, test_data_global, _, _, _, _ = load_partition_data_cifar10(
        args.dataset,
        args.data_dir,
        partition_method='homo',
        partition_alpha=1.0,
        client_number=1,
        batch_size=args.batch_size,
    )
    return test_data_global

def evaluate_subnet(subnet_model, test_loader, device, args):
    """Evaluates a single subnet on the test dataset."""
    eval_trainer = SubnetTrainer(model=subnet_model, device=device, args=args)
    
    eval_trainer.update_local_dataset(
        0, None, test_loader, 0
    )
    
    metrics = eval_trainer.local_test(True)
    
    accuracy = metrics["test_correct"] / metrics["test_total"]
    return accuracy

def decode_arch_from_row(row):
    """
    Decodes the architecture from a row of the new CSV format.
    It reads 'd', 'e', and 'w_indices' columns containing list-like strings.
    """
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

# --- Main Analysis Script ---

def main(args):
    """Main function to run the supernet analysis."""
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load models and data
    supernet_A = load_supernet_from_checkpoint(args.model_a_path, device)
    supernet_B = load_supernet_from_checkpoint(args.model_b_path, device)
    
    print(f"Loading architectures from {args.arch_csv_path}")
    arch_df = pd.read_csv(args.arch_csv_path)
    
    indices = np.linspace(0, len(arch_df) - 1, args.num_subnets, dtype=int)
    sampled_arch_df = arch_df.iloc[indices]
    print(f"Selected {len(sampled_arch_df)} subnets for evaluation.")
    
    eval_args = argparse.Namespace(
        dataset='cifar10', data_dir=args.data_dir, batch_size=args.batch_size, device=device,
        use_bn=True, feddyn_alpha=0.0, model='generic', reset_bn_stats=False,
        reset_bn_stats_test=False, verbose_test=False, ci=0, kd_ratio=0.0,
        n_words=0, n_chars=0
    )
    test_loader = get_cifar10_test_loader(eval_args)

    # 2. Evaluate all subnets and store rich results
    results = []
    
    ref_arch_params = supernet_A.arch_params
    width_mult_options = ref_arch_params['width_multiplier_choices']

    for _, row in tqdm(sampled_arch_df.iterrows(), total=len(sampled_arch_df), desc="Evaluating Subnets"):
        arch_config = decode_arch_from_row(row)
        if arch_config is None:
            continue
            
        subnet_A = supernet_A.get_subnet(**arch_config, preserve_weight=True)
        acc_A = evaluate_subnet(subnet_A, test_loader, device, eval_args)

        subnet_B = supernet_B.get_subnet(**arch_config, preserve_weight=True)
        acc_B = evaluate_subnet(subnet_B, test_loader, device, eval_args)

        macs, params = subnet_macs(arch_config['d'], arch_config['e'], arch_config['w_indices'], width_mult_options=width_mult_options, arch_config_params=ref_arch_params)
        
        L, avg_log_w = calculate_L_and_avg_log_w(arch_config['d'], arch_config['e'], arch_config['w_indices'], width_mult_options=width_mult_options, arch_config_params=ref_arch_params)
        rho = calculate_effectiveness_rho(L, avg_log_w)
        entropy, _ = calculate_entropy_objective(arch_config['d'], arch_config['e'], arch_config['w_indices'], width_mult_options=width_mult_options, arch_config_params=ref_arch_params)
        
        results.append({
            'arch_id': row.name,
            'acc_A': acc_A,
            'acc_B': acc_B,
            'macs_M': macs / 1e6,
            'params_M': params / 1e6,
            'rho': rho,
            'entropy': entropy
        })

    results_df = pd.DataFrame(results)
    
    # --- 3. Perform and Visualize Analyses ---
    
    print("\n--- Analysis Results ---")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"All analysis artifacts will be saved to: {args.output_dir}")
    
    # A. Direct Performance Comparison
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=results_df, x='acc_A', y='acc_B')
    plt.plot([min(results_df['acc_A'].min(), results_df['acc_B'].min()) * 0.99, 1], 
             [min(results_df['acc_A'].min(), results_df['acc_B'].min()) * 0.99, 1], 
             ls="--", c=".3")
    plt.xlabel(f"Accuracy from {args.model_a_name}")
    plt.ylabel(f"Accuracy from {args.model_b_name}")
    plt.title("Direct Subnet Performance Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'direct_performance_comparison.png'))
    plt.close()
    print(f"Saved direct performance plot.")

    # B. Performance vs. MACs (Pareto Front) with Explicit Frontier Lines
    plt.figure(figsize=(12, 8))
    
    pareto_A = get_pareto_frontier(results_df, 'macs_M', 'acc_A')
    pareto_B = get_pareto_frontier(results_df, 'macs_M', 'acc_B')
    
    sns.scatterplot(data=results_df, x='macs_M', y='acc_A', label=f"{args.model_a_name} (All Subnets)", alpha=0.5, color='lightblue')
    sns.scatterplot(data=results_df, x='macs_M', y='acc_B', label=f"{args.model_b_name} (All Subnets)", alpha=0.5, color='navajowhite')
    
    plt.plot(pareto_A['macs_M'], pareto_A['acc_A'], marker='o', linestyle='-', color='blue', label=f"{args.model_a_name} (Pareto Frontier)")
    plt.plot(pareto_B['macs_M'], pareto_B['acc_B'], marker='s', linestyle='-', color='darkorange', label=f"{args.model_b_name} (Pareto Frontier)")
    
    plt.xlabel("Computational Cost (MACs in Millions)")
    plt.ylabel("Subnet Accuracy")
    plt.title("Accuracy vs. MACs: Comparing Pareto Frontiers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'accuracy_vs_macs_pareto.png'))
    plt.close()
    print(f"Saved accuracy vs. MACs plot with explicit Pareto frontiers.")

    # C & D. Statistical Analysis
    corr, p_value = spearmanr(results_df['acc_A'], results_df['acc_B'])
    print(f"\nSpearman's Rank Correlation: {corr:.4f} (p-value: {p_value:.4f})")
    if p_value < 0.05: print(" -> The correlation is statistically significant.")
    
    avg_A = results_df['acc_A'].mean()
    avg_B = results_df['acc_B'].mean()
    print(f"\nAverage Accuracy ({args.model_a_name}): {avg_A:.4f}")
    print(f"Average Accuracy ({args.model_b_name}): {avg_B:.4f}")
    
    if avg_B > avg_A:
        t_stat, p_val_ttest = ttest_rel(results_df['acc_B'], results_df['acc_A'])
        print(f"Paired T-test (B vs A): t-statistic = {t_stat:.4f}, p-value = {p_val_ttest:.4f}")
        if p_val_ttest < 0.05:
            print(f" -> The improvement shown by '{args.model_b_name}' is statistically significant.")
        else:
            print(f" -> The improvement shown by '{args.model_b_name}' is NOT statistically significant.")
    
    results_csv_path = os.path.join(args.output_dir, 'analysis_results_with_metrics.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved detailed analysis results (including rho and entropy) to: {results_csv_path}")


if __name__ == "__main__":
    # --- CONFIGURATION SECTION: Hardcode your paths and settings here ---
    MODEL_A_PATH = "/home/bostan/projects/superfednas/fedml_experiments/standalone/superfednas/trained_models/cifar-100_4-stage_Baseline_TS_all_random.pt"
    MODEL_B_PATH = "/home/bostan/projects/superfednas/fedml_experiments/standalone/superfednas/trained_models/cifar100_cached_60_subnets_p1024_g1024_w-fine-grained.pt"
    SUPERNET_A = "4-Stage_TS_all_random" 
    SUPERNET_B = "cifar100_cached_60_subnets_p1024_g1024_w-fine-grained" 
    ARCH_CSV_PATH = "/home/bostan/projects/superfednas/4_stage_cache_60_subnets_p1024_g1024_w-fine-grained.csv"
    OUTPUT_DIR = "/home/bostan/projects/superfednas/fedml_experiments/standalone/superfednas/evaluation/eval_results/cifar100_cached_60_subnets_p1024_g1024_w-fine-grained"
    NUM_SUBNETS_TO_EVAL = 20
    
    parser = argparse.ArgumentParser(description="Supernet Analysis Script")
    parser.add_argument("--model_a_path", type=str, default=MODEL_A_PATH, help="Path to checkpoint for Supernet A.")
    parser.add_argument("--model_b_path", type=str, default=MODEL_B_PATH, help="Path to checkpoint for Supernet B.")
    parser.add_argument("--model_a_name", type=str, default=SUPERNET_A, help="Display name for Model A.")
    parser.add_argument("--model_b_name", type=str, default=SUPERNET_B, help="Display name for Model B.")
    parser.add_argument("--arch_csv_path", type=str, default=ARCH_CSV_PATH, help="Path to the CSV file containing architectures to test.")
    parser.add_argument("--num_subnets", type=int, default=NUM_SUBNETS_TO_EVAL, help="Number of subnets to sample and evaluate.")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10", help="Path to the CIFAR-10 data directory.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for evaluation.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save analysis plots and results CSV.")
    
    args = parser.parse_args()
    main(args)
