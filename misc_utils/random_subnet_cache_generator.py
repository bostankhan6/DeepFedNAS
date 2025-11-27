import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import sys
import argparse

try:
    from deepfednas.nas.deepfednas_fitness_maximizer import (
        calculate_entropy_objective,
        calculate_L_and_avg_log_w,
        calculate_effectiveness_rho
    )
    from deepfednas.utils.subnet_cost import subnet_macs
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Please ensure your PYTHONPATH is set up correctly.")
    print(f"Import Error: {e}")
    sys.exit(1)

def generate_random_subnet_for_target(target_macs, arch_config, max_attempts=5000):
    """
    Generates a random subnet that is as close as possible to the target_macs without exceeding it.
    """
    best_subnet = None
    min_macs_diff = float('inf')

    # Extract choices from arch_config
    width_choices = arch_config['width_multiplier_choices']
    exp_choices = arch_config['expansion_ratio_choices']
    depth_choices = list(range(arch_config['max_extra_blocks_per_stage'] + 1))
    
    num_stages = arch_config['num_stages']
    num_exp_slots = num_stages * (arch_config['max_extra_blocks_per_stage'] + 1)
    num_w_indices = num_stages + 1

    for _ in range(max_attempts):
        # Generate a completely random architecture
        d_vec = np.random.choice(depth_choices, num_stages).tolist()
        e_vec = np.random.choice(exp_choices, num_exp_slots).tolist()
        w_indices = np.random.choice(len(width_choices), num_w_indices).tolist()

        # Calculate its MACs
        current_macs, _ = subnet_macs(d_vec, e_vec, w_indices, width_choices, arch_config)

        # Check if it's a better fit
        if current_macs <= target_macs:
            diff = target_macs - current_macs
            if diff < min_macs_diff:
                min_macs_diff = diff
                best_subnet = {
                    'd': d_vec,
                    'e': e_vec,
                    'w_indices': w_indices
                }
    
    return best_subnet

def main(args):
    print("--- Generating Random Baseline Subnet Cache ---")

    # --- Load Architecture Configuration ---
    print(f"Loading architecture config from: {args.arch_config_path}")
    with open(args.arch_config_path, 'r') as f:
        arch_config_params = json.load(f)
        arch_config_params['original_stage_base_channels'] = np.array(arch_config_params['original_stage_base_channels'])
        arch_config_params['alpha_weights'] = np.array(arch_config_params.get('alpha_weights', [1.0] * arch_config_params['num_stages']))

    # --- Determine Mode and Target MACs ---
    if args.optimized_cache_path and os.path.exists(args.optimized_cache_path):
        # Match Mode
        print(f"Mode: Match. Loading target MACs from: {args.optimized_cache_path}")
        df_optimized = pd.read_csv(args.optimized_cache_path)
        target_macs_list = df_optimized['macs'].tolist()
        print(f"Found {len(target_macs_list)} target MACs values to match.")
    else:
        # Generate Mode
        print(f"Mode: Generate. Will create {args.num_samples} random subnets.")
        # Calculate the absolute min and max MACs of the supernet
        num_stages = arch_config_params['num_stages']
        max_extra_blocks = arch_config_params['max_extra_blocks_per_stage']
        num_exp_slots = num_stages * (max_extra_blocks + 1)
        num_w_indices = num_stages + 1
        
        min_arch = {'d': [0]*num_stages, 'e': [min(arch_config_params['expansion_ratio_choices'])]*num_exp_slots, 'w_indices': [0]*num_w_indices}
        max_arch = {'d': [max_extra_blocks]*num_stages, 'e': [max(arch_config_params['expansion_ratio_choices'])]*num_exp_slots, 'w_indices': [len(arch_config_params['width_multiplier_choices'])-1]*num_w_indices}
        
        macs_min, _ = subnet_macs(min_arch['d'], min_arch['e'], min_arch['w_indices'], arch_config_params['width_multiplier_choices'], arch_config_params)
        macs_max, _ = subnet_macs(max_arch['d'], max_arch['e'], max_arch['w_indices'], arch_config_params['width_multiplier_choices'], arch_config_params)
        
        print(f"Supernet MACs Range: [{macs_min/1e6:.2f}M, {macs_max/1e6:.2f}M]")
        target_macs_list = np.random.uniform(macs_min, macs_max, args.num_samples)

    # --- Generate Random Subnets ---
    random_subnets_data = []
    
    with tqdm(total=len(target_macs_list), desc="Finding Random Subnets") as pbar:
        for target_macs in target_macs_list:
            random_arch = generate_random_subnet_for_target(target_macs, arch_config_params)
            
            if random_arch:
                final_macs, final_params = subnet_macs(
                    random_arch['d'], random_arch['e'], random_arch['w_indices'],
                    arch_config_params['width_multiplier_choices'], arch_config_params
                )
                
                L, avg_log_w = calculate_L_and_avg_log_w(
                    random_arch['d'], random_arch['e'], random_arch['w_indices'],
                    arch_config_params['width_multiplier_choices'], arch_config_params
                )
                rho = calculate_effectiveness_rho(L, avg_log_w)

                entropy, _ = calculate_entropy_objective(
                    random_arch['d'], random_arch['e'], random_arch['w_indices'],
                    arch_config_params['width_multiplier_choices'], arch_config_params
                )

                random_subnets_data.append({
                    'macs': final_macs, 'params': final_params,
                    'effectiveness': rho, 'entropy': entropy,
                    'd': str(random_arch['d']), 'e': str(random_arch['e']),
                    'w_indices': str(random_arch['w_indices'])
                })
            pbar.update(1)

    # --- Save the New Baseline Cache ---
    if random_subnets_data:
        df_random = pd.DataFrame(random_subnets_data)
        df_random = df_random.sort_values(by='macs').reset_index(drop=True)
        
        df_random['fitness_score'] = 0 # Placeholder for consistent column structure
        cols = ['macs', 'params', 'fitness_score', 'effectiveness', 'entropy', 'd', 'e', 'w_indices']
        df_random = df_random[cols]

        df_random.to_csv(args.output_csv, index=False)
        print(f"\nSuccessfully generated and saved random baseline cache to '{args.output_csv}'")
        print(f"Generated {len(df_random)} random subnets.")
    else:
        print("\nCould not generate any random subnets. Please check the configuration.")

if __name__ == "__main__":

    ARCH_PATH = '/cache_gen_scripts/4-stage-config.json'
    OUTPUT_CSV = 'random_baseline_cache.csv'
    NUM_SAMPLES = 100

    parser = argparse.ArgumentParser(description="Generate a cache of random subnets for baseline comparison.")
    parser.add_argument('--arch_config_path', type=str, default=ARCH_PATH, help='Path to the JSON file describing the supernet architecture.')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV, help='Path to save the output CSV file.')
    
    # Optional arguments for controlling the generation mode
    parser.add_argument('--optimized_cache_path', type=str, default=None, help='(Optional) Path to an existing optimized cache. If provided, the script will match the MACs targets from this file.')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES, help='Number of random subnets to generate. Only used if --optimized_cache_path is NOT provided.')
    
    args = parser.parse_args()
    main(args)