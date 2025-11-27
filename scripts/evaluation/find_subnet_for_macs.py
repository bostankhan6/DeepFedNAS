import argparse
import os
import sys
import json
import numpy as np

# --- Import Project's Modules ---
try:
    # Import the Genetic Algorithm from your existing script
    from deepfednas.nas.deepfednas_fitness_maximizer import run_entropy_max_ga
    # Import the MACs calculator to verify the final result
    from deepfednas.utils.subnet_cost import subnet_macs
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Please check your PYTHONPATH.")
    print(f"Import Error: {e}")
    sys.exit(1)

def find_subnet(args):
    """
    Main function to find the best subnet for a specific MACs value using a genetic algorithm.
    """
    # 1. Load and parse the architecture configuration
    if not os.path.exists(args.arch_config_path):
        raise FileNotFoundError(f"Architecture config file not found at: {args.arch_config_path}")
        
    print(f"Loading supernet architecture from: {args.arch_config_path}")
    with open(args.arch_config_path, 'r') as f:
        arch_params = json.load(f)

    # Prepare parameters for the GA function
    arch_params['original_stage_base_channels'] = np.array(arch_params.get('original_stage_base_channels', []))
    arch_params['alpha_weights'] = np.array(arch_params.get('alpha_weights', [1.0] * arch_params.get('num_stages', 0)))
    
    width_mult_options = arch_params.get('width_multiplier_choices', [1.0])
    exp_opt_values = np.array(arch_params.get('expansion_ratio_choices', [0.25]))
    depth_choices = np.array(list(range(arch_params.get('max_extra_blocks_per_stage', 0) + 1)))
    
    # Convert target MACs from millions to raw value
    target_mac_budget = args.target_macs_m * 1e6

    print(f"\nStarting search for an architecture with MACs <= {args.target_macs_m:.2f}M...")

    # 2. Run the Genetic Algorithm to find the best subnet
    best_architecture, best_fitness = run_entropy_max_ga(
        mac_budget=target_mac_budget,
        arch_config_params=arch_params,
        width_mult_options=width_mult_options,
        depth_choices=depth_choices,
        exp_opt_values=exp_opt_values,
        rho0_constraint=arch_params.get('supernet_rho0_constraint', 2.0),
        effectiveness_fitness_weight=arch_params.get('supernet_effectiveness_fitness_weight', 1000),
        pop_size=args.population_size,
        generations=args.generations,
        mutate_p=0.3,
        seed=args.seed
    )

    # 3. Report the results
    if best_architecture is None:
        print("\n--- Search Failed ---")
        print("Could not find a valid architecture. The MACs constraint might be too low or the search parameters too restrictive.")
        print("---------------------")
        return

    # 4. Verify the MACs and Params of the found architecture
    final_macs, final_params = subnet_macs(
        depth_vec=best_architecture['d'],
        exp_vec=best_architecture['e'],
        w_indices=best_architecture['w_indices'],
        width_mult_options=width_mult_options,
        arch_config_params=arch_params
    )

    print("\n--- Search Complete: Best Subnet Found ---")
    print(f"Target MACs Constraint: <= {args.target_macs_m:,.2f} M")
    print("-" * 40)
    print(f"Actual MACs:            {final_macs/1e6:,.2f} M")
    print(f"Total Parameters:       {final_params/1e6:,.2f} M")
    print(f"Achieved Fitness Score: {best_fitness:.4f}")
    print("-" * 40)
    print("Architecture Configuration:")
    print(json.dumps(best_architecture, indent=4))
    print("------------------------------------------")


if __name__ == "__main__":
    
    # =================================================================================
    # --- CONFIGURATION SECTION: Edit the values below for your specific experiment ---
    # =================================================================================

    # Path to the JSON file describing the supernet architecture.
    ARCH_CONFIG_PATH = "configs/supernets/4-stage-supernet-deepfednas.json"
    
    # The target computational budget in Million MACs (e.g., 500.0).
    TARGET_MACS_M = 1445.0
    
    # --- Optional GA Parameters (for tuning the search) ---
    POPULATION_SIZE = 256
    GENERATIONS = 256
    SEED = 42

    # =================================================================================
    # --- End of Configuration ---
    # =================================================================================

    # Create a configuration object from the hardcoded values
    args = argparse.Namespace(
        arch_config_path=ARCH_CONFIG_PATH,
        target_macs_m=TARGET_MACS_M,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seed=SEED,
    )

    # --- Basic Validation ---
    if not os.path.exists(args.arch_config_path):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! ERROR: The specified arch_config_path does not exist: !!!")
        print(f"!!! '{args.arch_config_path}'")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

    find_subnet(args)