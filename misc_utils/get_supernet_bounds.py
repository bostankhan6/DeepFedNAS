import argparse
import os
import sys
import json
import numpy as np

try:
    # Import the core calculation utility from your existing codebase
    from deepfednas.utils.subnet_cost import subnet_macs
except ImportError as e:
    print(f"ERROR: Could not import the 'subnet_macs' function. Please ensure your PYTHONPATH is correct.")
    print(f"Import Error: {e}")
    sys.exit(1)

def main(args):
    """
    Main function to calculate and display the MACs and parameter bounds of a supernet.
    """
    
    # 1. Load and parse the architecture configuration from the JSON file
    if not os.path.exists(args.arch_config_path):
        raise FileNotFoundError(f"Architecture config file not found at: {args.arch_config_path}")
        
    print(f"Loading supernet architecture from: {args.arch_config_path}")
    with open(args.arch_config_path, 'r') as f:
        arch_params = json.load(f)

    # Ensure required keys are present
    required_keys = ['num_stages', 'max_extra_blocks_per_stage', 'expansion_ratio_choices', 'width_multiplier_choices']
    for key in required_keys:
        if key not in arch_params:
            raise ValueError(f"Architecture config is missing the required key: '{key}'")
    
    # For compatibility with subnet_macs, ensure some values are numpy arrays
    if 'original_stage_base_channels' in arch_params:
        arch_params['original_stage_base_channels'] = np.array(arch_params['original_stage_base_channels'])

    print("Architecture loaded successfully.")

    # 2. Construct the MINIMUM possible subnet architecture
    num_stages = arch_params['num_stages']
    num_exp_slots = num_stages * (arch_params['max_extra_blocks_per_stage'] + 1)
    
    min_arch_config = {
        'd': [0] * num_stages,
        'e': [min(arch_params['expansion_ratio_choices'])] * num_exp_slots,
        'w_indices': [0] * (num_stages + 1) # 0 is the index of the smallest width multiplier
    }

    # 3. Construct the MAXIMUM possible subnet architecture
    max_arch_config = {
        'd': [arch_params['max_extra_blocks_per_stage']] * num_stages,
        'e': [max(arch_params['expansion_ratio_choices'])] * num_exp_slots,
        'w_indices': [len(arch_params['width_multiplier_choices']) - 1] * (num_stages + 1) # Last index
    }

    # 4. Calculate MACs and Parameters for both architectures
    print("\nCalculating bounds...")
    
    min_macs, min_params = subnet_macs(
        depth_vec=min_arch_config['d'],
        exp_vec=min_arch_config['e'],
        w_indices=min_arch_config['w_indices'],
        width_mult_options=arch_params['width_multiplier_choices'],
        arch_config_params=arch_params
    )
    
    max_macs, max_params = subnet_macs(
        depth_vec=max_arch_config['d'],
        exp_vec=max_arch_config['e'],
        w_indices=max_arch_config['w_indices'],
        width_mult_options=arch_params['width_multiplier_choices'],
        arch_config_params=arch_params
    )

    # 5. Report the results
    print("\n--- Supernet Search Space Bounds ---")
    print(f"Configuration File: {args.arch_config_path}")
    print("-" * 35)
    print("Minimum Possible Subnet (Smallest):")
    print(f"  - MACs:     {min_macs/1e6:,.2f} M")
    print(f"  - Params:   {min_params/1e6:,.2f} M")
    print("-" * 35)
    print("Maximum Possible Subnet (Largest):")
    print(f"  - MACs:     {max_macs/1e6:,.2f} M")
    print(f"  - Params:   {max_params/1e6:,.2f} M")
    print("-" * 35)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the MACs and Parameter bounds for a given supernet architecture.")
    
    ARCH_PATH = '/home/bostan/projects/superfednas/fedml_api/standalone/superfednas/cache_gen_scripts/4-stage-config.json'

    parser.add_argument("--arch_config_path", type=str, default=ARCH_PATH,
                        help="Path to the JSON file describing the supernet architecture.")
    
    args = parser.parse_args()
    main(args)