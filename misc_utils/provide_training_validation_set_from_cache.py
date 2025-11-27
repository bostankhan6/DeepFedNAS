import numpy as np
import pandas as pd
import json
import sys
import os
import ast

# ==========================================================================================
# --- 1. CONFIGURATION ---
# ==========================================================================================

# --- Input and Output Files ---
# Path to your pre-computed optimal path cache
OPTIMAL_PATH_CACHE_PATH = "subnet_caches/4_stage_cache_60_subnets.csv"

# --- Validation Set Configuration ---
# The total number of subnets you want in your --diverse_subnets argument.
# This includes 2 boundaries + the number of anchors.
TOTAL_VALIDATION_SUBNETS = 4 

# ==========================================================================================
# --- 2. SCRIPT LOGIC ---
# ==========================================================================================

def create_validation_set():
    """
    Loads an optimal path cache, selects boundary and anchor subnets,
    and formats them for command-line use.
    """
    if not os.path.exists(OPTIMAL_PATH_CACHE_PATH):
        print(f"ERROR: Optimal path cache file not found at '{OPTIMAL_PATH_CACHE_PATH}'.")
        print("Please run the `generate_optimal_path_cache.py` script first.")
        sys.exit(1)

    print(f"Loading optimal path cache from: {OPTIMAL_PATH_CACHE_PATH}")
    df = pd.read_csv(OPTIMAL_PATH_CACHE_PATH)
    
    if len(df) < TOTAL_VALIDATION_SUBNETS:
        print(f"Warning: The cache contains only {len(df)} subnets, which is less than the requested {TOTAL_VALIDATION_SUBNETS}. Using all available subnets.")
        indices_to_sample = np.linspace(0, len(df) - 1, len(df), dtype=int)
    else:
        # Generate N equally spaced indices to sample from the cache
        indices_to_sample = np.linspace(0, len(df) - 1, TOTAL_VALIDATION_SUBNETS, dtype=int)

    print(f"Selecting {len(indices_to_sample)} subnets for the validation set...")
    
    selected_subnets_df = df.iloc[indices_to_sample]
    
    diverse_subnets = {}
    summary_data = []

    for i, (_, row) in enumerate(selected_subnets_df.iterrows()):
        # The architecture data is stored as strings, so we use ast.literal_eval
        # to safely convert them back to Python lists/dictionaries.
        arch_config = {
            "d": ast.literal_eval(row['d']),
            "e": ast.literal_eval(row['e']),
            "w_indices": ast.literal_eval(row['w_indices']),
        }
        diverse_subnets[str(i)] = arch_config
        
        label = "Smallest" if i == 0 else "Largest" if i == len(selected_subnets_df)-1 else f"Anchor {i}"
        
        summary_data.append({
            "Label": label,
            "MACs (M)": row['macs'] / 1e6,
            "Params (M)": row['params'] / 1e6,
            "Effectiveness (rho)": row['effectiveness'],
            "Entropy": row['fitness_score'] # Assuming 'fitness' column holds the entropy score
        })

    # --- Display a summary of the selected subnets ---
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("Summary of Selected Subnets for Validation (`--diverse_subnets`)")
    print(summary_df.to_string())
    print("="*80)

    # --- Construct the final command-line argument ---
    diverse_subnets_str = json.dumps(diverse_subnets)

    print("\nGenerated --diverse_subnets argument for your training command:")
    print(f"--diverse_subnets '{diverse_subnets_str}'")
    print("="*80)
    
    return diverse_subnets_str

if __name__ == "__main__":
    create_validation_set()
