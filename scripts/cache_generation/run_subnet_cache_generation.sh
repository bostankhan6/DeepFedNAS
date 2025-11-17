#!/bin/bash
set -e

# 1. Robustly find the Project Root
#    (We are in scripts/cache_generation, so we go up 2 levels)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 2. Define Paths relative to the Root
#    Adjust these if your config/output folders are named differently
PYTHON_SCRIPT="$PROJECT_ROOT/src/nas/generate_subnet_cache.py"
CONFIG_PATH="$PROJECT_ROOT/configs/supernets/4-stage-supernet-deepfednas.json"
OUTPUT_CSV="$PROJECT_ROOT/results/4_stage_supernet_cache_60_subnets.csv"

# 3. Ensure Output Directory Exists
mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "========================================================================"
echo "Running Subnet Cache Generation"
echo "Root:   $PROJECT_ROOT"
echo "Script: $PYTHON_SCRIPT"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_CSV"
echo "========================================================================"

# 4. Run the Python Script
#    We use "$PROJECT_ROOT" to ensure python can find your 'src' modules if needed
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

python "$PYTHON_SCRIPT" \
    --arch_config_path "$CONFIG_PATH" \
    --output_csv "$OUTPUT_CSV" \
    --bounds_mode relative \
    --sampling_mode equidistant \
    --macs_lower_bound 458970000 \
    --macs_upper_bound 3403370000 \
    --num_samples 60 \
    --rho0_constraint 0.51 \
    --ga_pop_size 256 \
    --ga_generations 512 \
    --save_interval 20