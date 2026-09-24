#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON_BIN:-python3}" "$ROOT/src/deepfednas/nas/generate_subnet_cache.py" \
    --arch_config_path "$ROOT/configs/supernets/4-stage-supernet-deepfednas.json" \
    --output_csv "$ROOT/subnet_caches/generated_60_subnets.csv" \
    --bounds_mode relative \
    --sampling_mode equidistant \
    --macs_lower_bound 458970000 \
    --macs_upper_bound 3403370000 \
    --num_samples 60 \
    --rho0_constraint 0.51 \
    --ga_pop_size 1024 \
    --ga_generations 1024 \
    --save_interval 20
