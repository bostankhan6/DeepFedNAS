#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATASET_NAME=cifar10
export METHOD_NAME=deepfednas
export PARTITION_ALPHA=100
export PARTICIPATION=0.6
exec bash "$ROOT/experiments/_run_training.sh"
