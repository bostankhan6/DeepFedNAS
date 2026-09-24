#!/usr/bin/env bash
set -euo pipefail

# Dataset-neutral launcher for the validation-only baseline predictor phase.
# The official test split is not loaded by any of the selected collectors.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${DATASET:?Set DATASET to cifar10, cifar100, or cinic10}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the completed baseline checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a new predictor artifact directory}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/$DATASET}"
INITIAL_DATASET="${INITIAL_DATASET:-}"
NUM_ARCHITECTURES="${NUM_ARCHITECTURES:-10000}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-restricted-bin-stratified}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

case "$DATASET" in
    cifar10)
        COLLECTOR="scripts/search/cifar10_validation_predictor.py"
        EXPECTED_VALIDATION="$DATA_DIR/val.pkl"
        ;;
    cifar100)
        COLLECTOR="scripts/search/cifar100_validation_predictor.py"
        EXPECTED_VALIDATION="$DATA_DIR/val.pkl"
        ;;
    cinic10)
        COLLECTOR="scripts/search/cinic10_validation_predictor.py"
        EXPECTED_VALIDATION="$DATA_DIR/val"
        ;;
    *)
        echo "Error: DATASET must be cifar10, cifar100, or cinic10."
        exit 1
        ;;
esac

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: baseline checkpoint does not exist: $CHECKPOINT"
    exit 1
fi
if [ ! -e "$EXPECTED_VALIDATION" ]; then
    echo "Error: validation data does not exist: $EXPECTED_VALIDATION"
    exit 1
fi
if [ -n "$INITIAL_DATASET" ] && [ ! -f "$INITIAL_DATASET" ]; then
    echo "Error: initial predictor dataset does not exist: $INITIAL_DATASET"
    exit 1
fi
if ! [[ "$NUM_ARCHITECTURES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_ARCHITECTURES must be a positive integer."
    exit 1
fi
if [ "$SAMPLING_STRATEGY" = "restricted-bin-stratified" ] \
    && [ $((NUM_ARCHITECTURES % 4)) -ne 0 ]; then
    echo "Error: strict-bin NUM_ARCHITECTURES must be divisible by four."
    exit 1
fi
if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "Error: DRY_RUN must be 0 or 1."
    exit 1
fi

COMMAND=(
    "$PYTHON_BIN" "$COLLECTOR"
    --checkpoint "$CHECKPOINT"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --num-architectures "$NUM_ARCHITECTURES"
    --sampling-strategy "$SAMPLING_STRATEGY"
    --gpu "$GPU_ID"
)
if [ -n "$INITIAL_DATASET" ]; then
    COMMAND+=(--initial-dataset "$INITIAL_DATASET")
fi
if [ "$DATASET" = "cinic10" ]; then
    COMMAND+=(
        --validation-subset-size "${VALIDATION_SUBSET_SIZE:-10000}"
        --subset-seed "${SUBSET_SEED:-20260805}"
    )
fi

cd "$REPO_ROOT"
printf 'Predictor command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete; no architectures were evaluated."
    exit 0
fi

"${COMMAND[@]}"
