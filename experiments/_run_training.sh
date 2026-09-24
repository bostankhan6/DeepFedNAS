#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME}"
METHOD_NAME="${METHOD_NAME:?Set METHOD_NAME}"
PARTITION_ALPHA="${PARTITION_ALPHA:-100}"
PARTICIPATION="${PARTICIPATION:-0.4}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
RESUME_TRAINING="${RESUME_TRAINING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "$SEED" != 0 ]]; then
    echo "These experiment settings use one training seed: 0" >&2
    exit 1
fi

case "$DATASET_NAME" in
    cifar10)
        TOTAL_CLIENTS=20
        ROUNDS=1500
        SCHEDULE='{"type":"maxnet_cos_all_subnet","num_steps":1200,"init":0.9,"final":0.125}'
        SPLIT_ARGS=(--use_train_pkl --client_partition_seed 0)
        alpha_label="${PARTITION_ALPHA//./p}"
        PARTITION_MANIFEST="$ROOT/configs/partitions/cifar10/alpha${alpha_label}_seed0.json"
        SPLIT_ARGS+=(--client_partition_manifest "$PARTITION_MANIFEST")
        ;;
    cifar100)
        TOTAL_CLIENTS=20
        ROUNDS=2000
        SCHEDULE='{"type":"maxnet_cos_all_subnet","num_steps":1600,"init":0.9,"final":0.125}'
        SPLIT_ARGS=(--use_train_pkl --client_partition_seed 0)
        ;;
    cinic10)
        TOTAL_CLIENTS=100
        ROUNDS=1500
        SCHEDULE='{"type":"maxnet_cos_all_subnet","num_steps":1200,"init":0.9,"final":0.025}'
        SPLIT_ARGS=()
        ;;
    *) echo "Unsupported dataset: $DATASET_NAME" >&2; exit 1 ;;
esac

case "$PARTICIPATION" in
    0.1) CLIENTS_PER_ROUND=$((TOTAL_CLIENTS / 10)) ;;
    0.2) CLIENTS_PER_ROUND=$((TOTAL_CLIENTS / 5)) ;;
    0.4) CLIENTS_PER_ROUND=$((TOTAL_CLIENTS * 2 / 5)) ;;
    0.6) CLIENTS_PER_ROUND=$((TOTAL_CLIENTS * 3 / 5)) ;;
    *) echo "Unsupported participation: $PARTICIPATION" >&2; exit 1 ;;
esac

CACHE="$ROOT/subnet_caches/4_stage_cache_60_subnets.csv"
case "$METHOD_NAME" in
    deepfednas) SAMPLER=TS_optimal_path ;;
    range_matched_random) SAMPLER=TS_range_matched_random ;;
    *) echo "Unsupported method: $METHOD_NAME" >&2; exit 1 ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/${DATASET_NAME}_alpha${PARTITION_ALPHA}_c${PARTICIPATION}/${METHOD_NAME}/seed0}"
RESUME_ARGS=()
if [[ "$RESUME_TRAINING" == 1 ]]; then
    for name in latest_round_model.pt best_checkpoint_supernet.pt; do
        [[ -f "$CHECKPOINT_DIR/$name" ]] || { echo "Missing $CHECKPOINT_DIR/$name" >&2; exit 1; }
    done
    RESUME_ARGS=(--resume_training --local_model_ckpt_path "$CHECKPOINT_DIR/latest_round_model.pt" --wandb_fresh_run_on_resume)
elif [[ "$RESUME_TRAINING" != 0 ]]; then
    echo "RESUME_TRAINING must be 0 or 1" >&2; exit 1
elif [[ -e "$CHECKPOINT_DIR/latest_round_model.pt" || -e "$CHECKPOINT_DIR/best_checkpoint_supernet.pt" ]]; then
    echo "Checkpoint exists; set RESUME_TRAINING=1 to continue" >&2; exit 1
fi

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -f "$CACHE" ]] || { echo "Missing cache: $CACHE" >&2; exit 1; }
    if [[ "$DATASET_NAME" == cinic10 ]]; then
        [[ -d "$ROOT/data/cinic10/train" && -d "$ROOT/data/cinic10/val" ]] || { echo "Prepare CINIC-10 train/val first" >&2; exit 1; }
    else
        [[ -f "$ROOT/data/$DATASET_NAME/train.pkl" && -f "$ROOT/data/$DATASET_NAME/val.pkl" ]] || { echo "Run scripts/data_setup/download_$DATASET_NAME.sh first" >&2; exit 1; }
        if [[ "$DATASET_NAME" == cifar10 ]]; then
            [[ -f "$PARTITION_MANIFEST" ]] || { echo "Missing partition: $PARTITION_MANIFEST" >&2; exit 1; }
        fi
    fi
fi

DIVERSE_SUBNETS='{"0": {"d": [2, 2, 2, 2], "e": [0.18, 0.1, 0.1, 0.18, 0.14, 0.1, 0.14, 0.14, 0.1, 0.1, 0.1, 0.1], "w_indices": [9, 4, 4, 4, 4]}, "1": {"d": [2, 2, 2, 2], "e": [0.14, 0.14, 0.14, 0.18, 0.14, 0.14, 0.14, 0.14, 0.1, 0.14, 0.14, 0.14], "w_indices": [9, 8, 7, 8, 7]}, "2": {"d": [2, 2, 2, 2], "e": [0.25, 0.14, 0.14, 0.25, 0.14, 0.14, 0.18, 0.18, 0.18, 0.18, 0.18, 0.14], "w_indices": [9, 9, 9, 9, 9]}, "3": {"d": [2, 2, 2, 2], "e": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], "w_indices": [9, 9, 9, 9, 9]}}'

CMD=("$PYTHON_BIN" "$ROOT/train.py"
    --model ofaresnet_generic
    --wandb_project_name DeepFedNAS
    --wandb_group "${DATASET_NAME}_alpha${PARTITION_ALPHA}_c${PARTICIPATION}"
    --wandb_run_name "${METHOD_NAME}_seed0"
    --gpu "$GPU_ID"
    --dataset "$DATASET_NAME"
    --data_dir "$ROOT/data/$DATASET_NAME"
    --partition_method hetero
    --partition_alpha "$PARTITION_ALPHA"
    --client_num_in_total "$TOTAL_CLIENTS"
    --client_num_per_round "$CLIENTS_PER_ROUND"
    --comm_round "$ROUNDS"
    --epochs 5 --batch_size 64 --val_batch_size 512
    --client_optimizer sgd --lr 0.1 --init_seed 0
    --augmentation mixaug --randaugment_num_ops 2 --randaugment_magnitude 6
    --mix_aug_mode alternating --mixup_alpha 0.4 --cutmix_alpha 1.0
    --max_norm 10.0 --verbose --frequency_of_the_test 20
    --best_model_freq 1000 --model_checkpoint_freq 500 --efficient_test
    --no_wandb_upload_checkpoints --checkpoint_dir "$CHECKPOINT_DIR"
    --weighted_avg_schedule "$SCHEDULE"
    --subnet_dist_type "$SAMPLER" --subnet_cache_path "$CACHE"
    --supernet_num_stages 4 --supernet_max_extra_blocks_per_stage 2
    --supernet_original_stage_base_channels '[256, 512, 1024, 2048]'
    --supernet_width_multiplier_choices '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]'
    --supernet_expansion_ratio_choices '[0.1, 0.14, 0.18, 0.22, 0.25]'
    --diverse_subnets "$DIVERSE_SUBNETS"
    "${SPLIT_ARGS[@]}" "${RESUME_ARGS[@]}")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    exec "${CMD[@]}"
fi
