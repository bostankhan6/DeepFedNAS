#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"
echo "Running experiment from root: $PROJECT_ROOT"

python train.py \
    --model ofaresnet_generic \
    --wandb_project_name new_deepfednas_repo \
    --wandb_group "non-iid-1.0_4_Stage_original_Supernet" \
    --wandb_run_name "Baseline_TS_all_random_seed0" \
    --gpu 0 \
    --dataset cifar10 \
    --data_dir "$PROJECT_ROOT/data/cifar10" \
    --partition_method hetero \
    --partition_alpha 1.0 \
    --client_num_in_total 20 \
    --client_num_per_round 8 \
    --comm_round 1500 \
    --epochs 5 \
    --batch_size 64 \
    --client_optimizer sgd \
    --lr 0.1 \
    --init_seed 0 \
    --max_norm 10.0 \
    --verbose \
    --frequency_of_the_test 20 \
    --efficient_test \
    --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":1200,"init":0.9,"final":0.125}' \
    --subnet_dist_type TS_all_random \
    --supernet_num_stages 4 \
    --supernet_max_extra_blocks_per_stage 2 \
    --supernet_original_stage_base_channels '[256, 512, 1024, 2048]' \
    --supernet_width_multiplier_choices '[1.0]' \
    --supernet_expansion_ratio_choices '[0.1, 0.14, 0.18, 0.22, 0.25]' \
    --diverse_subnets '{"0":{"d":[0,0,0,0],"e":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],"w_indices":[0,0,0,0,0]},"1":{"d":[0,1,0,1],"e":[0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14],"w_indices":[0,0,0,0,0]},"2":{"d":[1,1,1,1],"e":[0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18],"w_indices":[0,0,0,0,0]}, "3":{"d":[2,2,2,2],"e":[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],"w_indices":[0,0,0,0,0]}}'