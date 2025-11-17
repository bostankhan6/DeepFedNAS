import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import wandb
from collections import OrderedDict

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

# --- Argument Parsing ---
# ### NEW: Using the updated parser
from parse_args import add_args

# --- Data Loaders (No change) ---
from deepfednas.data.cifar10.data_loader import load_partition_data_cifar10
from deepfednas.data.cifar100.data_loader import load_partition_data_cifar100

from deepfednas.data.cinic10.data_loader import load_partition_data_cinic10

# --- Learning Rate and Trainers (No change) ---
from deepfednas.Server import (
    deepfednas_trainer
)
from deepfednas.Client.subnet_trainer import SubnetTrainer

### Import the new generic server model.
from deepfednas.Server.generic_server_model import GenericServerOFA

# The load_data and combine_batches functions remain unchanged as they are not specific to the model architecture.
def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        raise ValueError("Not supported")

    elif dataset_name == "femnist":
        raise ValueError("Not supported")
    #renamed from fed_shakespeare to tf_shakespeare since this is the Tensorflow sourced version of shakespeare dataset
    elif dataset_name == "tf_shakespeare":
        raise ValueError("Not supported")

    elif dataset_name == "fed_cifar100":
        raise ValueError("Not supported")
    elif dataset_name == "stackoverflow_lr":
        raise ValueError("Not supported")
    elif dataset_name == "stackoverflow_nwp":
        raise ValueError("Not supported")

    elif dataset_name == "ILSVRC2012":
        raise ValueError("Not supported")

    elif dataset_name == "gld23k":
        raise ValueError("Not supported")

    elif dataset_name == "gld160k":
        raise ValueError("Not supported")
    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
                args.val_batch_size
            )
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
                args.val_batch_size
            )
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
                args.val_batch_size
            )
         #Penn Tree Bank
        else:
            data_loader = load_partition_data_cifar10
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = data_loader(
                args.dataset,
                args.data_dir,
                args.partition_method,
                args.partition_alpha,
                args.client_num_in_total,
                args.batch_size,
            )

    if centralized:
        train_data_local_num_dict = {
            0: sum(
                user_train_data_num
                for user_train_data_num in train_data_local_num_dict.values()
            )
        }
        train_data_local_dict = {
            0: [
                batch
                for cid in sorted(train_data_local_dict.keys())
                for batch in train_data_local_dict[cid]
            ]
        }
        test_data_local_dict = {
            0: [
                batch
                for cid in sorted(test_data_local_dict.keys())
                for batch in test_data_local_dict[cid]
            ]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid])
            for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {
            cid: combine_batches(test_data_local_dict[cid])
            for cid in test_data_local_dict.keys()
        }
        args.batch_size = args_batch_size

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


### REMOVED: Old `custom_ofa_net` function. It is replaced by the logic in `create_model`.

### NEW: `create_model` function is now the primary model factory.
def create_model(args, output_dim, device, load_teacher=False):
    """
    Handles creation of a new model or loading a model from a checkpoint.
    This is now the central factory for the generic framework.
    """
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (args.model, output_dim)
    )

    # Determine which checkpoint path to use (for main model or teacher)
    ckpt_path = args.local_model_ckpt_path if not load_teacher else None # Add specific teacher path arg if needed

    # --- Case 1: Loading from a Checkpoint ---
    if ckpt_path and os.path.exists(ckpt_path):
        logging.info(f"Loading model from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Restore RNG states if resuming the main model
        if args.resume_round > 0 and not load_teacher:
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
                logging.info("Restored PyTorch RNG state from checkpoint.")
            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])
                logging.info("Restored NumPy RNG state from checkpoint.")

        # Load the architecture parameters from the checkpoint
        if "arch_params" not in checkpoint:
            raise ValueError(f"Checkpoint '{ckpt_path}' is missing the required 'arch_params' dictionary.")
        
        loaded_arch_params = checkpoint["arch_params"]
        logging.info(f"Loaded architecture parameters from checkpoint: {loaded_arch_params}")

        # Instantiate the model using the loaded architecture parameters
        model = GenericServerOFA(
            arch_params=loaded_arch_params,
            sampling_method=args.subnet_dist_type,
            num_cli_total=args.client_num_in_total,
            bn_gamma_zero_init=args.bn_gamma_zero_init,
            cli_subnet_track=checkpoint.get("cli_subnet_track") # Load tracker from checkpoint
        )

        # Load the model weights
        if "params" in checkpoint:
            model.set_model_params(checkpoint["params"])
            logging.info("Successfully loaded model weights from checkpoint.")
        else:
            raise ValueError(f"Checkpoint '{ckpt_path}' is missing the model weights ('params' key).")

    # --- Case 2: Creating a New Model from Scratch ---
    else:
        if ckpt_path:
            logging.warning(f"Checkpoint path specified but not found: {ckpt_path}. Creating a new model.")
        else:
            logging.info("No checkpoint path specified. Creating a new model from command-line arguments.")

        # Assemble architecture parameters from command-line arguments
        arch_params = {
            'num_stages': args.supernet_num_stages,
            'initial_input_hw': args.supernet_initial_input_hw,
            'initial_input_channels': args.supernet_initial_input_channels,
            'stem_stride': args.supernet_stem_stride,
            'original_stem_out_channels': args.supernet_original_stem_out_channels,
            'original_stage_base_channels': args.supernet_original_stage_base_channels,
            'stage_downsample_factors': args.supernet_stage_downsample_factors,
            'max_extra_blocks_per_stage': args.supernet_max_extra_blocks_per_stage,
            'channel_divisible_by': args.supernet_channel_divisible_by,
            'width_multiplier_choices': args.supernet_width_multiplier_choices,
            'expansion_ratio_choices': args.supernet_expansion_ratio_choices,
            'n_classes': output_dim,
            'bn_gamma_zero_init': args.bn_gamma_zero_init,
        }
        
        if args.model == 'ofaresnet_generic':
            model = GenericServerOFA(
                arch_params=arch_params,
                sampling_method=args.subnet_dist_type,
                num_cli_total=args.client_num_in_total,
                bn_gamma_zero_init=args.bn_gamma_zero_init,
                cli_subnet_track=args.cli_subnet_track # Use tracker from args if provided
            )
        else:
            raise ValueError(f"Model type '{args.model}' is not supported for new model creation.")

    return model


# The custom_server_trainer and custom_client_trainer functions remain unchanged.
def custom_server_trainer(server_trainer_params):
    assert server_trainer_params is not None
    return deepfednas_trainer(**server_trainer_params)


def custom_client_trainer(client_trainer_params):
    assert client_trainer_params is not None
    print(f"Using Subnet Trainer")
    return SubnetTrainer(**client_trainer_params)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description="FedAvg-standalone-generic"))
    args = parser.parse_args()
    logger.info(args)

    # KD logic remains the same
    if args.kd_ratio > 0 and not args.multi:
        assert (
            args.teacher_ckpt_name is not None and args.teacher_run_path is not None
        ), "Specify Pretrained model for knowledge distillation"

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # WandB init logic remains the same
    wandb_init_params = {
        "project": args.wandb_project_name,
        "name": args.wandb_run_name,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "config": args,
    }
    if args.wandb_run_id_resume:
        wandb_init_params["resume"] = "allow"
        wandb_init_params["id"] = args.wandb_run_id_resume
        logging.info(f"Attempting to resume W&B run with ID: {args.wandb_run_id_resume}")
    else:
        logging.info("Starting a new W&B run.")
    wandb.init(**wandb_init_params)

    # Seeding remains the same
    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed_all(args.init_seed)

    # Load data
    args.device = device
    dataset = load_data(args, args.dataset)
    
    # Set number of classes from the dataset
    args.num_classes = dataset[7]

    ### NEW: Collect all supernet architecture parameters into a dictionary
    arch_params = {
        'num_stages': args.supernet_num_stages,
        'initial_input_hw': args.supernet_initial_input_hw,
        'initial_input_channels': args.supernet_initial_input_channels,
        'stem_stride': args.supernet_stem_stride,
        'original_stem_out_channels': args.supernet_original_stem_out_channels,
        'original_stage_base_channels': args.supernet_original_stage_base_channels,
        'stage_downsample_factors': args.supernet_stage_downsample_factors,
        'max_extra_blocks_per_stage': args.supernet_max_extra_blocks_per_stage,
        'channel_divisible_by': args.supernet_channel_divisible_by,
        'width_multiplier_choices': args.supernet_width_multiplier_choices,
        'expansion_ratio_choices': args.supernet_expansion_ratio_choices,
        'n_classes': args.num_classes,
        'bn_gamma_zero_init': args.bn_gamma_zero_init,
        # You can add the other DeepFedNAS-related args here if needed by the model directly,
        # but they are primarily for the NAS searcher.
    }
    logger.info(f"Assembled Architecture Parameters: {arch_params}")


    ### NEW: Create the model using the new function and passing arch_params
    # The --model argument now acts as a switch for which *type* of generic model to use.
    server_model = create_model(args, output_dim=args.num_classes, device=device)
    
    # The rest of the pipeline remains largely the same, as it interacts with the
    # server_model object through the expected API.

    if args.wandb_watch:
        logging.warning("Watching model parameters")
        wandb.watch(
            server_model.model, log="parameters", log_freq=args.wandb_watch_freq,
        )

    # Client trainer setup
    client_trainer_params = {
        "model": None, # Client model is set per-round
        "device": device,
        "args": args
    }
    
    # Server trainer setup
    server_trainer_params = {
        "server_model": server_model,
        "dataset": dataset,
        "args": args
    }

    # Resume logic
    actual_start_round = 0
    if args.resume_round > 0:
        actual_start_round = args.resume_round
        logging.info(f"Resuming training from round {actual_start_round}")
    else:
        logging.info("Starting training from scratch (round 0).")
    server_trainer_params["start_round"] = actual_start_round

    # Teacher model setup
    teacher_model = None
    if args.kd_ratio > 0:
        # Assuming teacher model uses the same architecture
        teacher_model = create_model(
            args, arch_params, output_dim=args.num_classes, device=device, load_teacher=True
        )
        server_trainer_params["teacher_model"] = teacher_model
        client_trainer_params["teacher_model"] = teacher_model

    server_trainer_params["client_trainer"] = custom_client_trainer(client_trainer_params)
    
    # LR scheduler and weighted average setup (no change)
    flofa_lr_scheduler = None
    if args.lr_schedule and args.lr_schedule.get("type"):
        # ... (Your existing LR scheduler logic remains valid)
        pass # Placeholder for your existing logic
    server_trainer_params["lr_scheduler"] = flofa_lr_scheduler

    server_trainer_params["wt_avg_sched_method"] = "Uniform"
    if args.weighted_avg_schedule and args.weighted_avg_schedule.get("type"):
        server_trainer_params["wt_avg_sched_method"] = args.weighted_avg_schedule["type"]

    # Instantiate and start the training
    server_trainer = custom_server_trainer(server_trainer_params)
    server_trainer.train()