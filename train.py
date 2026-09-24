import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import wandb
from wandb.sdk.lib import RunDisabled
from collections import OrderedDict

# --- Path Setup ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# --- Argument Parsing ---
# ### NEW: Using the updated parser
from parse_args import add_args
from deepfednas.wandb_run_policy import (
    resolve_wandb_run_policy,
)

# --- Data Loaders (No change) ---
from deepfednas.data.cifar10.data_loader import load_partition_data_cifar10
from deepfednas.data.cifar100.data_loader import (
    load_partition_data_cifar100,
)
from deepfednas.data.cinic10.data_loader import load_partition_data_cinic10

# --- Learning Rate and Trainers (No change) ---
from deepfednas.checkpointing import (
    build_resume_config,
    restore_rng_state,
    validate_resume_checkpoint,
    validate_resume_config,
)
from deepfednas.Server.deepfednas_trainer import FLOFA_Trainer
from deepfednas.Client.subnet_trainer import SubnetTrainer

# --- Model Imports ---
### REMOVED: Old, specific server model imports.
# from deepfednas.Server.ServerModel import (
#     ServerResnet,
#     ServerResnet_10_26,
#     ServerMobilenetV3Large_32x32,
#     # ... other specific models
# )

### NEW: Import the new generic server model.
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
            args.val_batch_size,
            args.use_train_pkl,
            args.client_partition_manifest,
            args.client_partition_seed,
            args.augmentation,
            args.randaugment_num_ops,
            args.randaugment_magnitude,
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
            args.val_batch_size,
            args.use_train_pkl,
            args.augmentation,
            args.randaugment_num_ops,
            args.randaugment_magnitude,
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
            args.val_batch_size,
            args.augmentation,
            args.randaugment_num_ops,
            args.randaugment_magnitude,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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
def create_model(args, output_dim, device, load_teacher=False, checkpoint=None):
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
        if checkpoint is None:
            checkpoint = torch.load(
                ckpt_path, map_location=device, weights_only=False
            )

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


def custom_server_trainer(server_trainer_params):
    assert server_trainer_params is not None
    return FLOFA_Trainer(**server_trainer_params)


def custom_client_trainer(client_trainer_params):
    assert client_trainer_params is not None
    return SubnetTrainer(**client_trainer_params)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description="FedAvg-standalone-generic"))
    args = parser.parse_args()

    if args.augmentation == "mixaug" and args.dataset not in (
        "cifar10", "cifar100", "cinic10",
    ):
        parser.error(
            "--augmentation mixaug is implemented only for CIFAR-10, "
            "CIFAR-100, and CINIC-10"
        )
    if args.augmentation == "mixaug" and args.mix_aug_mode == "none":
        parser.error("--augmentation mixaug requires a non-'none' --mix_aug_mode")
    if args.augmentation == "basic" and args.mix_aug_mode != "none":
        parser.error("A non-'none' --mix_aug_mode requires --augmentation mixaug")
    if args.mixup_alpha <= 0 or args.cutmix_alpha <= 0:
        parser.error("--mixup_alpha and --cutmix_alpha must be positive")
    if args.randaugment_num_ops <= 0:
        parser.error("--randaugment_num_ops must be positive")
    if not 0 <= args.randaugment_magnitude <= 30:
        parser.error("--randaugment_magnitude must be between 0 and 30")
    if args.dataset not in ("cifar10", "cifar100", "cinic10"):
        parser.error("DeepFedNAS supports CIFAR-10, CIFAR-100, and CINIC-10")
    if args.cli_supernet or args.cli_supernet_ps or args.multi or args.feddyn:
        parser.error("This release supports the DeepFedNAS subnet training path only")
    if args.lr_schedule:
        parser.error("--lr_schedule is not used by these experiments; use --weighted_avg_schedule")
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

    if args.resume_training and not args.local_model_ckpt_path:
        parser.error("--resume_training requires --local_model_ckpt_path")
    if args.resume_round > 0 and not args.local_model_ckpt_path:
        parser.error("--resume_round requires --local_model_ckpt_path")
    loaded_checkpoint = None
    if args.local_model_ckpt_path:
        checkpoint_path = os.path.abspath(
            os.path.expanduser(args.local_model_ckpt_path)
        )
        if not os.path.isfile(checkpoint_path):
            if args.resume_training or args.resume_round > 0:
                raise FileNotFoundError(
                    f"Requested resume checkpoint does not exist: {checkpoint_path}"
                )
            logging.warning(
                "Checkpoint path specified but not found: %s. Creating a new model.",
                checkpoint_path,
            )
        else:
            loaded_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
    checkpoint_run_id = (
        loaded_checkpoint.get("wandb_run_id")
        if loaded_checkpoint is not None
        else None
    )
    checkpoint_next_round = (
        loaded_checkpoint.get("next_round")
        if loaded_checkpoint is not None
        else None
    )
    try:
        wandb_policy = resolve_wandb_run_policy(
            resume_training=args.resume_training,
            fresh_run_on_resume=args.wandb_fresh_run_on_resume,
            requested_run_id=args.wandb_run_id_resume,
            checkpoint_run_id=checkpoint_run_id,
            run_name=args.wandb_run_name,
            checkpoint_next_round=checkpoint_next_round,
        )
    except ValueError as exc:
        parser.error(str(exc))

    args.wandb_run_id_resume = wandb_policy.run_id
    args.wandb_run_name = wandb_policy.run_name
    # These runtime-only fields make independently logged continuation
    # segments traceable in W&B without affecting the checkpointed training
    # configuration.
    args.wandb_predecessor_run_id = wandb_policy.predecessor_run_id
    args.wandb_segment_start_round = wandb_policy.segment_start_round
    if wandb_policy.source == "checkpoint":
        logging.info(
            "Using W&B run ID %s stored in the resume checkpoint.",
            args.wandb_run_id_resume,
        )
    elif wandb_policy.source == "fresh_resume_segment":
        logging.info(
            "Starting fresh W&B segment %s for the checkpoint continuation; "
            "prior run ID %s will not be modified.",
            args.wandb_run_name,
            args.wandb_predecessor_run_id or "unavailable",
        )

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
        # You can add the other DeepMAD-related args here if needed by the model directly,
        # but they are primarily for the NAS searcher.
    }
    logger.info(f"Assembled Architecture Parameters: {arch_params}")


    ### NEW: Create the model using the new function and passing arch_params
    # The --model argument now acts as a switch for which *type* of generic model to use.
    server_model = create_model(
        args,
        output_dim=args.num_classes,
        device=device,
        checkpoint=loaded_checkpoint,
    )

    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    elif args.wandb_upload_checkpoints and not isinstance(wandb.run, RunDisabled):
        checkpoint_dir = wandb.run.dir
    else:
        checkpoint_dir = os.path.join(
            os.getcwd(), "checkpoints", args.wandb_run_name
        )
    current_resume_config = build_resume_config(args)
    server_model.configure_checkpointing(
        checkpoint_dir,
        upload_checkpoints=args.wandb_upload_checkpoints,
        resume_config=current_resume_config,
    )
    
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

    # Resume logic. Robust mode obtains the next round from the checkpoint;
    # --resume_round remains available only for legacy checkpoint compatibility.
    actual_start_round = 0
    resume_state = None
    if args.resume_training:
        if os.path.basename(checkpoint_path) != "latest_round_model.pt":
            raise ValueError(
                "Robust resume must use latest_round_model.pt, not a best or "
                "manually selected checkpoint."
            )
        checkpoint_parent = os.path.realpath(os.path.dirname(checkpoint_path))
        configured_checkpoint_dir = os.path.realpath(checkpoint_dir)
        if checkpoint_parent != configured_checkpoint_dir:
            raise ValueError(
                "For robust resume, --checkpoint_dir must be the directory "
                "containing latest_round_model.pt so the existing best checkpoint "
                "is preserved."
            )
        best_checkpoint_path = os.path.join(
            configured_checkpoint_dir, "best_checkpoint_supernet.pt"
        )
        if not os.path.isfile(best_checkpoint_path):
            raise FileNotFoundError(
                "Robust resume requires the existing best checkpoint beside the "
                f"latest checkpoint: {best_checkpoint_path}"
            )

        resume_state = validate_resume_checkpoint(loaded_checkpoint)
        validate_resume_config(
            loaded_checkpoint["resume_config"], current_resume_config
        )
        actual_start_round = resume_state["next_round"]
        if args.resume_round not in (0, actual_start_round):
            raise ValueError(
                "--resume_round disagrees with checkpoint metadata: "
                f"argument={args.resume_round}, checkpoint={actual_start_round}"
            )
        logging.info(
            "Faithfully resuming after completed round %s; next round is %s and "
            "best validation metric is %s",
            resume_state["completed_round"],
            actual_start_round,
            resume_state["best_metric"],
        )
    elif args.resume_round > 0:
        actual_start_round = args.resume_round
        logging.warning(
            "Using legacy manual resume from round %s. This mode cannot verify "
            "that all training state is complete; prefer --resume_training with "
            "a versioned latest checkpoint.",
            actual_start_round,
        )
    else:
        logging.info("Starting training from scratch (round 0).")
    if actual_start_round >= args.comm_round:
        raise ValueError(
            f"Checkpoint next round {actual_start_round} has already reached "
            f"--comm_round {args.comm_round}."
        )
    server_trainer_params["start_round"] = actual_start_round
    if resume_state is not None:
        server_trainer_params["resume_state"] = resume_state

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
    
    # The experiment schedule controls aggregation weights in the server.
    server_trainer_params["lr_scheduler"] = None

    server_trainer_params["wt_avg_sched_method"] = "Uniform"
    if args.weighted_avg_schedule and args.weighted_avg_schedule.get("type"):
        server_trainer_params["wt_avg_sched_method"] = args.weighted_avg_schedule["type"]

    # Instantiate and start the training
    server_trainer = custom_server_trainer(server_trainer_params)

    # Reconstruction above consumes random numbers. Restore the captured state
    # only now, immediately before entering the next round.
    if resume_state is not None:
        restore_rng_state(
            resume_state["rng_state"],
            require_cuda_state=torch.cuda.is_available(),
        )
        logging.info("Restored Python, NumPy, PyTorch CPU, and CUDA RNG states.")
    elif args.resume_round > 0 and loaded_checkpoint is not None:
        if "rng_state" in loaded_checkpoint:
            restore_rng_state(loaded_checkpoint["rng_state"])
        else:
            if "torch_rng_state" in loaded_checkpoint:
                torch.set_rng_state(loaded_checkpoint["torch_rng_state"].cpu())
            if "numpy_rng_state" in loaded_checkpoint:
                np.random.set_state(loaded_checkpoint["numpy_rng_state"])
            logging.warning(
                "Legacy checkpoint lacks Python/CUDA RNG state; continuation "
                "may not be bitwise faithful."
            )
    server_trainer.train()
