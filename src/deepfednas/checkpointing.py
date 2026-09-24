"""Checkpoint helpers for faithful and crash-safe training resumes."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 2

# Runtime and output-location arguments are intentionally excluded.  These are
# the training/data/model settings that must remain identical after a restart.
RESUME_CONFIG_FIELDS = (
    "model",
    "dataset",
    "partition_method",
    "partition_alpha",
    "client_num_in_total",
    "client_num_per_round",
    "client_partition_seed",
    "use_train_pkl",
    "comm_round",
    "epochs",
    "batch_size",
    "val_batch_size",
    "client_optimizer",
    "lr",
    "wd",
    "augmentation",
    "randaugment_num_ops",
    "randaugment_magnitude",
    "mix_aug_mode",
    "mixup_alpha",
    "cutmix_alpha",
    "max_norm",
    "init_seed",
    "frequency_of_the_test",
    "efficient_test",
    "weighted_avg_schedule",
    "subnet_dist_type",
    "top_k_maxnet",
    "bottom_k_maxnet",
    "num_multi_archs",
    "use_bn",
    "bn_gamma_zero_init",
    "inplace_kd",
    "optim_step_more",
    "largest_step_more",
    "largest_subnet_wd",
    "weight_dataset",
    "kd_ratio",
    "multi",
    "supernet_num_stages",
    "supernet_initial_input_hw",
    "supernet_initial_input_channels",
    "supernet_stem_stride",
    "supernet_original_stem_out_channels",
    "supernet_original_stage_base_channels",
    "supernet_stage_downsample_factors",
    "supernet_max_extra_blocks_per_stage",
    "supernet_channel_divisible_by",
    "supernet_width_multiplier_choices",
    "supernet_expansion_ratio_choices",
    "supernet_alpha_weights",
    "beta_depth_penalty",
    "supernet_rho0_constraint",
    "supernet_effectiveness_fitness_weight",
    "supernet_non_decreasing_channels_constraint",
    "non_decreasing_penalty_coeff",
    "ga_pop_size",
    "ga_generations",
    "ga_mutate_p",
    "subnet_cache_path",
    "ofa_config",
    "ofa_config_mbv3",
    "ofa_config_resnet",
    "ofa_config_resnet_10_26",
    "ofa_config_resnet_10_26_2",
    "ofa_config_resnet_10_26_3",
    "ofa_config_resnet_10_26_4",
    "ofa_config_resnet_10_26_5",
    "ofa_config_resnet_10_26_6",
    "original_model",
    "original_resnet_10_26",
    "original_resnet_10_26_2",
    "original_resnet_10_26_3",
    "original_resnet_10_26_4",
    "original_resnet_10_26_5",
    "original_resnet_10_26_6",
    "original_resnet_10_26_7",
    "original_resnet_10_26_8",
    "original_resnet_10_26_9",
    "original_resnet_10_26_10",
    "original_resnet_10_26_11",
    "original_resnet_10_26_12",
    "original_resnet_10_26_13",
    "original_resnet_10_26_14",
    "original_resnet_10_26_15",
    "original_resnet_10_26_16",
    "original_resnet_10_26_17",
    "original_resnet_10_26_18",
    "original_resnet_10_26_19",
    "original_resnet_10_26_20",
    "feddyn",
    "feddyn_alpha",
    "feddyn_max_norm",
    "feddyn_no_wd_modifier",
    "feddyn_override_wd",
    "mod_wd_dyn",
    "skip_train_largest",
    "multi_drop_largest",
    "multi_disable_rest_bn",
    "reset_bn_stats",
    "reset_bn_stats_test",
    "reset_bn_sample_size",
    "warmup_init_lr",
    "warmup_rounds",
    "kd_type",
    "skip_train_test",
    "clean_subnet",
    "diverse_subnets",
    "ckpt_subnets",
)

# Format-v2 checkpoints written before MixAug was introduced do not contain
# these fields. Their behavior is exactly the current basic/default pipeline,
# so supply only those historical defaults during configuration comparison.
LEGACY_RESUME_CONFIG_DEFAULTS = {
    "augmentation": "basic",
    "randaugment_num_ops": 2,
    "randaugment_magnitude": 6,
    "mix_aug_mode": "none",
    "mixup_alpha": 0.4,
    "cutmix_alpha": 1.0,
}


def _json_value(value):
    """Return a stable JSON-compatible representation of an argparse value."""
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_resume_config(args):
    """Build the canonical training configuration stored in each checkpoint."""
    config = {
        field: _json_value(getattr(args, field, None))
        for field in RESUME_CONFIG_FIELDS
    }
    # ckpt_subnets=None means "use diverse_subnets" in train.py. Store the
    # effective value so equivalent invocations compare equal.
    if config["ckpt_subnets"] is None:
        config["ckpt_subnets"] = config["diverse_subnets"]
    manifest_path = getattr(args, "client_partition_manifest", None)
    if manifest_path:
        manifest_path = Path(manifest_path).expanduser()
        if not manifest_path.is_file():
            raise ValueError(
                f"Client partition manifest does not exist: {manifest_path}"
            )
        config["client_partition_manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    else:
        config["client_partition_manifest_sha256"] = None
    return config


def resume_config_digest(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_resume_config(saved_config, current_config):
    """Raise when a continuation changes a training-critical setting."""
    mismatches = []
    all_keys = sorted(set(saved_config) | set(current_config))
    for key in all_keys:
        saved_value = saved_config.get(
            key,
            LEGACY_RESUME_CONFIG_DEFAULTS.get(key),
        )
        current_value = current_config.get(key)
        if saved_value != current_value:
            mismatches.append(
                f"{key}: checkpoint={saved_value!r}, current={current_value!r}"
            )
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            "Resume configuration does not match the checkpoint:\n  - " + details
        )


def capture_rng_state():
    """Capture every RNG used by the current training implementation."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state, require_cuda_state=False):
    """Restore RNGs after model, data, and trainer reconstruction is complete."""
    required = ("python", "numpy", "torch_cpu")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Checkpoint is missing RNG state: {', '.join(missing)}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state.get("torch_cuda_all")
    if cuda_states is None:
        if require_cuda_state and torch.cuda.is_available():
            raise ValueError("Checkpoint is missing CUDA RNG state")
        return
    if not torch.cuda.is_available():
        if require_cuda_state:
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        return

    device_count = torch.cuda.device_count()
    if len(cuda_states) != device_count:
        raise ValueError(
            "CUDA device-count mismatch during resume: "
            f"checkpoint={len(cuda_states)}, current={device_count}"
        )
    torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def validate_resume_checkpoint(checkpoint):
    """Validate and return the state required for an automatic continuation."""
    version = checkpoint.get("checkpoint_format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "This checkpoint does not support robust automatic resume "
            f"(format={version!r}, required={CHECKPOINT_FORMAT_VERSION})."
        )

    required = ("completed_round", "next_round", "best_metric", "resume_config", "rng_state")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Resume checkpoint is missing: {', '.join(missing)}")

    completed_round = int(checkpoint["completed_round"])
    next_round = int(checkpoint["next_round"])
    if next_round != completed_round + 1:
        raise ValueError(
            "Invalid round metadata: "
            f"completed_round={completed_round}, next_round={next_round}"
        )

    expected_digest = checkpoint.get("resume_config_digest")
    actual_digest = resume_config_digest(checkpoint["resume_config"])
    if expected_digest != actual_digest:
        raise ValueError("Resume configuration digest is missing or invalid")

    # Validate the mandatory CPU-side RNG entries now. CUDA compatibility is
    # checked at restoration time after the target device is initialized.
    rng_state = checkpoint["rng_state"]
    missing_rng = [key for key in ("python", "numpy", "torch_cpu") if key not in rng_state]
    if missing_rng:
        raise ValueError(f"Resume checkpoint is missing RNG state: {', '.join(missing_rng)}")

    return {
        "completed_round": completed_round,
        "next_round": next_round,
        "best_metric": checkpoint["best_metric"],
        "server_state": checkpoint.get("server_state"),
        "rng_state": rng_state,
    }


def atomic_torch_save(payload, destination):
    """Durably replace a checkpoint without exposing a partial final file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(str(temporary_path), str(destination))
        temporary_path = None

        # Ensure the rename itself reaches stable storage where the filesystem
        # supports directory fsync (Linux local and most cluster filesystems).
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(str(destination.parent), directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    return destination
