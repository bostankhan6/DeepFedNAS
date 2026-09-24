import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from deepfednas.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    build_resume_config,
    capture_rng_state,
    restore_rng_state,
    resume_config_digest,
    validate_resume_checkpoint,
    validate_resume_config,
)
from deepfednas.Server.base_server_model import (
    BaseServerModel,
)


class _CheckpointTestServerModel(BaseServerModel):
    def init_model(self, _init_params):
        return torch.nn.Linear(2, 2)

    def is_max_net(self, _arch):
        return False

    def is_min_net(self, _arch):
        return False

    def get_subnet(self, **_kwargs):
        return None

    def add_subnet(self, _shared_param_sum, _shared_param_count, _w_local):
        return None

    def active_subnet_index(self):
        return None

    def max_subnet_arch(self):
        return {}

    def min_subnet_arch(self):
        return {}

    def random_subnet_arch(self):
        return {}

    def random_depth_subnet_arch(self):
        return {}

    def random_compound_subnet_arch(self):
        return {}

    def mutate_sample(self, sample_arch, _mut_prob):
        return sample_arch


class CheckpointingTest(unittest.TestCase):
    def test_server_save_produces_a_robust_resume_payload(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = _CheckpointTestServerModel(
                init_params=None,
                sampling_method="all_random",
                num_cli_total=2,
            )
            resume_config = {"dataset": "cinic10", "comm_round": 1500}
            model.configure_checkpointing(
                temporary_directory,
                upload_checkpoints=False,
                resume_config=resume_config,
            )
            destination = model.save(
                "latest_round_model.pt",
                {
                    "completed_round": 749,
                    "next_round": 750,
                    "best_metric": 0.71,
                },
            )

            checkpoint = torch.load(destination, weights_only=False)
            state = validate_resume_checkpoint(checkpoint)
            self.assertEqual(state["next_round"], 750)
            self.assertIn("cli_subnet_track", checkpoint)
            self.assertIn("torch_rng_state", checkpoint)
            self.assertIn("numpy_rng_state", checkpoint)

    def test_atomic_save_replaces_checkpoint_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            destination = directory / "latest_round_model.pt"

            atomic_torch_save({"round": 1}, destination)
            atomic_torch_save({"round": 2}, destination)

            self.assertEqual(
                torch.load(destination, weights_only=False), {"round": 2}
            )
            self.assertEqual(
                list(directory.glob(".latest_round_model.pt.*.tmp")), []
            )

    def test_failed_atomic_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            destination = directory / "latest_round_model.pt"
            atomic_torch_save({"round": 7}, destination)

            def partial_write_then_fail(_payload, file_object):
                file_object.write(b"partial checkpoint")
                raise RuntimeError("simulated interruption")

            with mock.patch(
                "deepfednas.checkpointing.torch.save",
                side_effect=partial_write_then_fail,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    atomic_torch_save({"round": 8}, destination)

            self.assertEqual(
                torch.load(destination, weights_only=False), {"round": 7}
            )
            self.assertEqual(
                list(directory.glob(".latest_round_model.pt.*.tmp")), []
            )

    def test_rng_restore_reproduces_python_numpy_and_torch_sequences(self):
        random.seed(41)
        np.random.seed(42)
        torch.manual_seed(43)
        saved_state = capture_rng_state()

        expected_python = [random.random() for _ in range(4)]
        expected_numpy = np.random.random(4)
        expected_torch = torch.rand(4)

        # Simulate reconstruction consuming every global RNG before restoration.
        [random.random() for _ in range(11)]
        np.random.random(11)
        torch.rand(11)

        restore_rng_state(saved_state)

        self.assertEqual([random.random() for _ in range(4)], expected_python)
        np.testing.assert_array_equal(np.random.random(4), expected_numpy)
        torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)

    def test_rng_restore_applies_all_visible_cuda_states(self):
        saved_state = capture_rng_state()
        saved_state["torch_cuda_all"] = [torch.tensor([1, 2, 3], dtype=torch.uint8)]

        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.device_count", return_value=1
        ), mock.patch("torch.cuda.set_rng_state_all") as set_cuda_state:
            restore_rng_state(saved_state, require_cuda_state=True)

        set_cuda_state.assert_called_once()
        torch.testing.assert_close(
            set_cuda_state.call_args.args[0][0],
            saved_state["torch_cuda_all"][0],
            rtol=0,
            atol=0,
        )

    def test_resume_checkpoint_round_and_digest_validation(self):
        config = {"dataset": "cinic10", "comm_round": 1500}
        checkpoint = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "completed_round": 749,
            "next_round": 750,
            "best_metric": 0.71,
            "resume_config": config,
            "resume_config_digest": resume_config_digest(config),
            "rng_state": capture_rng_state(),
        }

        state = validate_resume_checkpoint(checkpoint)
        self.assertEqual(state["completed_round"], 749)
        self.assertEqual(state["next_round"], 750)
        self.assertAlmostEqual(state["best_metric"], 0.71)

        checkpoint["next_round"] = 751
        with self.assertRaisesRegex(ValueError, "Invalid round metadata"):
            validate_resume_checkpoint(checkpoint)

    def test_resume_config_rejects_training_changes_and_ignores_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            args = SimpleNamespace(
                dataset="cinic10",
                comm_round=1500,
                init_seed=0,
                checkpoint_dir=str(directory / "first"),
                diverse_subnets={"0": {"d": [0]}},
                ckpt_subnets=None,
                client_partition_manifest=None,
            )
            original = build_resume_config(args)

            args.checkpoint_dir = str(directory / "second")
            same_training = build_resume_config(args)
            validate_resume_config(original, same_training)

            args.init_seed = 1
            changed_training = build_resume_config(args)
            with self.assertRaisesRegex(ValueError, "init_seed"):
                validate_resume_config(original, changed_training)

    def test_pre_mixaug_checkpoint_uses_only_legacy_basic_defaults(self):
        saved = {"dataset": "cinic10", "comm_round": 1500}
        current_basic = {
            **saved,
            "augmentation": "basic",
            "randaugment_num_ops": 2,
            "randaugment_magnitude": 6,
            "mix_aug_mode": "none",
            "mixup_alpha": 0.4,
            "cutmix_alpha": 1.0,
        }
        validate_resume_config(saved, current_basic)

        current_mixaug = {
            **current_basic,
            "augmentation": "mixaug",
            "mix_aug_mode": "alternating",
        }
        with self.assertRaisesRegex(ValueError, "augmentation"):
            validate_resume_config(saved, current_mixaug)

    def test_mixaug_resume_config_supports_cifar10_and_cinic10(self):
        for dataset in ("cifar10", "cinic10"):
            args = SimpleNamespace(
                dataset=dataset,
                comm_round=1500,
                init_seed=0,
                augmentation="mixaug",
                randaugment_num_ops=2,
                randaugment_magnitude=6,
                mix_aug_mode="alternating",
                mixup_alpha=0.4,
                cutmix_alpha=1.0,
                diverse_subnets={"0": {"d": [2, 2, 2, 2]}},
                ckpt_subnets=None,
                client_partition_manifest=None,
            )
            saved = build_resume_config(args)
            current = build_resume_config(args)

            validate_resume_config(saved, current)
            self.assertEqual(saved["dataset"], dataset)
            self.assertEqual(saved["augmentation"], "mixaug")
            self.assertEqual(saved["randaugment_num_ops"], 2)
            self.assertEqual(saved["randaugment_magnitude"], 6)

            args.randaugment_magnitude = 7
            changed_policy = build_resume_config(args)
            with self.assertRaisesRegex(ValueError, "randaugment_magnitude"):
                validate_resume_config(saved, changed_policy)


if __name__ == "__main__":
    unittest.main()
