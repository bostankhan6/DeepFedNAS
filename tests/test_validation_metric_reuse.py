import random
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from deepfednas.Server.deepfednas_trainer import (
    FLOFA_Trainer,
)


class ValidationMetricReuseTest(unittest.TestCase):
    @staticmethod
    def _trainer(diverse_subnets, checkpoint_subnets, dataset="cinic10"):
        trainer = object.__new__(FLOFA_Trainer)
        trainer.args = SimpleNamespace(
            diverse_subnets=diverse_subnets,
            ckpt_subnets=checkpoint_subnets,
            dataset=dataset,
        )
        trainer.server_model = mock.Mock()
        trainer.client_trainer = mock.Mock()
        return trainer

    def test_identical_checkpoint_subnets_reuse_periodic_metrics(self):
        diverse_subnets = {
            "0": {"d": [0], "e": [0.1], "w_indices": [0]},
            "1": {"d": [2], "e": [0.25], "w_indices": [9]},
        }
        trainer = self._trainer(
            diverse_subnets,
            list(diverse_subnets.values()),
        )

        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        mean_metric = trainer._checkpoint_mean_metric(
            {"0": 0.71, "1": 0.83},
        )

        self.assertAlmostEqual(mean_metric, 0.77)
        trainer.server_model.get_subnet.assert_not_called()
        trainer.client_trainer.set_test_model.assert_not_called()
        trainer.client_trainer.local_test.assert_not_called()
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(
            all(
                np.array_equal(current, saved)
                if isinstance(current, np.ndarray)
                else current == saved
                for current, saved in zip(np.random.get_state(), numpy_state)
            )
        )
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_default_checkpoint_subnets_reuse_periodic_metrics(self):
        diverse_subnets = {
            "0": {"d": [0]},
            "1": {"d": [2]},
        }
        trainer = self._trainer(diverse_subnets, None)

        mean_metric = trainer._checkpoint_mean_metric(
            {"0": 0.65, "1": 0.75},
        )

        self.assertAlmostEqual(mean_metric, 0.70)
        trainer.client_trainer.local_test.assert_not_called()

    def test_distinct_checkpoint_subnets_keep_independent_evaluation(self):
        diverse_subnets = {"0": {"d": [0]}}
        checkpoint_subnets = [{"d": [1]}, {"d": [2]}]
        trainer = self._trainer(diverse_subnets, checkpoint_subnets)
        trainer.server_model.get_subnet.side_effect = ["subnet-1", "subnet-2"]
        trainer.client_trainer.local_test.side_effect = [
            {"test_correct": 72, "test_total": 100},
            {"test_correct": 84, "test_total": 100},
        ]

        mean_metric = trainer._checkpoint_mean_metric({"0": 0.91})

        self.assertAlmostEqual(mean_metric, 0.78)
        self.assertEqual(trainer.server_model.get_subnet.call_count, 2)
        self.assertEqual(trainer.client_trainer.local_test.call_count, 2)
        trainer.client_trainer.local_test.assert_has_calls(
            [mock.call(True), mock.call(True)]
        )

    def test_missing_periodic_metric_keeps_independent_evaluation(self):
        diverse_subnets = {"0": {"d": [0]}, "1": {"d": [2]}}
        trainer = self._trainer(
            diverse_subnets,
            list(diverse_subnets.values()),
        )
        trainer.server_model.get_subnet.side_effect = ["subnet-0", "subnet-1"]
        trainer.client_trainer.local_test.side_effect = [
            {"test_correct": 60, "test_total": 100},
            {"test_correct": 80, "test_total": 100},
        ]

        mean_metric = trainer._checkpoint_mean_metric({"0": 0.60})

        self.assertAlmostEqual(mean_metric, 0.70)
        self.assertEqual(trainer.client_trainer.local_test.call_count, 2)

    def test_ptb_fallback_preserves_perplexity_selection_metric(self):
        trainer = self._trainer(
            {"0": {"d": [0]}},
            [{"d": [1]}, {"d": [2]}],
            dataset="ptb",
        )
        trainer.client_trainer.local_test.side_effect = [
            {"test_ppl": 90.0},
            {"test_ppl": 110.0},
        ]

        mean_metric = trainer._checkpoint_mean_metric({"0": 80.0})

        self.assertAlmostEqual(mean_metric, 100.0)

    def test_training_round_reuses_metrics_for_best_checkpoint_decision(self):
        diverse_subnets = {"0": {"d": [0]}, "1": {"d": [2]}}
        trainer = self._trainer(
            diverse_subnets,
            list(diverse_subnets.values()),
        )
        trainer.args.client_num_in_total = 0
        trainer.args.client_num_per_round = 0
        trainer.args.dry_run = False
        trainer.args.weight_dataset = False
        trainer.args.feddyn = False
        trainer.args.wandb_watch = False
        trainer.args.frequency_of_the_test = 20
        trainer.args.comm_round = 1500
        trainer.args.efficient_test = True
        trainer.wt_avg_sched_method = "test"
        trainer.weighted_avg_scheduler = {"test": lambda _round: []}
        trainer.feddyn = False
        trainer.prev_best = 0.75
        trainer._aggregate = mock.Mock(return_value={})
        trainer._efficient_local_test_on_all_clients = mock.Mock(
            return_value=({}, {"0": 0.71, "1": 0.83})
        )

        with mock.patch(
            "deepfednas.Server."
            "deepfednas_trainer.wandb.log"
        ):
            trainer.train_one_round(20)

        self.assertAlmostEqual(trainer.prev_best, 0.77)
        trainer.client_trainer.local_test.assert_not_called()
        save_calls = trainer.server_model.save.call_args_list
        self.assertEqual(save_calls[0].args[0], "best_checkpoint_supernet.pt")
        self.assertAlmostEqual(save_calls[0].args[1]["best_metric"], 0.77)
        self.assertEqual(save_calls[1].args[0], "latest_round_model.pt")
        self.assertEqual(save_calls[1].args[1]["completed_round"], 20)
        self.assertEqual(save_calls[1].args[1]["next_round"], 21)


if __name__ == "__main__":
    unittest.main()
