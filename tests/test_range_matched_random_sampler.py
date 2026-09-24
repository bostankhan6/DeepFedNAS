import csv
import os
import tempfile
import unittest

from deepfednas.Server.base_server_model import (
    BaseServerModel,
)


class FakeServer(BaseServerModel):
    def __init__(self, random_macs):
        self.arch_params = {}
        self._random_macs = iter(random_macs)
        super().__init__({}, "TS_range_matched_random", 4)

    def init_model(self, init_params):
        return object()

    def is_max_net(self, arch):
        return arch == self.max_subnet_arch()

    def is_min_net(self, arch):
        return arch == self.min_subnet_arch()

    def get_subnet(self, **kwargs):
        return kwargs

    def add_subnet(self, shared_param_sum, shared_param_count, w_local):
        raise NotImplementedError

    def active_subnet_index(self):
        raise NotImplementedError

    def max_subnet_arch(self):
        return {"d": [4000], "e": [0.1], "w_indices": [0]}

    def min_subnet_arch(self):
        return {"d": [7], "e": [0.1], "w_indices": [0]}

    def random_subnet_arch(self):
        return {"d": [next(self._random_macs)], "e": [0.1], "w_indices": [0]}

    def random_depth_subnet_arch(self):
        return self.random_subnet_arch()

    def random_compound_subnet_arch(self):
        return self.random_subnet_arch()

    def mutate_sample(self, sample_arch, mut_prob):
        return sample_arch

    def architecture_macs(self, arch_config):
        return arch_config["d"][0]


class TestRangeMatchedRandomSampler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.temp_dir.name, "cache.csv")
        with open(self.cache_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["macs", "d", "e", "w_indices"]
            )
            writer.writeheader()
            writer.writerow(
                {"macs": 458, "d": "[458]", "e": "[0.1]", "w_indices": "[0]"}
            )
            writer.writerow(
                {"macs": 3365, "d": "[3403]", "e": "[0.1]", "w_indices": "[0]"}
            )
        self.args = {"subnet_cache_path": self.cache_path}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_uses_cache_endpoints_instead_of_global_endpoints(self):
        server = FakeServer([700])
        server.smallest_subnet_min_idx = {0}
        server.largest_subnet_min_idx = {1}

        sampled_min = server.sample_subnet(0, 0, 0, self.args)
        sampled_max = server.sample_subnet(0, 1, 1, self.args)

        self.assertEqual(sampled_min["d"], [458])
        self.assertEqual(sampled_max["d"], [3403])
        self.assertNotEqual(sampled_min, server.min_subnet_arch())
        self.assertNotEqual(sampled_max, server.max_subnet_arch())

    def test_rejects_random_architectures_outside_operational_range(self):
        server = FakeServer([100, 4000, 700])
        sampled = server.sample_subnet(3, 2, 2, self.args)

        self.assertEqual(sampled["d"], [700])
        self.assertGreaterEqual(server.architecture_macs(sampled), 458)
        self.assertLessEqual(server.architecture_macs(sampled), 3403)

    def test_existing_all_random_sampler_keeps_global_minimum(self):
        server = FakeServer([700])
        server.sampling_method = "TS_all_random"
        server.smallest_subnet_min_idx = {0}

        sampled = server.sample_subnet(0, 0, 0, self.args)

        self.assertEqual(sampled, server.min_subnet_arch())


if __name__ == "__main__":
    unittest.main(verbosity=2)
