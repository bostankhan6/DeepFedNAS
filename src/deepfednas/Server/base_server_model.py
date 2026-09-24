from abc import ABC, abstractmethod
import wandb
import os
import torch
import numpy as np
from wandb.sdk.lib import RunDisabled
import copy
import random
import heapq
import pandas as pd
import logging

from deepfednas.nas.deepfednas_fitness_maximizer import run_entropy_max_ga
from deepfednas.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    capture_rng_state,
    resume_config_digest,
)

# def _decode_chromosome_from_row(row, arch_params):
#     num_stages = arch_params['num_stages']
#     max_extra_blocks = arch_params['max_extra_blocks_per_stage']
#     exp_choices = arch_params['expansion_ratio_choices']

#     num_depth_genes = num_stages
#     num_exp_genes = num_stages * (max_extra_blocks + 1)
    
#     # Extract gene values from the row
#     genes = row.filter(like='gene_').values

#     d_vec = genes[:num_depth_genes].astype(int).tolist()
#     e_indices = genes[num_depth_genes : num_depth_genes + num_exp_genes].astype(int)
#     w_indices = genes[num_depth_genes + num_exp_genes :].astype(int).tolist()

#     # Convert expansion ratio indices to their float values
#     e_vec = [exp_choices[i] for i in e_indices]
    
#     return d_vec, e_vec, w_indices

def _decode_chromosome_from_row(row, arch_params):
    """
    Decodes an architecture dictionary from a row of a DataFrame.
    """
    try:
        import ast
        d_vec = ast.literal_eval(row['d'])
        e_vec = ast.literal_eval(row['e'])
        w_indices = ast.literal_eval(row['w_indices'])
        return {'d': d_vec, 'e': e_vec, 'w_indices': w_indices}
    except Exception as e:
        logging.info(f"Error decoding row {row.name}: {e}.")
        return None

class BaseServerModel(ABC):
    def __init__(
        self, init_params, sampling_method, num_cli_total, cli_subnet_track=None,
    ):
        self.model = self.init_model(init_params)
        self.checkpoint_dir = None
        self.upload_checkpoints = True
        self.resume_config = None
        self.sampling_method = sampling_method
        if cli_subnet_track is None:
            self.cli_subnet_track = dict()
            for idx in range(num_cli_total):
                self.cli_subnet_track[idx] = dict()
                self.cli_subnet_track[idx]["largest"] = 0
                self.cli_subnet_track[idx]["smallest"] = 0
        else:
            self.cli_subnet_track = cli_subnet_track
        self.client_sample_count = None
        self.cli_indices = []
        self.top_k = 1
        self.bottom_k = 1
        self.largest_subnet_min_idx = set()
        self.smallest_subnet_min_idx = set()
        self.max_sample_count_idx = -1
        self.second_max_sample_count_idx = -1
        self.cur_round = -1
        self.cur_arch = None
        self.subnet_sampling = dict()
        self.subnet_sampling["static"] = self.static_sample
        self.subnet_sampling["dynamic"] = self.dynamic_sample
        self.subnet_sampling["all_random"] = self.random_subnet_sample
        self.subnet_sampling["compound"] = self.compound_subnet_sample
        self.subnet_sampling["sandwich_all_random"] = self.sandwich_all_subnet_sample
        self.subnet_sampling["sandwich_compound"] = self.sandwich_compound_subnet_sample
        self.subnet_sampling["TS_all_random"] = self.tracking_sandwich_all_subnet_sample
        self.subnet_sampling[
            "TS_range_matched_random"
        ] = self.tracking_sandwich_range_matched_random_sample
        self.subnet_sampling["TS_entropy_maximizer"] = self.tracking_sandwich_entropy_maximizer
        self.subnet_sampling[
            "TS_compound"
        ] = self.tracking_sandwich_compound_subnet_sample
        self.subnet_sampling["max_sample_count"] = self.max_client_dataset_all_subnet
        self.subnet_sampling["multi_sandwich"] = self.multi_sandwich_sample
        self.subnet_sampling["TS_KD"] = self.tracking_sandwich_kd
        self.subnet_sampling["PS"] = self.ps

        # self.subnet_sampling["TS_cached_entropy_maximizer"] = self.tracking_sandwich_cached_entropy_maximizer
        
        self.subnet_sampling["TS_optimal_path"] = self.tracking_sandwich_optimal_path_sampler

        self.subnet_cache = None
        self.op_smallest_subnet = None
        self.op_largest_subnet = None
        self.op_macs_min = None
        self.op_macs_max = None

    def configure_checkpointing(
        self, checkpoint_dir=None, upload_checkpoints=True, resume_config=None
    ):
        """Configure the local checkpoint location and W&B upload policy."""
        self.upload_checkpoints = upload_checkpoints
        self.resume_config = resume_config
        if checkpoint_dir is not None:
            self.checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        logging.info(
            "Checkpoint directory: %s (W&B upload: %s)",
            self.checkpoint_dir or "auto",
            self.upload_checkpoints,
        )

    def _checkpoint_path(self, name):
        """Resolve a checkpoint path without requiring an active W&B run."""
        if self.checkpoint_dir is None:
            run_dir = getattr(wandb.run, "dir", None)
            if self.upload_checkpoints and run_dir and not isinstance(wandb.run, RunDisabled):
                self.checkpoint_dir = run_dir
            else:
                self.checkpoint_dir = os.path.abspath(os.path.join(os.getcwd(), "checkpoints"))
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        return os.path.join(self.checkpoint_dir, name)
        
    def _prepare_subnet_cache(self, args):
        """Loads and prepares the optimal path cache. Called once."""
        if self.subnet_cache is not None:
            return # Avoid reloading

        cache_path = args.get("subnet_cache_path")
        if not cache_path or not os.path.exists(cache_path):
            raise FileNotFoundError(f"Subnet cache file not found at: {cache_path}.")

        logging.info(f"Loading and preparing optimal path cache from: {cache_path}")
        df = pd.read_csv(cache_path)
        df = df.sort_values(by='macs').reset_index(drop=True)

        # Automatically determine boundaries from the cache itself
        self.op_smallest_subnet = _decode_chromosome_from_row(df.iloc[0], self.arch_params)
        self.op_largest_subnet = _decode_chromosome_from_row(df.iloc[-1], self.arch_params)
        cached_macs_min = float(df.iloc[0]["macs"])
        cached_macs_max = float(df.iloc[-1]["macs"])
        try:
            self.op_macs_min = float(self.architecture_macs(self.op_smallest_subnet))
            self.op_macs_max = float(self.architecture_macs(self.op_largest_subnet))
        except NotImplementedError:
            # Retain compatibility with older server models that consume the
            # optimal-path cache but cannot calculate generic architecture MACs.
            self.op_macs_min = cached_macs_min
            self.op_macs_max = cached_macs_max

        if not np.isclose(self.op_macs_min, cached_macs_min) or not np.isclose(
            self.op_macs_max, cached_macs_max
        ):
            logging.warning(
                "Subnet-cache MAC labels differ from values recomputed with the "
                "active architecture parameters: cache=[%.0f, %.0f], "
                "recomputed=[%.0f, %.0f]. Using the recomputed bounds.",
                cached_macs_min,
                cached_macs_max,
                self.op_macs_min,
                self.op_macs_max,
            )
        
        # Store the entire path for random sampling
        self.subnet_cache = [_decode_chromosome_from_row(row, self.arch_params) for _, row in df.iterrows()]
        self.subnet_cache = [arch for arch in self.subnet_cache if arch is not None]

        logging.info(f"Subnet cache prepared with {len(self.subnet_cache)} subnets.")
        logging.info(f"Operational Smallest Subnet (MACs): {self.op_macs_min/1e6:.2f}M")
        logging.info(f"Operational Largest Subnet (MACs): {self.op_macs_max/1e6:.2f}M")

    def architecture_macs(self, arch_config):
        """Return MACs for an architecture when the server supports it."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement architecture_macs()."
        )

    def random_subnet_arch_in_macs_range(
        self, min_macs, max_macs, max_attempts=1000
    ):
        """Uniformly sample architecture genes, conditioned on a MAC interval."""
        for _ in range(max_attempts):
            arch_config = self.random_subnet_arch()
            macs = self.architecture_macs(arch_config)
            if min_macs <= macs <= max_macs:
                return arch_config
        raise RuntimeError(
            "Unable to sample a random subnet inside the operational MAC range "
            f"[{min_macs}, {max_macs}] after {max_attempts} attempts."
        )

    def set_top_bottom_k(self, top_k, bottom_k):
        self.top_k = top_k
        self.bottom_k = bottom_k

    def load_client_sample_counts(self, sample_counts):
        self.client_sample_count = sample_counts

    def update_sample(self):
        self.largest_subnet_min_idx = set()
        self.smallest_subnet_min_idx = set()
        temp = set()
        heap = []
        available_cli = set()
        for cli in self.cli_indices:
            available_cli.add(cli)
        for i in available_cli:
            heapq.heappush(heap, self.cli_subnet_track[i]["largest"])
        for i in range(self.top_k):
            temp.add(heap[i])
        for cli in available_cli:
            if len(self.largest_subnet_min_idx) >= self.top_k:
                break
            if self.cli_subnet_track[cli]["largest"] in temp:
                self.largest_subnet_min_idx.add(cli)
        available_cli = available_cli - self.largest_subnet_min_idx
        # of the remaining clients temporally load balance min subnet
        heap = []
        for i in available_cli:
            heapq.heappush(heap, self.cli_subnet_track[i]["smallest"])
        temp = set()
        for i in range(self.bottom_k):
            temp.add(heap[i])
        for cli in available_cli:
            if len(self.smallest_subnet_min_idx) >= self.bottom_k:
                break
            if self.cli_subnet_track[cli]["smallest"] in temp:
                self.smallest_subnet_min_idx.add(cli)

        max_sample_count_idx = -1
        for idx in range(len(self.cli_indices)):
            cur_idx = self.cli_indices[idx]
            if (
                    max_sample_count_idx == -1
                    or self.client_sample_count[cur_idx]
                    > self.client_sample_count[max_sample_count_idx]
            ):
                max_sample_count_idx = cur_idx
        # Find min smallest
        second_max_sample_count_idx = -1
        for idx in range(len(self.cli_indices)):
            cur_idx = self.cli_indices[idx]
            if cur_idx != max_sample_count_idx:
                if (
                    second_max_sample_count_idx == -1
                    or self.client_sample_count[cur_idx]
                    > self.client_sample_count[second_max_sample_count_idx]
                ):
                    second_max_sample_count_idx = cur_idx
        self.max_sample_count_idx = max_sample_count_idx
        self.second_max_sample_count_idx = second_max_sample_count_idx

    def set_cli_indices(self, client_indices):
        self.cli_indices = client_indices

    def set_model_params(self, params):
        self.model.load_state_dict(params)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def get_model_copy(self):
        return copy.deepcopy(self.model)

    # superimpose a (numpy) vectorized version of it's MAX network onto supernetwork
    def superimpose_vec(self, vec):
        with torch.no_grad():
            vec = torch.from_numpy(vec)
            idx = 0
            for p in self.model.parameters():
                idx_next = idx + p.view(-1).size()[0]
                p.copy_(vec[idx:idx_next].reshape(p.shape))
                idx = idx_next

    # sums supernetwork with a (numpy) vectorized version of it's MAX network in place
    def sum_supernet_w_vec(self, vec):
        with torch.no_grad():
            vec = torch.from_numpy(vec)
            idx = 0
            for p in self.model.parameters():
                idx_next = idx + p.view(-1).size()[0]
                p.copy_(p.data + vec[idx:idx_next].reshape(p.shape))
                idx = idx_next

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    def forward(self, x):
        return self.model(x)
    
    def save(self, name, training_state=None):
        """Save one local checkpoint file.

        Callers should use stable names (for example ``best_checkpoint_supernet.pt``
        and ``latest_round_model.pt``) so repeated saves overwrite the previous
        version instead of accumulating round-specific files.
        """
        save_data = {"checkpoint_format_version": CHECKPOINT_FORMAT_VERSION}
        save_data["params"] = self.get_model_params()
        
        ### NEW: Save the architecture parameters dictionary
        # Check if the instance has arch_params (for GenericServerOFA)
        if hasattr(self, 'arch_params'):
            save_data["arch_params"] = self.arch_params
        
        ### NEW: Also save the model class name for easy reconstruction
        save_data["model_class_name"] = self.model.__class__.__name__

        # Capture this only after the round is complete. It is restored after
        # all continuation objects have been reconstructed, immediately before
        # the next training round starts.
        save_data["rng_state"] = capture_rng_state()
        # Keep the two historical keys so existing evaluation/loading tools can
        # still consume newly generated checkpoints.
        save_data["torch_rng_state"] = save_data["rng_state"]["torch_cpu"]
        save_data["numpy_rng_state"] = save_data["rng_state"]["numpy"]
        save_data["cli_subnet_track"] = self.cli_subnet_track
        if wandb.run is not None and not isinstance(wandb.run, RunDisabled):
            save_data["wandb_run_id"] = wandb.run.id

        if self.resume_config is not None:
            save_data["resume_config"] = self.resume_config
            save_data["resume_config_digest"] = resume_config_digest(
                self.resume_config
            )

        if training_state is not None:
            save_data.update(training_state)
        
        filename = self._checkpoint_path(name)
        atomic_torch_save(save_data, filename)
        return filename

    def upload_checkpoint(self, name):
        """Upload an existing local checkpoint to W&B when explicitly enabled."""
        if not self.upload_checkpoints or isinstance(wandb.run, RunDisabled):
            return False

        filename = self._checkpoint_path(name)
        if not os.path.isfile(filename):
            logging.warning("Cannot upload missing checkpoint: %s", filename)
            return False
        wandb.save(filename, base_path=self.checkpoint_dir)
        return True

    #TODO: Abstract this function away and allow different supernetworks to specify expected input size
    def wandb_pass(self):
        self.model.set_max_net()
        self.model.eval()
        self.model(torch.zeros((1, 3, 32, 32)))

    def feddyn_global_model_update(self, alpha):
        for server_param, state_param in zip(
            self.model.parameters(), self.server_state.parameters()
        ):
            server_param.data -= (1 / alpha) * state_param

    def sample_subnet(self, round_num, client_idx, idx, args):
        return self.subnet_sampling[self.sampling_method](
            round_num, client_idx, idx, args
        )

    def static_sample(self, round_num, client_idx, idx, args):
        subnets = args["diverse_subnets"]
        id = str(client_idx % len(subnets))
        return self.get_subnet(**subnets[id])

    def dynamic_sample(self, round_num, client_idx, idx, args):
        subnets = args["diverse_subnets"]
        id = str((client_idx + round_num) % len(subnets))
        return self.get_subnet(**subnets[id])

    def random_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        arch_config = self.random_subnet_arch()
        return self.get_subnet(**arch_config)

    def compound_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        arch_config = self.random_compound_subnet_arch()
        return self.get_subnet(**arch_config)

    def sandwich_all_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if idx == (round_num % args["client_per_round"]):
            arch_config = self.min_subnet_arch()
        elif idx == ((round_num + 1) % args["client_per_round"]):
            arch_config = self.max_subnet_arch()
        else:
            arch_config = self.random_subnet_arch()
        return self.get_subnet(**arch_config)

    def sandwich_compound_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if idx == (round_num % args["client_per_round"]):
            arch_config = self.min_subnet_arch()
        elif idx == ((round_num + 1) % args["client_per_round"]):
            arch_config = self.max_subnet_arch()
        else:
            arch_config = self.random_compound_subnet_arch()
        return self.get_subnet(**arch_config)

    def tracking_sandwich_all_subnet_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)

    def tracking_sandwich_range_matched_random_sample(
        self, round_num, client_idx, idx, args
    ):
        """Random sandwich sampling within the optimal-path operational range.

        The endpoint roles use the same cache-defined minimum and maximum as
        ``TS_optimal_path``. Interior roles remain random architecture samples;
        the cache contributes only the comparison interval and endpoints.
        """
        if self.subnet_cache is None:
            self._prepare_subnet_cache(args)

        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = {
                "largest": 0,
                "smallest": 0,
            }

        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.op_smallest_subnet
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            arch_config = self.op_largest_subnet
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_subnet_arch_in_macs_range(
                self.op_macs_min, self.op_macs_max
            )
            if arch_config == self.op_largest_subnet:
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif arch_config == self.op_smallest_subnet:
                self.cli_subnet_track[client_idx]["smallest"] += 1

        return self.get_subnet(**arch_config)


    def tracking_sandwich_entropy_maximizer(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)

        if client_idx not in self.cli_subnet_track:
            self.cli_subnet_track[client_idx] = {"largest": 0, "smallest": 0}

        # 1. Sandwich logic (unchanged)
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            # 2. Middle clients get DeepMAD-optimized subnets
            target_macs = np.random.uniform(self._macs_min, self._macs_max)

            ga_arch_params = self.arch_params
            ga_width_options = self.arch_params['width_multiplier_choices']
            ga_exp_options = self.arch_params['expansion_ratio_choices']
            ga_depth_choices = list(range(self.arch_params['max_extra_blocks_per_stage'] + 1))
            
            ### CORRECTED: Read parameters from the top-level 'args' dictionary ###
            ga_rho0_constraint = args.get('supernet_rho0_constraint', 2.0)
            ga_effectiveness_weight = args.get('supernet_effectiveness_fitness_weight', 100)
            ga_pop_size = args.get('ga_pop_size', 128)
            ga_generations = args.get('ga_generations', 100)
            ga_mutate_p = args.get('ga_mutate_p', 0.3)

            temp_arch_config, best_entropy_score = run_entropy_max_ga(
                mac_budget=target_macs,
                arch_config_params=ga_arch_params,
                width_mult_options=ga_width_options,
                depth_choices=ga_depth_choices,
                exp_opt_values=ga_exp_options,
                rho0_constraint=ga_rho0_constraint,
                effectiveness_fitness_weight=ga_effectiveness_weight,
                pop_size=ga_pop_size,
                generations=ga_generations,
                mutate_p=ga_mutate_p,
                seed=round_num * 100 + client_idx
            )

            if temp_arch_config is None:
                logging.info(f"WARNING: GA failed for target MACs {target_macs/1e6:.2f}M. Falling back to random_subnet_arch.")
                arch_config = self.random_subnet_arch()
            else:
                arch_config = temp_arch_config

            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1

        # 3. Return the instantiated subnet
        return self.get_subnet(**arch_config)

    ### --- START NEW SAMPLER FUNCTION --- ###
    # def tracking_sandwich_cached_entropy_maximizer(self, round_num, client_idx, idx, args):
        
    #     if self.subnet_cache is None:
    #         self._prepare_subnet_cache(args)

    #     np.random.seed(round_num + client_idx)

    #     if client_idx not in self.cli_subnet_track:
    #         self.cli_subnet_track[client_idx] = {"largest": 0, "smallest": 0}

    #     # 1. Sandwich logic (unchanged)
    #     if client_idx in self.smallest_subnet_min_idx:
    #         arch_config = self.min_subnet_arch()
    #         self.cli_subnet_track[client_idx]["smallest"] += 1

    #     elif client_idx in self.largest_subnet_min_idx:
    #         arch_config = self.max_subnet_arch()
    #         self.cli_subnet_track[client_idx]["largest"] += 1
        
    #     # 2. Middle clients get a pre-computed subnet from the cache
    #     else:
    #         if self.subnet_cache is None:
    #             # Fallback to random sampling if cache is not available
    #             logging.info("WARNING: Architecture cache is not loaded. Falling back to random sampling.")
    #             arch_config = self.random_subnet_arch()
    #         else:
    #             # Pick a random target MACs budget
    #             target_macs = np.random.uniform(self._macs_min, self._macs_max)
                
    #             # Find the architecture in the cache with the closest MACs to the target
    #             # This is an extremely fast lookup
    #             closest_arch_row = self.subnet_cache.iloc[(self.subnet_cache['macs'] - target_macs).abs().argsort()[:1]]
                
    #             # Extract the architecture configuration from the row
    #             arch_config = {
    #                 "d": closest_arch_row.iloc[0]['d'],
    #                 "e": closest_arch_row.iloc[0]['e'],
    #                 "w_indices": closest_arch_row.iloc[0]['w_indices'],
    #             }
            
    #             logging.info(f"Client {client_idx} selected cached architecture with MACs: {closest_arch_row.iloc[0]['macs']:.2f} for target {target_macs:.2f}")
    #             logging.info(f"Selected architecture: d={arch_config['d']}, e={arch_config['e'][0]}, w_indices={arch_config['w_indices']}")

    #         # Update tracking for load balancing
    #         if self.is_max_net(arch_config):
    #             self.cli_subnet_track[client_idx]["largest"] += 1
    #         elif self.is_min_net(arch_config):
    #             self.cli_subnet_track[client_idx]["smallest"] += 1
        
    #     # 3. Return the instantiated subnet
    #     return self.get_subnet(**arch_config)
    ### --- END NEW SAMPLER FUNCTION --- ###

    def tracking_sandwich_optimal_path_sampler(self, round_num, client_idx, idx, args):
        """
        A structured, principled sampler that uses a pre-computed optimal path.
        """
        # 1. Prepare the cache on the first call
        if self.subnet_cache is None:
            self._prepare_subnet_cache(args)

        np.random.seed(round_num + client_idx)

        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()

        # 2. Assign clients to roles based on tracking
        if client_idx in self.smallest_subnet_min_idx:
            # This client gets the pre-defined operational smallest subnet
            arch_config = self.op_smallest_subnet
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            # This client gets the pre-defined operational largest subnet
            arch_config = self.op_largest_subnet
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            # Other clients get a random subnet from the optimal path cache
            arch_config = random.choice(self.subnet_cache)
            if arch_config == self.op_largest_subnet:
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif arch_config == self.op_smallest_subnet:
                self.cli_subnet_track[client_idx]["smallest"] += 1
            
        if arch_config is None:
            # Fallback in case of an issue
            logging.info("Warning: Sampled arch_config is None. Falling back to random architecture.")
            return self.get_subnet(**self.random_subnet_arch())
            
        # 3. Return the instantiated subnet
        return self.get_subnet(**arch_config)

    def tracking_sandwich_compound_subnet_sample(
        self, round_num, client_idx, idx, args
    ):
        np.random.seed(round_num + client_idx)
        if self.cli_subnet_track[client_idx] is None:
            self.cli_subnet_track[client_idx] = dict()
        if client_idx in self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_compound_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)

    def max_client_dataset_all_subnet(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if client_idx == self.second_max_sample_count_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx == self.max_sample_count_idx:
            arch_config = self.max_subnet_arch()
            self.cli_subnet_track[client_idx]["largest"] += 1
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        return self.get_subnet(**arch_config)
    def multi_sandwich_sample(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        model_dict = dict()
        model_dict["max"] = self.get_subnet(**self.max_subnet_arch())
        model_dict["min"] = self.get_subnet(**self.min_subnet_arch())
        model_dict["mid"] = []
        for _ in range(args["K"]):
            model_dict["mid"].append(self.get_subnet(**self.random_subnet_arch()))
        return model_dict

    def tracking_sandwich_kd(self, round_num, client_idx, idx, args):
        np.random.seed(round_num + client_idx)
        if client_idx == self.smallest_subnet_min_idx:
            arch_config = self.min_subnet_arch()
            self.cli_subnet_track[client_idx]["smallest"] += 1
        elif client_idx in self.largest_subnet_min_idx:
            model_dict = dict()
            model_dict["max"] = None
            model_dict["min"] = None
            model_dict["mid"] = [self.get_subnet(**self.max_subnet_arch())]
            self.cli_subnet_track[client_idx]["largest"] += 1
            return model_dict
        else:
            arch_config = self.random_subnet_arch()
            if self.is_max_net(arch_config):
                self.cli_subnet_track[client_idx]["largest"] += 1
            elif self.is_min_net(arch_config):
                self.cli_subnet_track[client_idx]["smallest"] += 1
        model_dict = dict()
        model_dict["max"] = self.get_subnet(**self.max_subnet_arch())
        model_dict["min"] = None
        model_dict["mid"] = [arch_config]
        return model_dict

    def ps(self, round_num, client_idx, idx, args):
        if self.cur_round < round_num:
            self.cur_round = round_num
            #np.random.seed(round_num)
            if self.cur_round < args["ps_depth_only"]:
                self.cur_arch = self.random_depth_subnet_arch()
            else:
                self.cur_arch = self.random_subnet_arch()
        return self.get_subnet(**self.cur_arch)

    def crossover_sample(self, par1, par2):
        new_sample = copy.deepcopy(par1)
        for key in new_sample.keys():
            if not isinstance(new_sample[key], list):
                continue
            for i in range(len(new_sample[key])):
                new_sample[key][i] = random.choice([par1[key][i], par2[key][i]])
        return new_sample

    # TODO:Need to reimpliment updated sandwich sampling with dynamic width
    def dynamic_width_sandwich_sample(self, round_num, client_idx, idx, args):
        pass

    @abstractmethod
    def init_model(self, init_params):
        pass

    @abstractmethod
    def is_max_net(self, arch):
        pass

    @abstractmethod
    def is_min_net(self, arch):
        pass

    @abstractmethod
    def get_subnet(self, **kwargs):
        pass

    @abstractmethod
    def add_subnet(self, shared_param_sum, shared_param_count, w_local):
        pass

    @abstractmethod
    def active_subnet_index(self):
        pass

    @abstractmethod
    def max_subnet_arch(self):
        pass

    @abstractmethod
    def min_subnet_arch(self):
        pass

    @abstractmethod
    def random_subnet_arch(self):
        pass

    @abstractmethod
    def random_depth_subnet_arch(self):
        pass

    @abstractmethod
    def random_compound_subnet_arch(self):
        pass

    @abstractmethod
    def mutate_sample(self, sample_arch, mut_prob):
        pass
