import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import os
import sys
import json
from tqdm import tqdm
import random
import copy
import time
import argparse

# --- Import Project's Modules ---
try:
    from deepfednas.Server.generic_server_model import GenericServerOFA
    from deepfednas.utils.subnet_cost import subnet_macs
    from deepfednas.data.cifar10.data_loader import load_partition_data_cifar10
    from deepfednas.data.cifar100.data_loader import load_partition_data_cifar100
    from deepfednas.data.cinic10.data_loader import load_partition_data_cinic10
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Please check your PYTHONPATH.")
    sys.exit(1)

# ==========================================================================================
# --- 1. CONFIGURATION SECTION ---
# ==========================================================================================
MODEL_PATHS = [
    "trained_models/Cinic10_baseline_best_checkpoint_final.pt"
]
DATASET_NAME = 'cinic10' # Ensure this matches your model and data
DATA_PATH = "data/cinic10" # Ensure this matches your dataset path

# Base directory for all output files
MAIN_DIR = "evaluation/bsaeline_cinic10_results_on_unseen_val"

NUM_SEARCH_RUNS = 1 # Number of evolutionary runs per MACs target/bin
BASE_SEED = 42
NUM_SAMPLES_FOR_PREDICTOR = 10000
PREDICTOR_EPOCHS = 400
PREDICTOR_LR = 1e-4
PREDICTOR_BATCH_SIZE = 256
EVO_POPULATION_SIZE = 256
EVO_GENERATIONS = 512
EVO_PARENT_RATIO = 0.25
EVO_MUTATION_RATIO = 0.3

BATCH_SIZE = 2048
GPU_ID = 0

PREDICTOR_DATASET_FNAME = "predictor_dataset.csv"
PREDICTOR_MODEL_FNAME = "predictor_model.pt"

# Checkpoint interval for predictor dataset generation
CHECKPOINT_INTERVAL = 100 # Save dataset every K samples

# --- MACS TARGET CONFIGURATION ---
# Choose between 'linspace' or 'binned'
MACS_TARGET_MODE = 'linspace' # 'linspace' for N targets, 'binned' for predefined bins

# Configuration for 'linspace' mode:
MACS_START = 0.95e9    # 0.95 Billion
MACS_END = 3.75e9      # 3.75 Billion
NUM_MACS_TARGETS = 5  # Number of discrete MACs targets to generate

# Configuration for 'binned' mode:
MACS_BINS = {
    "0.45-0.95 B": (0.45e9, 0.95e9),
    "0.95-1.45 B": (0.95e9, 1.45e9),
    "1.45-2.45 B": (1.45e9, 2.45e9),
    "2.45-3.75 B": (2.45e9, 3.75e9),
}
# IMPORTANT: When using 'binned' mode, the search constraint for a bin (e.g., "0.45-0.95 B")
# will be the upper bound of that bin (0.95e9 in this example).

# ==========================================================================================
# --- 2. Classes and Helper Functions ---
# ==========================================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_supernet_and_config(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch_params = checkpoint["arch_params"]
    arch_params['original_stage_base_channels'] = np.array(arch_params['original_stage_base_channels'])
    server_model = GenericServerOFA(arch_params=arch_params, sampling_method='all_random', num_cli_total=1)
    server_model.set_model_params(checkpoint["params"])
    return server_model, arch_params

def get_data_loaders(dataset_name, data_path, batch_size):
    loader_map = {'cifar10': load_partition_data_cifar10, 'cifar100': load_partition_data_cifar100, 'cinic10': load_partition_data_cinic10}
    loader_fn = loader_map.get(dataset_name)
    if not loader_fn: raise ValueError(f"Unsupported dataset: {dataset_name}")
    _, _, train_loader_global, test_loader_global, _, _, _, _ = loader_fn(dataset_name, data_path, "hetero", 1, 1, batch_size=batch_size, val_batch_size=batch_size)
    train_full_dataset = train_loader_global.dataset
    test_dataset = test_loader_global.dataset
    val_set_size = 10000
    train_set_size = len(train_full_dataset) - val_set_size
    _, val_subset = random_split(train_full_dataset, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Created a validation set of {len(val_subset)} images.")
    return val_loader, test_loader

def evaluate_accuracy(model, data_loader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="    Evaluating Subnet", leave=False, ncols=100):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100

def arch_to_feature_vector(arch_dict, arch_params):
    depth_features = np.array(arch_dict['d'])
    exp_choices = arch_params['expansion_ratio_choices']
    exp_one_hot = np.zeros(len(arch_dict['e']) * len(exp_choices))
    for i, val in enumerate(arch_dict['e']):
        exp_one_hot[i * len(exp_choices) + exp_choices.index(val)] = 1
    width_features = np.array(arch_dict['w_indices'])
    return np.concatenate([depth_features, exp_one_hot, width_features])

class AccuracyPredictor(nn.Module):
    def __init__(self, input_features, hidden_dim1=400, hidden_dim2=400):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_features, hidden_dim1), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, 1),
        )
    def forward(self, x):
        return self.layer(x)

class EvolutionFinder:
    def __init__(self, server_model, arch_params, predictor, args):
        self.server_model = server_model
        self.arch_params = arch_params
        self.predictor = predictor
        self.args = args
        self.device = next(predictor.parameters()).device
        sample_arch = self.server_model.random_subnet_arch()
        self.feature_len = len(arch_to_feature_vector(sample_arch, self.arch_params))

    def accuracy_predictor(self, arch_dict):
        feature_vector = arch_to_feature_vector(arch_dict, self.arch_params)
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.predictor(feature_tensor).item()

    def random_valid_sample(self, constraint):
        for _ in range(2000): # Increased attempts to find a valid sample
            arch = self.server_model.random_subnet_arch()
            macs, _ = subnet_macs(arch['d'], arch['e'], arch['w_indices'], self.arch_params['width_multiplier_choices'], self.arch_params)
            if macs <= constraint:
                return arch
        return None

    def mutate_sample(self, sample_arch):
        return self.server_model.mutate_sample(sample_arch, mut_prob=0.3)

    def crossover_sample(self, arch1, arch2):
        new_sample = copy.deepcopy(arch1)
        for key in new_sample:
            if isinstance(new_sample[key], list):
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice([arch1[key][i], arch2[key][i]])
        return new_sample

    def run_evolution_search(self, constraint, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # --- Initial population generation (Robust Logic from Script A) ---
        population = []
        for _ in range(self.args.population_size * 2): # Try more times to fill population
            sample = self.random_valid_sample(constraint)
            if sample is not None:
                population.append((self.accuracy_predictor(sample), sample))
            if len(population) >= self.args.population_size:
                break
        
        if len(population) < self.args.population_size:
            raise RuntimeError(f"Could not generate initial population of size {self.args.population_size} for MACs <= {constraint/1e9:.2f}B. Found only {len(population)} valid samples.")
            
        population = sorted(population, key=lambda x: x[0], reverse=True)[:self.args.population_size] # Trim to exact population size

        # --- Evolutionary loop (Robust Logic from Script A) ---
        for _ in tqdm(range(self.args.generations), desc="    Evolving Subnets", leave=False):
            parents = sorted(population, key=lambda x: x[0], reverse=True)[:self.args.parent_size]
            next_population = list(parents)
            
            mutation_attempts = 0
            # Limit attempts to prevent infinite loops if finding valid children is hard
            while len(next_population) < self.args.population_size and mutation_attempts < self.args.population_size * 5:
                p1 = random.choice(parents)[1]
                
                child = None
                if random.random() < self.args.mutation_ratio:
                    child = self.mutate_sample(p1)
                else:
                    p2 = random.choice(parents)[1]
                    child = self.crossover_sample(p1, p2)
                
                if child is not None:
                    macs, _ = subnet_macs(child['d'], child['e'], child['w_indices'], self.arch_params['width_multiplier_choices'], self.arch_params)
                    if macs <= constraint:
                        next_population.append((self.accuracy_predictor(child), child))
                mutation_attempts += 1 
                        
            population = next_population
        return sorted(population, key=lambda x: x[0], reverse=True)[0][1]

# ==========================================================================================
# --- 3. Main Workflow for a SINGLE Model with TIMING ---
# ==========================================================================================

def process_single_model(model_path, device):
    model_name = os.path.basename(model_path).replace('.pt', '')
    print(f"\n{'#'*60}\n# Processing Model: {model_name}\n{'#'*60}")
    
    server_model, arch_params = load_supernet_and_config(model_path, device)
    if server_model is None: return

    val_loader, test_loader = get_data_loaders(DATASET_NAME, DATA_PATH, BATCH_SIZE)
    
    # Construct full paths using MAIN_DIR and model_name
    OUTPUT_MODEL_DIR = os.path.join(MAIN_DIR, model_name)
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True) # Ensure model-specific directory exists
    
    predictor_dataset_path = os.path.join(OUTPUT_MODEL_DIR, f"{model_name}_{DATASET_NAME}_{PREDICTOR_DATASET_FNAME}")
    predictor_model_path = os.path.join(OUTPUT_MODEL_DIR, f"{model_name}_{DATASET_NAME}_{PREDICTOR_MODEL_FNAME}")
    accuracy_output_path = os.path.join(OUTPUT_MODEL_DIR, f"baseline_results_{model_name}_{DATASET_NAME}.csv")
    timing_output_path = os.path.join(OUTPUT_MODEL_DIR, f"timing_results_{model_name}.csv")
    subnet_details_output_path = os.path.join(OUTPUT_MODEL_DIR, f"subnet_details_{model_name}_{DATASET_NAME}.csv")


    time_s1 = 0
    df_predictor = None
    predictor_data = [] # Initialize here to accumulate new samples or load existing
    
    print(f"\n--- Step 1: Generating Predictor Dataset (saving to {predictor_dataset_path}) ---")
    if os.path.exists(predictor_dataset_path):
        # Load existing data
        df_existing = pd.read_csv(predictor_dataset_path)
        predictor_data = df_existing.to_dict('records')
        print(f"  Loaded {len(predictor_data)} existing samples from {predictor_dataset_path}.")

        if len(predictor_data) >= NUM_SAMPLES_FOR_PREDICTOR:
            print(f"  Dataset already contains {len(predictor_data)} samples, which is >= required {NUM_SAMPLES_FOR_PREDICTOR}. SKIPPING further data generation.")
            df_predictor = df_existing # Use the loaded DataFrame
        else:
            print(f"  Continuing to generate {NUM_SAMPLES_FOR_PREDICTOR - len(predictor_data)} more samples.")
            start_index = len(predictor_data)
            num_to_generate = NUM_SAMPLES_FOR_PREDICTOR - start_index
            
            start_time_s1 = time.time()
            with tqdm(total=num_to_generate, initial=0, desc="  Sampling Architectures (Cont.)") as pbar:
                for i in range(num_to_generate):
                    arch = server_model.random_subnet_arch()
                    static_subnet_client = server_model.get_subnet(**arch)
                    static_subnet = static_subnet_client.model
                    val_accuracy = evaluate_accuracy(static_subnet, val_loader, device)
                    feature_vector = arch_to_feature_vector(arch, arch_params)
                    row = {f'feature_{j}': val for j, val in enumerate(feature_vector)}
                    row['accuracy'] = val_accuracy
                    predictor_data.append(row)
                    pbar.update(1) 
                    
                    if (start_index + i + 1) % CHECKPOINT_INTERVAL == 0 or (start_index + i + 1) == NUM_SAMPLES_FOR_PREDICTOR:
                        df_predictor_current = pd.DataFrame(predictor_data)
                        df_predictor_current.to_csv(predictor_dataset_path, index=False)
                        print(f"\n  Checkpoint: Saved {len(predictor_data)} samples to {predictor_dataset_path}")
            
            time_s1 = time.time() - start_time_s1
            df_predictor = pd.DataFrame(predictor_data) # Final DataFrame
            df_predictor.to_csv(predictor_dataset_path, index=False) # Final save
            print(f"Step 1 (continuation) completed in {time_s1:.2f} seconds. Total samples: {len(df_predictor)}.")
            
    else: # No existing file, generate from scratch
        print(f"  No existing dataset found. Generating {NUM_SAMPLES_FOR_PREDICTOR} samples from scratch.")
        start_time_s1 = time.time()
        with tqdm(total=NUM_SAMPLES_FOR_PREDICTOR, initial=0, desc="  Sampling Architectures") as pbar:
            for i in range(NUM_SAMPLES_FOR_PREDICTOR):
                arch = server_model.random_subnet_arch()
                static_subnet_client = server_model.get_subnet(**arch)
                static_subnet = static_subnet_client.model
                val_accuracy = evaluate_accuracy(static_subnet, val_loader, device)
                feature_vector = arch_to_feature_vector(arch, arch_params)
                row = {f'feature_{j}': val for j, val in enumerate(feature_vector)}
                row['accuracy'] = val_accuracy
                predictor_data.append(row)
                pbar.update(1) 
                
                if (i + 1) % CHECKPOINT_INTERVAL == 0 or (i + 1) == NUM_SAMPLES_FOR_PREDICTOR:
                    df_predictor_current = pd.DataFrame(predictor_data)
                    df_predictor_current.to_csv(predictor_dataset_path, index=False)
                    print(f"\n  Checkpoint: Saved {len(predictor_data)} samples to {predictor_dataset_path}")

        time_s1 = time.time() - start_time_s1
        df_predictor = pd.DataFrame(predictor_data) # Final DataFrame
        df_predictor.to_csv(predictor_dataset_path, index=False) # Final save
        print(f"Step 1 completed in {time_s1:.2f} seconds. Total samples: {len(df_predictor)}.")


    feature_len = len([c for c in df_predictor.columns if 'feature_' in c])
    
    time_s2 = 0
    predictor = AccuracyPredictor(input_features=feature_len).to(device)
    if not os.path.exists(predictor_model_path):
        start_time_s2 = time.time()
        print(f"\n--- Step 2: Training Accuracy Predictor (saving to {predictor_model_path}) ---")
        dataset = TensorDataset(torch.tensor(df_predictor.filter(like='feature_').values, dtype=torch.float32),
                                torch.tensor(df_predictor['accuracy'].values, dtype=torch.float32).unsqueeze(1))
        val_size = int(0.1 * len(dataset)); train_size = len(dataset) - val_size
        train_dataset, _ = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=PREDICTOR_BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(predictor.parameters(), lr=PREDICTOR_LR, weight_decay=1e-5)
        criterion = nn.MSELoss()
        for _ in tqdm(range(PREDICTOR_EPOCHS), desc="  Training Predictor"):
            predictor.train()
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = predictor(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        torch.save(predictor.state_dict(), predictor_model_path)
        time_s2 = time.time() - start_time_s2
        print(f"Step 2 completed in {time_s2:.2f} seconds.")
    else:
        print(f"\n--- Step 2: SKIPPED (Loading existing predictor model from {predictor_model_path}) ---")
        predictor.load_state_dict(torch.load(predictor_model_path, map_location=device))
    predictor.eval()

    print("\n--- Step 3: Running Predictor-Guided Search ---")
    finder_args = argparse.Namespace(population_size=EVO_POPULATION_SIZE, generations=EVO_GENERATIONS, parent_size=int(EVO_POPULATION_SIZE * EVO_PARENT_RATIO), mutation_ratio=EVO_MUTATION_RATIO)
    finder = EvolutionFinder(server_model, arch_params, predictor, finder_args)
    
    results_data = [] # For mean_accuracy, std_dev_accuracy per target/bin
    subnet_details_data = [] # For individual subnet details: MACs, Params, Accuracy, Arch

    search_times_per_run_for_all_constraints = [] # Collect all search times

    # --- Determine MACs targets based on selected mode ---
    macs_constraints_info = [] # List of (display_name, actual_constraint_value)
    if MACS_TARGET_MODE == 'linspace':
        macs_targets_linspace = np.linspace(MACS_START, MACS_END, NUM_MACS_TARGETS)
        for macs_target in macs_targets_linspace:
            macs_constraints_info.append((f"MACs_{macs_target/1e9:.2f}B", macs_target))
    elif MACS_TARGET_MODE == 'binned':
        for bin_name, (macs_lower, macs_upper) in MACS_BINS.items():
            macs_constraints_info.append((bin_name, macs_upper))
    else:
        raise ValueError(f"Invalid MACS_TARGET_MODE: {MACS_TARGET_MODE}. Choose 'linspace' or 'binned'.")

    for constraint_name, macs_constraint_value in macs_constraints_info:
        print(f"\n--- Searching for Target: {constraint_name} (Constraint: {macs_constraint_value/1e9:.2f}B) ---")
        constraint_accuracies = []
        
        for run_idx in range(NUM_SEARCH_RUNS):
            run_seed = BASE_SEED + run_idx
            print(f"  Run {run_idx+1}/{NUM_SEARCH_RUNS} (Seed: {run_seed})")
            
            start_search_time = time.time()
            best_arch = finder.run_evolution_search(macs_constraint_value, run_seed)
            search_time_for_run = time.time() - start_search_time
            search_times_per_run_for_all_constraints.append(search_time_for_run)
            print(f"    Search for run {run_idx+1} took {search_time_for_run:.2f} seconds.")
            
            static_subnet_client = server_model.get_subnet(**best_arch)
            static_subnet = static_subnet_client.model
            
            num_params = count_parameters(static_subnet)
            
            test_accuracy = evaluate_accuracy(static_subnet, test_loader, device)
            
            actual_macs, _ = subnet_macs(best_arch['d'], best_arch['e'], best_arch['w_indices'], arch_params['width_multiplier_choices'], arch_params)

            print(f"    Found Arch Test Accuracy: {test_accuracy:.2f}% (Actual MACs: {actual_macs/1e9:.2f}B, Params: {num_params/1e6:.2f}M)")
            constraint_accuracies.append(test_accuracy)
            
            subnet_details_data.append({
                'model_name': model_name,
                'macs_constraint_name': constraint_name, # Use the descriptive name here
                'macs_constraint_value': macs_constraint_value,
                'run_seed': run_seed,
                'actual_macs': actual_macs,
                'num_parameters': num_params,
                'test_accuracy': test_accuracy,
                'arch_d': str(best_arch['d']),
                'arch_e': str(best_arch['e']),
                'arch_w_indices': str(best_arch['w_indices'])
            })
            
        mean_acc, std_acc = np.mean(constraint_accuracies), np.std(constraint_accuracies)
        print(f"  Target '{constraint_name}' Summary: Accuracy = {mean_acc:.2f} ± {std_acc:.2f}")
        results_data.append({'macs_constraint_name': constraint_name, 'macs_constraint_value': macs_constraint_value, 'mean_accuracy': mean_acc, 'std_dev_accuracy': std_acc})
    
    avg_search_time_all_constraints = np.mean(search_times_per_run_for_all_constraints) if search_times_per_run_for_all_constraints else 0
    print(f"\nAverage search time per single evolutionary run across all MACs constraints: {avg_search_time_all_constraints:.2f} seconds.")

    df_results = pd.DataFrame(results_data)
    df_results.to_csv(accuracy_output_path, index=False)
    print(f"\nAggregated accuracy results for {model_name} saved to: {accuracy_output_path}")
    
    df_subnet_details = pd.DataFrame(subnet_details_data)
    df_subnet_details.to_csv(subnet_details_output_path, index=False)
    print(f"Detailed subnet information for {model_name} saved to: {subnet_details_output_path}")

    timing_df = pd.DataFrame([{'model_name': model_name, 'data_generation_time_s': time_s1, 'predictor_training_time_s': time_s2, 'avg_search_time_per_run_s': avg_search_time_all_constraints}])
    timing_df.to_csv(timing_output_path, index=False)
    print(f"Timing results for {model_name} saved to: {timing_output_path}")

# ==========================================================================================
# --- 4. Main Execution Loop ---
# ==========================================================================================

if __name__ == "__main__":
    print(f"--- Phase 2: SuperFedNAS Baseline Search and Evaluation (Batch Mode, MACs Mode: {MACS_TARGET_MODE}) ---")
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    for model_path in MODEL_PATHS:
        process_single_model(model_path, device)
    print(f"\n{'='*30}\nAll baseline models processed!\n{'='*30}")