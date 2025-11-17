import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # No longer strictly needed for scaler, but kept for potential future use or consistency

# --- Path Setup ---
# Adjust this path to ensure all necessary modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

# --- Import Your Project's Modules ---
try:
    # Assuming GenericServerOFA and its associated SuperNet class are here
    from deepfednas.Server.generic_server_model import GenericServerOFA
    # subnet_macs needs to be imported
    from deepfednas.utils.subnet_cost import subnet_macs
    # If decode_chromosome is in a specific utility file, import it.
    # Otherwise, it's defined here for now.
    # from fedml_api.standalone.superfednas.utils.your_utils_file import decode_chromosome
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Please check your sys.path and module names. {e}")
    print("Ensure fedml_api.standalone.superfednas.Server.ServerModel.generic_server_model "
          "and fedml_api.standalone.superfednas.utils.subnet_macs are accessible.")
    sys.exit(1)


# ==========================================================================================
# --- CONFIGURATION (User customizable) ---
# ==========================================================================================
class Config:
    # CRITICAL CUSTOMIZATION 1: Path to your trained DeepFedNAS SuperNet checkpoint
    SUPERNET_PATH = "/home/bostan/projects/superfednas/fedml_experiments/standalone/superfednas/trained_models/4-stage_continued_cached_60_subnets_p1024_g1024_w-fine-grained.pt" 
    
    # CRITICAL CUSTOMIZATION 2: Output folder for dataset and LPM model
    OUTPUT_FOLDER = "/home/bostan/projects/superfednas/fedml_experiments/standalone/superfednas/evaluation/latency_prediction/datasets_and_predictor_models_cuda_bs-1" 

    # CRITICAL CUSTOMIZATION 3: Target device for latency measurement
    DEVICE_TYPE = "cpu" # "cpu" or "cuda:0" etc.

    # Data Collection Parameters
    TOTAL_SAMPLES_TO_COLLECT = 5000 # Number of unique subnets to sample for latency data
    LATENCY_MEASUREMENT_BATCH_SIZE = 1 # Batch size for running inference on subnets
    LATENCY_MEASUREMENT_WARMUP_RUNS = 5 # Warmup runs before actual measurement
    LATENCY_MEASUREMENT_AVG_RUNS = 20 # Number of runs to average for latency
    DATASET_SAVE_INTERVAL = 100 # Save CSV every X samples during collection
    SEED = 42 # Random seed for reproducibility

    # Input image dimensions for the SuperNet (e.g., CIFAR-10/100 -> 32, ImageNet -> 224)
    INPUT_IMAGE_SIZE = 32 
    INPUT_IMAGE_CHANNELS = 3

    # LPM Training Parameters
    LPM_HIDDEN_LAYERS = [128, 64] # Architecture of the MLP
    LPM_LEARNING_RATE = 0.0001
    LPM_EPOCHS = 330
    LPM_BATCH_SIZE = 32
    LPM_VALIDATION_SPLIT = 0.2 # Fraction of data to use for validation
    
    # File names
    DATASET_FILENAME = f"latency_dataset_{DEVICE_TYPE.replace(':', '_')}.csv"
    LPM_MODEL_FILENAME = f"lpm_model_and_scaler_{DEVICE_TYPE.replace(':', '_')}.pth" # Changed filename
    # SCALER_FILENAME is no longer directly used as a separate file

# ==========================================================================================
# --- UTILITY FUNCTIONS (Adapted from your project) ---
# ==========================================================================================

# make_divisible is needed to configure channels in the subnet model.
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None: min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v: new_v += divisor
    return int(new_v)

# decode_chromosome is needed to interpret the sampled architecture.
# CRITICAL: Ensure this matches the version used in your GA and SuperNet.
def decode_chromosome(chrom, num_depth_genes, num_exp_genes, num_width_idx_genes, exp_opt_values_list):
    depth_genes_end = num_depth_genes
    exp_genes_end = num_depth_genes + num_exp_genes
    d_vec = chrom[:depth_genes_end].astype(int).tolist()
    e_indices = chrom[depth_genes_end:exp_genes_end].astype(int)
    w_indices = chrom[exp_genes_end:exp_genes_end + num_width_idx_genes].astype(int).tolist()
    
    # Clip e_indices to prevent out-of-bounds errors if random sampling generates invalid indices
    e_indices = np.clip(e_indices, 0, len(exp_opt_values_list) - 1)
    
    e_vec_values = [exp_opt_values_list[i] for i in e_indices]
    return {"d": d_vec, "e": e_vec_values, "w_indices": w_indices}

# This function loads your SuperNet checkpoint and extracts the model and arch_params dict
def load_supernet_and_config(ckpt_path, device_map_location):
    print(f"Loading supernet and config from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint file not found at '{ckpt_path}'.")
        return None, None
    
    # Use weights_only=False due to numpy objects in checkpoint, as discussed
    checkpoint = torch.load(ckpt_path, map_location=device_map_location, weights_only=False)
    
    if "arch_params" not in checkpoint:
        print(f"ERROR: 'arch_params' dictionary not found in '{ckpt_path}'.")
        return None, None
    if "params" not in checkpoint:
        print(f"ERROR: 'params' (model state_dict) not found in '{ckpt_path}'.")
        return None, None

    arch_params = checkpoint["arch_params"]
    # Ensure original_stage_base_channels is a numpy array if GenericServerOFA expects it
    if not isinstance(arch_params.get('original_stage_base_channels'), np.ndarray):
        arch_params['original_stage_base_channels'] = np.array(arch_params['original_stage_base_channels'])
    
    # CRITICAL CUSTOMIZATION 4: Instantiate your GenericServerOFA (SuperNet wrapper)
    # Ensure GenericServerOFA's constructor matches these arguments
    model_wrapper = GenericServerOFA(
        arch_params=arch_params, sampling_method='all_random', num_cli_total=1 
        # Add any other necessary arguments for GenericServerOFA initialization
    )
    model_wrapper.set_model_params(checkpoint["params"]) # Load the state_dict into the SuperNet

    print("Supernet and config loaded successfully.")
    return model_wrapper.model, arch_params # model_wrapper.model is the actual SuperNet (nn.Module), arch_params is the dict

# ==========================================================================================
# --- LATENCY PREDICTOR MLP DEFINITION ---
# ==========================================================================================
class LatencyPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(LatencyPredictorMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1)) # Output is a single latency value
        layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    # Removed the 'predict' method here because it will be part of a separate helper
    # that handles scaling *before* calling forward. This keeps the nn.Module pure.

# ==========================================================================================
# --- ARCHITECTURE FEATURE EXTRACTION ---
# ==========================================================================================
# This function converts a decoded architecture into a flat feature vector for the MLP
# CRITICAL CUSTOMIZATION 5: Adjust feature extraction based on your `arch_params` structure
def arch_to_feature_vector_and_names(arch, supernet_arch_params):
    features = []
    feature_names = []

    # Depth genes (arch["d"] is a list of depths for each stage)
    for i, d in enumerate(arch["d"]):
        features.append(d)
        feature_names.append(f"depth_stage{i}")

    # Expansion ratio genes (arch["e"] is a list of actual expansion ratio values)
    # This assumes a consistent number of blocks per stage for feature vector creation.
    # If blocks per stage vary, you might need a more sophisticated mapping or pad.
    # For simplicity, we are taking one long list of exp ratios.
    for i, e_ratio in enumerate(arch["e"]):
        features.append(e_ratio)
        feature_names.append(f"exp_ratio_block{i}") 

    # Width multiplier indices (arch["w_indices"] is a list of width multiplier indices)
    # We use the actual multiplier value for the feature.
    width_mult_options = supernet_arch_params['width_multiplier_choices']
    for i, w_idx in enumerate(arch["w_indices"]):
        features.append(width_mult_options[w_idx]) 
        feature_names.append(f"width_mult_stage{i}") 

    # Add MACs and Parameters as features (they are strong indicators for latency)
    macs, params = subnet_macs(
        arch["d"], arch["e"], arch["w_indices"], 
        width_mult_options, supernet_arch_params
    )
    features.append(macs)
    feature_names.append("macs")
    features.append(params)
    feature_names.append("params")

    return np.array(features, dtype=np.float32), feature_names

# ==========================================================================================
# --- LATENCY MEASUREMENT FUNCTION ---
# ==========================================================================================
def measure_subnet_latency(subnet_model, input_hw, input_channels, measurement_batch_size, 
                           warmup_runs, avg_runs, device):
    subnet_model.eval()
    subnet_model.to(device)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(measurement_batch_size, input_channels, input_hw, input_hw).to(device)

    # Warm-up runs
    for _ in range(warmup_runs):
        _ = subnet_model(dummy_input)
    
    # Measure
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros(avg_runs)
    
    with torch.no_grad():
        for rep in range(avg_runs):
            if 'cuda' in str(device):
                starter.record()
                _ = subnet_model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                timings[rep] = starter.elapsed_time(ender) # milliseconds
            else: # CPU timing
                start_time = time.perf_counter()
                _ = subnet_model(dummy_input)
                end_time = time.perf_counter()
                timings[rep] = (end_time - start_time) * 1000 # milliseconds

    mean_latency = np.mean(timings)
    return mean_latency

# ==========================================================================================
# --- MAIN PIPELINE ---
# ==========================================================================================
def main():
    config = Config()

    # --- Setup Output Folder ---
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    dataset_path = os.path.join(config.OUTPUT_FOLDER, config.DATASET_FILENAME)
    lpm_model_and_scaler_path = os.path.join(config.OUTPUT_FOLDER, config.LPM_MODEL_FILENAME) # Use new combined name

    # --- Load SuperNet and Extract Parameters ---
    # Load SuperNet on CPU first to avoid device issues during parameter extraction
    supernet_instance, arch_params = load_supernet_and_config(config.SUPERNET_PATH, torch.device('cpu'))
    
    if supernet_instance is None or arch_params is None:
        print("Failed to load SuperNet or architecture parameters. Exiting.")
        sys.exit(1)

    # Extract relevant architectural parameters from the `arch_params` dictionary
    # CRITICAL CUSTOMIZATION 6: Verify these keys match your `arch_params` dict
    try:
        supernet_arch_params = {
            'num_stages': arch_params['num_stages'],
            'max_extra_blocks_per_stage': arch_params['max_extra_blocks_per_stage'],
            'original_stem_out_channels': arch_params['original_stem_out_channels'],
            'original_stage_base_channels': arch_params['original_stage_base_channels'],
            'channel_divisible_by': arch_params['channel_divisible_by'],
            'initial_input_hw': config.INPUT_IMAGE_SIZE, 
            'stem_stride': arch_params['stem_stride'],
            'stage_downsample_factors': arch_params['stage_downsample_factors'],
            'initial_input_channels': config.INPUT_IMAGE_CHANNELS,
            'width_multiplier_choices': arch_params['width_multiplier_choices'],
            'expansion_ratio_choices': arch_params['expansion_ratio_choices'],
            'n_classes': arch_params['n_classes'], # Added missing key
            # Add any other arch_config_params needed by subnet_macs or your GA's fitness function
        }
        width_mult_options = arch_params['width_multiplier_choices']
        depth_choices = np.array(list(range(arch_params['max_extra_blocks_per_stage'] + 1))) # Reconstruct depth choices
        exp_opt_values = arch_params['expansion_ratio_choices'] # These are actual values
    except KeyError as e:
        print(f"ERROR: Missing key in 'arch_params' dictionary: {e}")
        sys.exit(1)

    # Calculate chromosome dimensions for random sampling, based on the SuperNet's architecture
    num_depth_genes = supernet_arch_params['num_stages']
    max_blocks_total_per_stage_supernet = supernet_arch_params['max_extra_blocks_per_stage'] + 1
    num_exp_genes = supernet_arch_params['num_stages'] * max_blocks_total_per_stage_supernet
    num_width_idx_genes = supernet_arch_params['num_stages'] + 1 # Stem + num_stages (for width multipliers)
    # chromosome_len is not explicitly needed for sampling but good for sanity check
    # chromosome_len = num_depth_genes + num_exp_genes + num_width_idx_genes 
    
    target_device = torch.device(config.DEVICE_TYPE)

    # --- Latency Data Collection ---
    print(f"\n--- Starting Latency Data Collection for {config.DEVICE_TYPE} ---")
    latency_df = pd.DataFrame()
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}")
        latency_df = pd.read_csv(dataset_path)
        print(f"Found {len(latency_df)} existing samples.")
    
    current_samples = len(latency_df)
    rng = np.random.default_rng(config.SEED)

    if current_samples < config.TOTAL_SAMPLES_TO_COLLECT:
        print(f"Collecting {config.TOTAL_SAMPLES_TO_COLLECT - current_samples} more samples...")
        pbar = tqdm(total=config.TOTAL_SAMPLES_TO_COLLECT, initial=current_samples, desc="Collecting Latency Data", ncols=100)
        
        # Use a set to track feature vectors to prevent duplicate measurements
        # Convert existing dataframe rows to tuples for efficient lookup
        if len(latency_df) > 0:
            existing_feature_names = [col for col in latency_df.columns if col != 'latency_ms']
            existing_feature_vectors = set(
                tuple(row[existing_feature_names].values) 
                for idx, row in latency_df.iterrows()
            )
        else:
            existing_feature_vectors = set()

        while len(latency_df) < config.TOTAL_SAMPLES_TO_COLLECT:
            # 1. Sample a random chromosome
            depth_g = rng.choice(depth_choices, num_depth_genes)
            exp_g_indices = rng.integers(0, len(exp_opt_values), num_exp_genes)
            w_g_indices = rng.integers(0, len(width_mult_options), num_width_idx_genes)
            chromosome = np.concatenate([depth_g, exp_g_indices, w_g_indices])

            # 2. Decode to architecture dict
            arch = decode_chromosome(chromosome, num_depth_genes, num_exp_genes, num_width_idx_genes, exp_opt_values)

            # 3. Convert arch to feature vector and get names for uniqueness check
            features, feature_names = arch_to_feature_vector_and_names(arch, supernet_arch_params)
            features_tuple = tuple(features) # Convert to tuple for hashing

            # Check for uniqueness based on feature vector
            if features_tuple in existing_feature_vectors:
                continue # Skip if already collected

            # 4. Instantiate subnet from supernet_instance
            try:
                # set_active_subnet expects expansion *indices*, so convert arch['e'] (which are values)
                e_indices_for_supernet = [list(exp_opt_values).index(val) for val in arch['e']]
                
                supernet_instance.set_active_subnet(
                    d=arch['d'], 
                    e_indices=e_indices_for_supernet, 
                    w_indices=arch['w_indices']
                )
                subnet = supernet_instance.get_active_subnet(preserve_weight=True)
            except Exception as e:
                # print(f"Warning: Failed to get subnet for arch {arch}. Skipping. Error: {e}")
                continue # Skip if subnet creation fails

            # 5. Measure latency
            latency_ms = measure_subnet_latency(
                subnet, config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_CHANNELS, 
                config.LATENCY_MEASUREMENT_BATCH_SIZE, config.LATENCY_MEASUREMENT_WARMUP_RUNS, 
                config.LATENCY_MEASUREMENT_AVG_RUNS, target_device
            )

            # 6. Add to DataFrame
            current_row_data = dict(zip(feature_names, features))
            current_row_data['latency_ms'] = latency_ms
            latency_df = pd.concat([latency_df, pd.DataFrame([current_row_data])], ignore_index=True)
            existing_feature_vectors.add(features_tuple) # Add to set of collected features

            pbar.update(1)
            if len(latency_df) % config.DATASET_SAVE_INTERVAL == 0:
                latency_df.to_csv(dataset_path, index=False)
                
        pbar.close()
        latency_df.to_csv(dataset_path, index=False)
        print(f"Finished collecting {len(latency_df)} samples. Saved to {dataset_path}")

    else:
        print(f"Dataset already contains {len(latency_df)} samples ({dataset_path}). Skipping collection.")

    # --- Latency Predictor MLP Training ---
    print("\n--- Starting Latency Predictor MLP Training ---")
    if len(latency_df) < 10: # Minimum samples for training
        print(f"ERROR: Not enough latency data collected ({len(latency_df)} samples). Cannot train MLP.")
        sys.exit(1)

    X = latency_df.drop('latency_ms', axis=1)
    y = latency_df['latency_ms']

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    # joblib.dump(scaler, scaler_path) # No longer saving scaler separately
    print("Feature scaler fitted.")

    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=config.LPM_VALIDATION_SPLIT, random_state=config.SEED
    )

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    lpm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatencyPredictorMLP(X_train.shape[1], config.LPM_HIDDEN_LAYERS).to(lpm_device)
    optimizer = optim.Adam(model.parameters(), lr=config.LPM_LEARNING_RATE)
    criterion = nn.L1Loss()

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.LPM_BATCH_SIZE, shuffle=True)

    print(f"Training LPM on {lpm_device}...")
    for epoch in range(config.LPM_EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(lpm_device), batch_y.to(lpm_device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t.to(lpm_device))
            test_loss = criterion(test_outputs, y_test_t.to(lpm_device))
        
        if (epoch + 1) % 10 == 0 or epoch == 1: # Print at epoch 1 and every 10 epochs
            print(f"Epoch {epoch+1}/{config.LPM_EPOCHS}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    # --- Save Trained LPM and Scaler ---
    # Saving a dictionary containing model state, scaler, and model config for easy loading
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X_train.shape[1],
        'hidden_layers': config.LPM_HIDDEN_LAYERS,
        'scaler': scaler,
    }, lpm_model_and_scaler_path) # Use the combined filename
    print(f"\nLatency Predictor MLP and Scaler saved to: {lpm_model_and_scaler_path}")

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t.to(lpm_device)).cpu().squeeze().numpy()
        test_actual = y_test_t.cpu().squeeze().numpy()
        mae = np.mean(np.abs(test_pred - test_actual))
        rmse = np.sqrt(np.mean((test_pred - test_actual)**2))
        print(f"Final MAE on test set: {mae:.4f} ms")
        print(f"Final RMSE on test set: {rmse:.4f} ms")

if __name__ == "__main__":
    main()