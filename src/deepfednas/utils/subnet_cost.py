# subnet_macs.py (Updated for Parameterized Architecture, Flexible Downsampling, and Parameter Counting)

import numpy as np

# ─────────────────────────── HELPERS ─────────────────────────────────────
def make_divisible(v, divisor=8, min_value=None):
    """
    Ensures that channel counts are divisible by `divisor`.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v: # Prevent rounding down by more than 10%
        new_v += divisor
    return int(new_v)

# ─────────────────────────── MACS & PARAMS CALCULATION CORE ───────────────────────

# --- MACs Helpers ---
def _stem_macs(scaled_stem_out_channels, input_hw=32, kernel_size=3, initial_input_channels=3):
    return initial_input_channels * scaled_stem_out_channels * kernel_size * kernel_size * input_hw * input_hw

def _projection_macs(cin_proj, cout_proj, h_out):
    return cin_proj * cout_proj * 1 * 1 * h_out * h_out

# --- Parameter Helpers ---
def _stem_params(scaled_stem_out_channels, kernel_size=3, initial_input_channels=3):
    conv_weights = initial_input_channels * scaled_stem_out_channels * kernel_size * kernel_size
    bn_params = 2 * scaled_stem_out_channels # (weight + bias for BN)
    return conv_weights + bn_params

def _block_conv_params(cin_conv, cout_conv, kernel_size=3):
    conv_weights = cin_conv * cout_conv * kernel_size * kernel_size
    bn_params = 2 * cout_conv # (weight + bias for BN)
    return conv_weights + bn_params

def _projection_params(cin_proj, cout_proj):
    # Assuming 1x1 convolution for projection
    conv_weights = cin_proj * cout_proj * 1 * 1
    bn_params = 2 * cout_proj # (weight + bias for BN)
    return conv_weights + bn_params

def _fc_params(cin_fc, num_classes):
    fc_weights = cin_fc * num_classes
    fc_bias_params = num_classes
    return fc_weights + fc_bias_params


def subnet_macs( # Renamed function for clarity
    depth_vec, exp_vec, w_indices,
    width_mult_options,
    arch_config_params
):
    """
    MAC and Parameter estimator for OFA-ResNet-like supernets with per-stage width multipliers and configurable stages.
    Returns: (total_macs, total_params)
    """
    num_stages = arch_config_params['num_stages']
    original_stem_out_channels = arch_config_params['original_stem_out_channels']
    original_stage_base_channels = arch_config_params['original_stage_base_channels']
    initial_input_hw = arch_config_params['initial_input_hw']
    initial_input_channels = arch_config_params.get('initial_input_channels', 3)
    stage_downsample_factors = arch_config_params['stage_downsample_factors']
    max_extra_blocks_per_stage = arch_config_params['max_extra_blocks_per_stage']
    channel_divisible_by = arch_config_params['channel_divisible_by']
    num_classes = arch_config_params['n_classes']

    # Input validations (as before)
    if len(depth_vec) != num_stages:
        raise ValueError(f"depth_vec length ({len(depth_vec)}) must match num_stages ({num_stages}).")
    if len(w_indices) != num_stages + 1:
        raise ValueError(f"w_indices length ({len(w_indices)}) must be num_stages ({num_stages}) + 1 (for stem).")
    if len(original_stage_base_channels) != num_stages:
        raise ValueError(f"original_stage_base_channels length ({len(original_stage_base_channels)}) must match num_stages ({num_stages}).")
    if len(stage_downsample_factors) != num_stages:
        raise ValueError(f"stage_downsample_factors length ({len(stage_downsample_factors)}) must match num_stages ({num_stages}).")
    for w_idx in w_indices:
        if w_idx < 0 or w_idx >= len(width_mult_options):
            raise ValueError(f"Invalid w_idx {w_idx} for width_mult_options of length {len(width_mult_options)}")
    max_blocks_total_per_stage_supernet = max_extra_blocks_per_stage + 1
    expected_exp_vec_len = num_stages * max_blocks_total_per_stage_supernet
    if len(exp_vec) != expected_exp_vec_len:
        raise ValueError(f"exp_vec length ({len(exp_vec)}) is inconsistent. Expected {expected_exp_vec_len}.")

    stem_width_multiplier = width_mult_options[w_indices[0]]
    stage_width_multipliers = [width_mult_options[w_indices[i+1]] for i in range(num_stages)]

    scaled_stem_cout = make_divisible(original_stem_out_channels * stem_width_multiplier, divisor=channel_divisible_by)
    if scaled_stem_cout < channel_divisible_by: scaled_stem_cout = channel_divisible_by

    scaled_stage_base_channels = []
    for i in range(num_stages):
        ch = make_divisible(original_stage_base_channels[i] * stage_width_multipliers[i], divisor=channel_divisible_by)
        if ch < channel_divisible_by: ch = channel_divisible_by
        scaled_stage_base_channels.append(ch)

    # Initialize MACs and Parameters
    total_macs = _stem_macs(scaled_stem_cout, input_hw=initial_input_hw, initial_input_channels=initial_input_channels)
    total_params = _stem_params(scaled_stem_cout, initial_input_channels=initial_input_channels)

    current_cin = scaled_stem_cout
    stem_stride = arch_config_params.get('stem_stride', 1)
    current_stage_input_hw = initial_input_hw // stem_stride

    for stage_idx in range(num_stages):
        num_extra_blocks = depth_vec[stage_idx]
        num_blocks_in_stage_active = 1 + num_extra_blocks

        if num_blocks_in_stage_active > max_blocks_total_per_stage_supernet:
             raise ValueError(f"Stage {stage_idx} implies {num_blocks_in_stage_active} active blocks, "
                             f"but supernet structure allows max {max_blocks_total_per_stage_supernet} blocks/stage.")

        stage_target_cout = scaled_stage_base_channels[stage_idx]
        exp_vec_stage_start_idx = stage_idx * max_blocks_total_per_stage_supernet
        downsample_factor = stage_downsample_factors[stage_idx]

        for block_num in range(num_blocks_in_stage_active):
            block_output_hw = current_stage_input_hw
            
            if block_num == 0 and downsample_factor > 1:
                block_output_hw = block_output_hw // downsample_factor

            current_exp_ratio = exp_vec[exp_vec_stage_start_idx + block_num]
            cmid = make_divisible(stage_target_cout * current_exp_ratio, divisor=channel_divisible_by)
            if cmid < channel_divisible_by and stage_target_cout > 0:
                cmid = channel_divisible_by

            # Conv1
            if current_cin > 0 and cmid > 0:
                 total_macs += current_cin * cmid * 3*3 * block_output_hw * block_output_hw
                 total_params += _block_conv_params(current_cin, cmid, kernel_size=3)
            # Conv2
            if cmid > 0 and stage_target_cout > 0:
                 total_macs += cmid * stage_target_cout * 3*3 * block_output_hw * block_output_hw
                 total_params += _block_conv_params(cmid, stage_target_cout, kernel_size=3)

            # Projection Shortcut
            if block_num == 0:
                if current_cin > 0 and stage_target_cout > 0 : # Check if projection is active/needed
                    # MACs for projection
                    total_macs += _projection_macs(current_cin, stage_target_cout, block_output_hw)
                    # Params for projection (1x1 conv + BN)
                    total_params += _projection_params(current_cin, stage_target_cout)

            current_cin = stage_target_cout
            current_stage_input_hw = block_output_hw

    # Classifier FC Layer
    if current_cin > 0 and num_classes > 0:
        total_macs += current_cin * num_classes # MACs for FC
        total_params += _fc_params(current_cin, num_classes) # Params for FC

    return int(total_macs), int(total_params)

# ------------ EXAMPLE USAGE (for testing this script) ----------------
if __name__ == '__main__':
    print("Testing subnet_macs_params.py with Parameterized Architecture...")

    arch_config_8_stages = {
        'num_stages': 8,
        'initial_input_channels': 3,
        'original_stem_out_channels': 32,
        'original_stage_base_channels': np.array([40, 104, 112, 152, 168, 176, 248, 256]),
        'initial_input_hw': 32,
        'stem_stride': 1,
        'stage_downsample_factors': [1, 1, 1, 1, 1, 2, 2, 2],
        'max_extra_blocks_per_stage': 7,
        'channel_divisible_by': 8,
        'n_classes': 10
    }
    width_opts_test = [1.0]

    max_blocks_total_stage = arch_config_8_stages['max_extra_blocks_per_stage'] + 1
    exp_vec_len_8_stages = arch_config_8_stages['num_stages'] * max_blocks_total_stage

    depth_vec_min = [0] * arch_config_8_stages['num_stages']
    exp_vec_min = [0.10] * exp_vec_len_8_stages
    w_indices_min = [0] * (arch_config_8_stages['num_stages'] + 1)

    macs_min, params_min = subnet_macs(depth_vec_min, exp_vec_min, w_indices_min, width_opts_test, arch_config_8_stages)
    print(f"Min Arch (8 stages, 1 block/stage, 0.1exp, 0.1width): {macs_min/1e6:.2f} M MACs, {params_min/1e6:.2f} M Params")

    depth_vec_max = [arch_config_8_stages['max_extra_blocks_per_stage']] * arch_config_8_stages['num_stages']
    exp_vec_max = [0.25] * exp_vec_len_8_stages # Using 1.0 for max expansion as per some OFA configs; adjust if 0.25 is strict max
    w_indices_max = [len(width_opts_test)-1] * (arch_config_8_stages['num_stages'] + 1)

    macs_max, params_max = subnet_macs(depth_vec_max, exp_vec_max, w_indices_max, width_opts_test, arch_config_8_stages)
    print(f"Max Arch (8 stages, {max_blocks_total_stage} blocks/stage, 1.0exp, 1.0width): {macs_max/1e6:.2f} M MACs, {params_max/1e6:.2f} M Params")

    arch_config_4_stages = {
        'num_stages': 4,
        'initial_input_channels': 3,
        'original_stem_out_channels': 64,
        'original_stage_base_channels': np.array([256, 512, 1024, 2048]), # Typical ResNet-50 stage output features before width mult
        'initial_input_hw': 32, # For CIFAR
        'stem_stride': 1, # OFA ResNet for CIFAR often uses stride 1 in stem
        'stage_downsample_factors': [1, 2, 2, 2], # Stage1 no downsample, Stage2,3,4 downsample
        'max_extra_blocks_per_stage': 2, # Max (2+1)=3 blocks per stage (like original file)
        'channel_divisible_by': 8,
        'n_classes': 10
    }
    max_blocks_total_stage_4 = arch_config_4_stages['max_extra_blocks_per_stage'] + 1
    exp_vec_len_4_stages = arch_config_4_stages['num_stages'] * max_blocks_total_stage_4


    depth_vec_min_4s = [0] * arch_config_4_stages['num_stages'] # Min depth (1 block per stage)
    exp_vec_min_4s = [0.10] * exp_vec_len_4_stages # Min expansion
    w_indices_min_4s = [0] * (arch_config_4_stages['num_stages'] + 1) # Min width multiplier index

    macs_min_4s, params_min_4s = subnet_macs(depth_vec_min_4s, exp_vec_min_4s, w_indices_min_4s, width_opts_test, arch_config_4_stages)
    print(f"Min Arch (4 stages, 1 block/stage, 0.1exp, 0.1width): {macs_min_4s/1e6:.2f} M MACs, {params_min_4s/1e6:.2f} M Params")

    depth_vec_max_4s = [arch_config_4_stages['max_extra_blocks_per_stage']] * arch_config_4_stages['num_stages'] # Max depth
    exp_vec_max_4s = [0.25] * exp_vec_len_4_stages # Max expansion (as per previous example values)
    w_indices_max_4s = [len(width_opts_test)-1] * (arch_config_4_stages['num_stages'] + 1) # Max width multiplier index

    macs_max_4s, params_max_4s = subnet_macs(depth_vec_max_4s, exp_vec_max_4s, w_indices_max_4s, width_opts_test, arch_config_4_stages)
    print(f"Max Arch (4 stages, {max_blocks_total_stage_4} blocks/stage, 0.25exp, 1.0width): {macs_max_4s/1e6:.2f} M MACs, {params_max_4s/1e6:.2f} M Params")

    # Example with specific architecture for 4 stages
    # Based on the original example from entropy_maximizer.py before effectiveness
    # d_vec = [0,0,0,0] (1 block per stage), e_vec = [0.1]*12, w_indices = [0,0,0,0,0] (0.1 width_mult)
    specific_d_vec = [2,2,2,2] # 1 block per stage (0 extra blocks)
    # For 4 stages, max_extra_blocks_per_stage = 2 => max_blocks_total_per_stage = 3
    # exp_vec length should be 4 stages * 3 blocks/stage = 12
    specific_e_vec = [0.25, 0.25, 0.25, # Stage 1 (only first 0.1 used if d[0]=0)
                      0.25, 0.25, 0.25, # Stage 2 (only first 0.1 used if d[1]=0)
                      0.25, 0.25, 0.25, # Stage 3 (only first 0.1 used if d[2]=0)
                      0.25, 0.25, 0.25] # Stage 4 (only first 0.1 used if d[3]=0)
    specific_w_indices = [0,0,0,0,0] # index 0 for stem, indices 1-4 for stages
    
    macs_specific, params_specific = subnet_macs(specific_d_vec, specific_e_vec, specific_w_indices, width_opts_test, arch_config_4_stages)
    print(f"Specific Small Arch (4 stages, d=[0,0,0,0], e=0.1 active, w=0.1): {macs_specific/1e6:.2f} M MACs, {params_specific/1e6:.2f} M Params")