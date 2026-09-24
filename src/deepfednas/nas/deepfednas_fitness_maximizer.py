# entropy_maximizer_with_effectiveness.py (Updated for Dynamic HxW and DeepMAD Guidelines)
import numpy as np
from tqdm import tqdm
import os
import sys



try:
    from deepfednas.utils.subnet_cost import subnet_macs
except ImportError:
    print("ERROR: Could not import 'subnet_macs' from subnet_macs.py in entropy_maximizer.")
    sys.exit(1)

K_MAIN_CONV = 3.0
K_PROJ_CONV = 1.0
GROUPS = 1.0
EPS = 1e-9

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None: min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v: new_v += divisor
    return int(new_v)

def _calculate_dynamic_stage_output_hws(arch_config_params):
    initial_hw = arch_config_params['initial_input_hw']
    stem_stride = arch_config_params['stem_stride']
    num_stages = arch_config_params['num_stages']
    stage_downsample_factors = arch_config_params['stage_downsample_factors']

    output_hws = []
    current_hw = initial_hw // stem_stride # HxW after stem

    for i in range(num_stages):
        current_hw = current_hw // stage_downsample_factors[i]
        output_hws.append(current_hw)
    return np.array(output_hws)

# ------------ ENTROPY CALCULATION (Parameterized) ------------
def calculate_paper_H_j_for_stage(
    stage_idx,
    d_extra_block,
    e_ratios_for_this_stage,
    scaled_c_in_to_stage,
    scaled_c_out_of_stage,
    stage_output_hw_for_entropy, # Specific HxW for this stage's entropy term
    arch_config_params
    ):
    channel_divisible_by = arch_config_params['channel_divisible_by']
    
    log_output_volume_term = np.log(stage_output_hw_for_entropy**2 * scaled_c_out_of_stage + EPS)
    sum_log_widths_term = 0.0
    current_scaled_c_in_to_block = float(scaled_c_in_to_stage)
    num_blocks_in_stage = 1 + d_extra_block

    for block_num in range(num_blocks_in_stage):
        current_exp_ratio = e_ratios_for_this_stage[block_num]
        scaled_c_mid_conv1 = float(make_divisible(scaled_c_out_of_stage * current_exp_ratio, divisor=channel_divisible_by))
        if scaled_c_mid_conv1 < channel_divisible_by: scaled_c_mid_conv1 = float(channel_divisible_by)

        width_conv1 = current_scaled_c_in_to_block * (K_MAIN_CONV**2) / GROUPS
        sum_log_widths_term += np.log(width_conv1 + EPS if width_conv1 > 0 else EPS)

        width_conv2 = scaled_c_mid_conv1 * (K_MAIN_CONV**2) / GROUPS
        sum_log_widths_term += np.log(width_conv2 + EPS if width_conv2 > 0 else EPS)

        if block_num == 0: # Projection
            width_proj = float(scaled_c_in_to_stage) * (K_PROJ_CONV**2) / GROUPS
            sum_log_widths_term += np.log(width_proj + EPS if width_proj > 0 else EPS)
        current_scaled_c_in_to_block = scaled_c_out_of_stage 

    if np.isneginf(log_output_volume_term) or np.isneginf(sum_log_widths_term) or \
       np.isnan(log_output_volume_term) or np.isnan(sum_log_widths_term):
        return 0.0
    return log_output_volume_term * sum_log_widths_term


def calculate_paper_deepmad_entropy_objective(depth_vec, exp_vec_values, w_indices, width_mult_options, arch_config_params):
    H_j_list = []
    num_stages = arch_config_params['num_stages']
    original_stem_out_channels = arch_config_params['original_stem_out_channels']
    original_stage_base_channels = arch_config_params['original_stage_base_channels']
    
    stage_output_hws_for_entropy = _calculate_dynamic_stage_output_hws(arch_config_params)
    
    alpha_weights = np.array(arch_config_params.get('alpha_weights', [1.0]*num_stages)) 
    channel_divisible_by = arch_config_params['channel_divisible_by']
    max_extra_blocks_per_stage = arch_config_params['max_extra_blocks_per_stage']
    max_blocks_total_per_stage_supernet = max_extra_blocks_per_stage + 1

    stem_width_multiplier = width_mult_options[w_indices[0]]
    scaled_stem_cout = make_divisible(original_stem_out_channels * stem_width_multiplier, divisor=channel_divisible_by)
    current_scaled_c_in_to_stage = float(scaled_stem_cout)

    for j in range(num_stages):
        stage_width_multiplier = width_mult_options[w_indices[j+1]]
        scaled_c_out_of_stage = make_divisible(original_stage_base_channels[j] * stage_width_multiplier, divisor=channel_divisible_by)
        if scaled_c_out_of_stage < channel_divisible_by: scaled_c_out_of_stage = channel_divisible_by

        num_actual_blocks_in_stage = 1 + depth_vec[j]
        exp_vec_stage_start_idx = j * max_blocks_total_per_stage_supernet
        e_ratios_for_current_stage = exp_vec_values[exp_vec_stage_start_idx : exp_vec_stage_start_idx + num_actual_blocks_in_stage]
        
        H_j = calculate_paper_H_j_for_stage(
            stage_idx=j,
            d_extra_block=depth_vec[j],
            e_ratios_for_this_stage=e_ratios_for_current_stage,
            scaled_c_in_to_stage=current_scaled_c_in_to_stage,
            scaled_c_out_of_stage=scaled_c_out_of_stage,
            stage_output_hw_for_entropy=stage_output_hws_for_entropy[j],
            arch_config_params=arch_config_params
        )
        H_j_list.append(H_j)
        current_scaled_c_in_to_stage = scaled_c_out_of_stage
        
    weighted_H_sum = np.dot(alpha_weights[:num_stages], np.array(H_j_list))
    return weighted_H_sum, H_j_list

# ------------ EFFECTIVENESS CALCULATION (Parameterized) -----------
def calculate_L_and_avg_log_w(depth_vec, exp_vec_values, w_indices, width_mult_options, arch_config_params):
    L_total = 0
    sum_log_w = 0.0
    num_stages = arch_config_params['num_stages']
    original_stem_out_channels = arch_config_params['original_stem_out_channels']
    original_stage_base_channels = arch_config_params['original_stage_base_channels']
    channel_divisible_by = arch_config_params['channel_divisible_by']
    max_extra_blocks_per_stage = arch_config_params['max_extra_blocks_per_stage']
    max_blocks_total_per_stage_supernet = max_extra_blocks_per_stage + 1

    stem_width_multiplier = width_mult_options[w_indices[0]]
    scaled_stem_cout = make_divisible(original_stem_out_channels * stem_width_multiplier, divisor=channel_divisible_by)
    
    c_in_stem = arch_config_params.get('initial_input_channels', 3.0) # typically 3 for RGB
    w_stem = c_in_stem * (K_MAIN_CONV**2) / GROUPS 
    sum_log_w += np.log(w_stem + EPS)
    L_total += 1
    current_scaled_c_in_to_next_stage = float(scaled_stem_cout)

    for stage_idx in range(num_stages):
        stage_width_multiplier = width_mult_options[w_indices[stage_idx+1]]
        scaled_c_out_of_this_stage = make_divisible(original_stage_base_channels[stage_idx] * stage_width_multiplier, divisor=channel_divisible_by)
        if scaled_c_out_of_this_stage < channel_divisible_by: scaled_c_out_of_this_stage = channel_divisible_by
        
        scaled_c_in_to_first_block_of_stage = current_scaled_c_in_to_next_stage
        
        num_blocks_in_stage_active = 1 + depth_vec[stage_idx]
        exp_vec_stage_start_idx = stage_idx * max_blocks_total_per_stage_supernet

        for block_num in range(num_blocks_in_stage_active):
            current_scaled_c_in_to_block = scaled_c_in_to_first_block_of_stage if block_num == 0 else scaled_c_out_of_this_stage
            current_exp_ratio = exp_vec_values[exp_vec_stage_start_idx + block_num]
            scaled_c_mid = float(make_divisible(scaled_c_out_of_this_stage * current_exp_ratio, divisor=channel_divisible_by))
            if scaled_c_mid < channel_divisible_by: scaled_c_mid = float(channel_divisible_by)

            w1 = current_scaled_c_in_to_block * (K_MAIN_CONV**2) / GROUPS
            sum_log_w += np.log(w1 + EPS if w1 > 0 else EPS)
            L_total += 1
            
            w2 = scaled_c_mid * (K_MAIN_CONV**2) / GROUPS
            sum_log_w += np.log(w2 + EPS if w2 > 0 else EPS)
            L_total += 1

            if block_num == 0: 
                w_proj = scaled_c_in_to_first_block_of_stage * (K_PROJ_CONV**2) / GROUPS
                sum_log_w += np.log(w_proj + EPS if w_proj > 0 else EPS)
                L_total += 1
        current_scaled_c_in_to_next_stage = scaled_c_out_of_this_stage
    
    if L_total == 0: return 0, 0.0
    avg_log_w = sum_log_w / L_total
    return L_total, avg_log_w

def calculate_effectiveness_rho(L_total, avg_log_w):
    if L_total == 0: return float('inf') # Penalize if no layers
    bar_w = np.exp(avg_log_w)
    if bar_w < EPS: return float('inf') # Penalize if average width is effectively zero
    return L_total / bar_w


# ------------ GA GENE ENCODING/DECODING HELPERS --------------------
def decode_chromosome(chrom, num_depth_genes, num_exp_genes, num_width_idx_genes, exp_opt_values_list):
    depth_genes_end = num_depth_genes
    exp_genes_end = num_depth_genes + num_exp_genes
    width_genes_end = exp_genes_end + num_width_idx_genes

    d_vec = chrom[:depth_genes_end].astype(int).tolist()
    e_indices = chrom[depth_genes_end:exp_genes_end].astype(int)
    w_indices = chrom[exp_genes_end:width_genes_end].astype(int).tolist()

    e_indices = np.clip(e_indices, 0, len(exp_opt_values_list) - 1)
    e_vec_values = [exp_opt_values_list[i] for i in e_indices]
    return {"d": d_vec, "e": e_vec_values, "w_indices": w_indices}

# ------------ GA FITNESS FUNCTION -----------------------------------
def calculate_fitness(chromosome, mac_limit, rho0_constraint, 
                      width_mult_options, exp_opt_values_list,
                      num_depth_genes, num_exp_genes, num_width_idx_genes,
                      arch_config_params_ga,
                      effectiveness_weight=0.1, verbose_debug=False):
    arch = decode_chromosome(chromosome, num_depth_genes, num_exp_genes, num_width_idx_genes, exp_opt_values_list)
    
    # Guideline 3: Non-Decreasing Number of Channels Constraint
    num_stages_fit = arch_config_params_ga['num_stages']
    original_stage_base_channels_fit = arch_config_params_ga['original_stage_base_channels']
    channel_divisible_by_fit = arch_config_params_ga['channel_divisible_by']
    
    scaled_stage_output_channels = []
    for j in range(num_stages_fit):
        stage_width_multiplier = width_mult_options[arch["w_indices"][j+1]] # w_indices[0] is for stem
        scaled_c_out_of_stage = make_divisible(
            original_stage_base_channels_fit[j] * stage_width_multiplier, 
            divisor=channel_divisible_by_fit
        )
        if scaled_c_out_of_stage < channel_divisible_by_fit: # Ensure min channel count
             scaled_c_out_of_stage = channel_divisible_by_fit
        scaled_stage_output_channels.append(scaled_c_out_of_stage)

    quantified_violation = 0.0
    # Optional: count the number of violating stage transitions
    # violation_count = 0 

    if arch_config_params_ga.get('non_decreasing_channels', True): # Check if constraint is active
        for i in range(len(scaled_stage_output_channels) - 1):
            current_channel = scaled_stage_output_channels[i]
            next_channel = scaled_stage_output_channels[i+1]
            if current_channel > next_channel:
                # Accumulate the magnitude of the violation
                quantified_violation += (current_channel - next_channel)
                # Optional:
                # violation_count += 1
    
    current_macs, _ = subnet_macs(arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga)
    if current_macs > mac_limit:
        if verbose_debug: print(f"Failed MAC: {current_macs/1e6:.2f}M > {mac_limit/1e6:.2f}M")
        return -1e9 

    L_total, avg_log_w = calculate_L_and_avg_log_w(arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga)
    rho = calculate_effectiveness_rho(L_total, avg_log_w)

    if rho > rho0_constraint:
        if verbose_debug: print(f"Failed Rho: {rho:.4f} > {rho0_constraint:.4f}")
        return -1e8 
        
    raw_paper_entropy_score, _ = calculate_paper_deepmad_entropy_objective(
        arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga
    )

    # Guideline 2: Uniform Stage Depth Penalty
    actual_stage_depths = [1 + d_extra for d_extra in arch["d"]]
    variance_of_depths = np.var(actual_stage_depths)
    Q_penalty_term = np.exp(variance_of_depths)
    beta_depth_penalty = arch_config_params_ga.get('beta_depth_penalty', 10.0)
    
    penalized_entropy_score = raw_paper_entropy_score - beta_depth_penalty * Q_penalty_term
    
    combined_fitness = penalized_entropy_score + effectiveness_weight * rho

    # Apply penalty for non-decreasing channel violation
    if arch_config_params_ga.get('non_decreasing_channels', True) and quantified_violation > 0:
        # Introduce a new hyperparameter for the penalty coefficient
        # You can add 'non_decreasing_penalty_coeff' to your arch_config_params_ga
        # For example, it could be passed via the example_arch_config_8_stages in your __main__
        channel_penalty_coeff = arch_config_params_ga.get('non_decreasing_penalty_coeff', 1.0) # Default to 1.0, tune as needed
        
        channel_fitness_penalty = channel_penalty_coeff * quantified_violation
        combined_fitness -= channel_fitness_penalty
        
        if verbose_debug:
            print(f"    Applied Non-Decreasing Channel Penalty. ViolationSum: {quantified_violation:.2f}, Coeff: {channel_penalty_coeff}, PenaltyValue: {channel_fitness_penalty:.2f}") 
    
    if verbose_debug: 
        print(f"PASSED: MACs={current_macs/1e6:.2f}M, ρ={rho:.4f}, StageDepths={actual_stage_depths}, VarDepths={variance_of_depths:.3f}, QPenalty={Q_penalty_term:.3f}, RawEntropy={raw_paper_entropy_score:.4f}, PenalizedEntropy={penalized_entropy_score:.4f}, CombinedFitness={combined_fitness:.4f}")
    return combined_fitness

# ------------ GA EVOLUTIONARY SEARCH -------------------------------
def run_entropy_max_ga(
    mac_budget,
    arch_config_params, 
    width_mult_options, 
    depth_choices, 
    exp_opt_values,
    rho0_constraint=2.0,
    effectiveness_fitness_weight=1000, 
    pop_size=64,
    generations=25,
    mutate_p=0.3,
    seed=0,
    max_init_attempts_factor=2000
    ):
    
    num_stages = arch_config_params['num_stages']
    max_extra_blocks_per_stage = arch_config_params['max_extra_blocks_per_stage']

    num_depth_genes_ga = num_stages
    max_blocks_total_per_stage = max_extra_blocks_per_stage + 1
    num_exp_genes_ga = num_stages * max_blocks_total_per_stage
    num_width_idx_genes_ga = num_stages + 1 
    chromosome_len_ga = num_depth_genes_ga + num_exp_genes_ga + num_width_idx_genes_ga

    rng = np.random.default_rng(seed)
    print(f"\nStarting GA: MAC Budget: {mac_budget/1e6:.2f} M, ρ₀ <= {rho0_constraint}")
    print(f"Arch Config: {num_stages} stages, Max {max_blocks_total_per_stage} blocks/stage.")
    print(f"Width Opts: {width_mult_options}, Depth Choices (extra): {depth_choices}, Exp Opts: {exp_opt_values}")
    print(f"Chromosome: {num_depth_genes_ga}d + {num_exp_genes_ga}e + {num_width_idx_genes_ga}w = {chromosome_len_ga} genes")
    print(f"Fitness: (Entropy - beta*Q_depth) + {effectiveness_fitness_weight} * Rho")

    pop = []
    max_total_init_attempts = pop_size * max_init_attempts_factor
    pbar_init = tqdm(total=pop_size, desc="Initializing Population", unit="individual", ncols=120)
    init_attempts_count = 0

    while len(pop) < pop_size and init_attempts_count < max_total_init_attempts:
        init_attempts_count +=1
        depth_g = rng.choice(depth_choices, num_depth_genes_ga)
        exp_g_indices = rng.integers(0, len(exp_opt_values), num_exp_genes_ga)
        w_g_indices = rng.integers(0, len(width_mult_options), num_width_idx_genes_ga)
        chromosome = np.concatenate([depth_g, exp_g_indices, w_g_indices])
        
        fit_score = calculate_fitness(
            chromosome, mac_budget, rho0_constraint, width_mult_options, exp_opt_values,
            num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga,
            arch_config_params, effectiveness_fitness_weight # verbose_debug=True for init can be helpful
        )
        # Check against -1e6 as higher penalties are for more specific constraint violations
        if fit_score > -1e6: # valid individual (passed all constraints)
            pop.append(chromosome)
            pbar_init.update(1)
        if init_attempts_count > 0 and init_attempts_count % (pop_size * 10) == 0:
             pbar_init.set_postfix_str(f"Generated: {init_attempts_count}, Found: {len(pop)}")
    pbar_init.close()

    if not pop:
        print(f"ERROR: Could not initialize ANY valid individuals after {init_attempts_count} attempts. Constraints/config error or MAC budget too restrictive.")
        return None, -float('inf')
    if len(pop) < pop_size:
        print(f"WARNING: GA Init only created {len(pop)}/{pop_size} individuals from {init_attempts_count} attempts.")
    
    pop = np.stack(pop)
    current_pop_size = len(pop)
    best_overall_fitness = -float('inf')
    best_overall_chromosome = None

    for gen in tqdm(range(generations), desc="GA Generations", ncols=120):
        if current_pop_size == 0: break
            
        current_pop_fitness_scores = np.array([
            calculate_fitness(
                c, mac_budget, rho0_constraint, width_mult_options, exp_opt_values,
                num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga,
                arch_config_params, effectiveness_fitness_weight
            ) for c in pop
        ])

        if current_pop_fitness_scores.size > 0:
            gen_best_fitness_idx = np.argmax(current_pop_fitness_scores)
            gen_best_fitness = current_pop_fitness_scores[gen_best_fitness_idx]
            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_chromosome = pop[gen_best_fitness_idx].copy()
        else: break 

####### new code start
        parent_chromosomes_list = [] # This will be a list of 1D arrays (chromosomes)
        
        # Determine number of parents to select (typically same as current_pop_size for this type of GA)
        num_parents_to_select = current_pop_size

        if current_pop_size == 0:
            tqdm.write(f"Gen {gen+1}/{generations} - Population is empty. Stopping.")
            break 
        elif current_pop_size < 3: # Population size 1 or 2
            # For very small populations, we'll select with replacement to get enough for tournament logic to not fail,
            # or ensure crossover has pairs. Let's aim to fill up to num_parents_to_select.
            # The original code selected current_pop_size parents.
            # If current_pop_size is 1, rng.choice(1,1,replace=True) is fine.
            # If current_pop_size is 2, rng.choice(2,2,replace=True) is fine.
            selected_indices = rng.choice(current_pop_size, num_parents_to_select, replace=True)
            for individual_idx in selected_indices:
                parent_chromosomes_list.append(pop[individual_idx])
        else: # current_pop_size >= 3, use tournament selection
            for _ in range(num_parents_to_select): 
                competitor_indices = rng.choice(current_pop_size, 3, replace=False)
                winner_idx_in_competitors = np.argmax(current_pop_fitness_scores[competitor_indices])
                parent_chromosomes_list.append(pop[competitor_indices[winner_idx_in_competitors]])
        
        if not parent_chromosomes_list: # Check if the list is empty
            tqdm.write(f"Gen {gen+1}/{generations} - No parents selected. Population might have become unfit or too small. Stopping.")
            break
        
        # 'parents' (as a 2D NumPy array) is used for crossover logic later
        parents = np.stack(parent_chromosomes_list) 
        
        # Calculate fitness scores for the selected parents array (used in tqdm log)
        # Note: gen_best_fitness was already found from current_pop_fitness_scores which includes these parents.
        # This is more for logging consistency if you want to log best *parent* fitness.
        # If not strictly needed for logging, this recalculation can be skipped.
        # For now, keeping it for the tqdm output.
        if parents.shape[0] > 0:
            parent_fitness_scores = np.array([calculate_fitness(
                p_chrom, mac_budget, rho0_constraint, width_mult_options, exp_opt_values,
                num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga,
                arch_config_params, effectiveness_fitness_weight
            ) for p_chrom in parents]) # Iterate over rows of the parents array
        else: # Should not happen if "if not parent_chromosomes_list: break" is hit
            parent_fitness_scores = np.array([-float('inf')])
        
        children = []
        num_parents = len(parents)
        if num_parents >= 2:
            for i in range(0, num_parents -1 , 2): 
                p1, p2 = parents[i], parents[i+1]
                cx = rng.integers(1, chromosome_len_ga -1) if chromosome_len_ga > 2 else 1
                children.extend([np.concatenate([p1[:cx], p2[cx:]]), np.concatenate([p2[:cx], p1[cx:]])])
            if num_parents % 2 == 1 and num_parents > 0: children.append(parents[-1]) 
        
        if not children and num_parents > 0: pop = np.copy(parents)
        elif children: pop = np.stack(children[:current_pop_size]) 
        else: break 

        for i in range(len(pop)): 
            if rng.random() < mutate_p:
                gene_to_mutate_idx = rng.integers(0, chromosome_len_ga)
                if gene_to_mutate_idx < num_depth_genes_ga:
                    pop[i, gene_to_mutate_idx] = rng.choice(depth_choices)
                elif gene_to_mutate_idx < num_depth_genes_ga + num_exp_genes_ga:
                    pop[i, gene_to_mutate_idx] = rng.integers(0, len(exp_opt_values))
                else: 
                    pop[i, gene_to_mutate_idx] = rng.integers(0, len(width_mult_options))
        
        if best_overall_chromosome is not None and len(pop) > 0: 
            child_fitness_scores_for_elitism = np.array([
                calculate_fitness(
                    c, mac_budget, rho0_constraint, width_mult_options, exp_opt_values,
                    num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga,
                    arch_config_params, effectiveness_fitness_weight
                ) for c in pop
            ])
            if child_fitness_scores_for_elitism.size > 0:
                worst_child_idx = np.argmin(child_fitness_scores_for_elitism)
                if calculate_fitness(
                    best_overall_chromosome, mac_budget, rho0_constraint, width_mult_options, exp_opt_values,
                    num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga,
                    arch_config_params, effectiveness_fitness_weight
                    ) > child_fitness_scores_for_elitism[worst_child_idx]:
                    pop[worst_child_idx] = best_overall_chromosome.copy()
        
        current_pop_size = len(pop)
        tqdm.write(f"Gen {gen+1}/{generations} - Overall Best Fitness (PenalizedEnt+w*Rho): {best_overall_fitness:.4f}", end='\r')

    if best_overall_chromosome is None and len(pop) > 0: # If best_overall_chromosome wasn't updated in the loop (e.g. only 1 gen)
        final_scores = np.array([
             calculate_fitness(
                c, mac_budget, rho0_constraint, width_mult_options, exp_opt_values,
                num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga,
                arch_config_params, effectiveness_fitness_weight
            ) for c in pop
        ])
        if final_scores.size > 0:
            best_overall_chromosome = pop[np.argmax(final_scores)]
            best_overall_fitness = np.max(final_scores)
    
    if best_overall_chromosome is None:
        print("ERROR: GA failed to find any valid solution after generations.")
        return None, -float('inf')
            
    print(f"\nEvolutionary search completed. Best overall combined fitness: {best_overall_fitness:.4f}")
    final_arch = decode_chromosome(best_overall_chromosome, num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga, exp_opt_values)
    return final_arch, best_overall_fitness


# ------------ EXAMPLE USAGE (main) ----------------------------------
if __name__ == '__main__':
    print("Running example: Parameterized GA with Dynamic HxW and DeepMAD Guidelines...")
    
    example_arch_config_8_stages = {
        'num_stages': 8,
        'initial_input_hw': 32, # For CIFAR-10 like data
        'initial_input_channels': 3, # For RGB images
        'stem_stride': 1, # Assuming ResNet stem for CIFAR-10 keeps 32x32
        'original_stem_out_channels': 32, # Example: Small stem output for a smaller overall net
        'original_stage_base_channels': np.array([32, 64, 128, 128, 256, 256, 512, 512]), # Example for 8 stages
        'stage_downsample_factors': [1, 1, 1, 1, 1, 2, 2, 2], # Example downsampling pattern
        'max_extra_blocks_per_stage': 7, # Max (3+1)=4 blocks per stage allows d_vec [0,1,2,3]
        'channel_divisible_by': 8,
        'n_classes': 10, # For CIFAR-10
        'alpha_weights': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), # Equal weights for 8 stages
        'beta_depth_penalty': 10.0, # DeepMAD paper's suggestion
        'non_decreasing_channels': True, # Enforce non-decreasing channel constraint
        'non_decreasing_penalty_coeff': 100.0, # Coefficient for non-decreasing channel penalty
    }
    
    # For the GA, depth_choices should be the range of *extra* blocks
    example_depth_options = np.array(list(range(example_arch_config_8_stages['max_extra_blocks_per_stage'] + 1)))


    example_width_options = [0.1, 0.25, 0.5, 0.75, 1.0] 
    example_exp_options = np.array([0.25], dtype=np.float64)

    target_mac_budget_eg = 3404 * 1e6  # 500 M MACs
    target_rho0_eg = 0.50             # Effectiveness constraint
    eff_weight_eg = 1000             # Weight for effectiveness in fitness

    # Test with a smaller population and fewer generations for quicker testing
    best_arch_res, best_fitness_score_res = run_entropy_max_ga(
        mac_budget=target_mac_budget_eg,
        arch_config_params=example_arch_config_8_stages,
        width_mult_options=example_width_options,
        depth_choices=example_depth_options, # Pass the choices for extra blocks
        exp_opt_values=example_exp_options,
        rho0_constraint=target_rho0_eg,
        effectiveness_fitness_weight=eff_weight_eg,
        pop_size=100, 
        generations=50, 
        seed=52,
        mutate_p=0.3,
        max_init_attempts_factor=2000
    )

    if best_arch_res:
        print("\nGA Search Complete. Best Architecture Found:")
        print(f"  Depth Vector (d_extra blocks): {best_arch_res['d']}")
        actual_depths = [1 + d for d in best_arch_res['d']]
        print(f"  Actual Blocks per Stage: {actual_depths}")
        w_actual_vals = [example_width_options[i] for i in best_arch_res['w_indices']]
        print(f"  Width Multiplier Indices: {best_arch_res['w_indices']} (Actual Vals: {w_actual_vals})")
        
        # Verify parameters of the found architecture
        final_macs_res, _ = subnet_macs(
            best_arch_res['d'], best_arch_res['e'], best_arch_res['w_indices'], 
            example_width_options, example_arch_config_8_stages
        )
        L_f_res, avg_log_w_f_res = calculate_L_and_avg_log_w(
            best_arch_res['d'], best_arch_res['e'], best_arch_res['w_indices'], 
            example_width_options, example_arch_config_8_stages
        )
        rho_f_res = calculate_effectiveness_rho(L_f_res, avg_log_w_f_res)
        
        raw_entropy_f_res, _ = calculate_paper_deepmad_entropy_objective(
            best_arch_res['d'], best_arch_res['e'], best_arch_res['w_indices'], 
            example_width_options, example_arch_config_8_stages
        )
        variance_of_depths_f_res = np.var(actual_depths)
        Q_penalty_term_f_res = np.exp(variance_of_depths_f_res)
        beta_depth_penalty_f_res = example_arch_config_8_stages.get('beta_depth_penalty', 10.0)
        penalized_entropy_f_res = raw_entropy_f_res - beta_depth_penalty_f_res * Q_penalty_term_f_res


        print(f"  Achieved MACs: {final_macs_res / 1e6:.2f} M (Target: {target_mac_budget_eg/1e6:.2f} M)")
        print(f"  Effectiveness ρ: {rho_f_res:.4f} (Target ρ₀ <= {target_rho0_eg})")
        print(f"  Raw Paper Entropy Score (Σ αH): {raw_entropy_f_res:.4f}")
        print(f"  Depth Variance Penalty (Q): {Q_penalty_term_f_res:.4f} (beta={beta_depth_penalty_f_res})")
        print(f"  Penalized Entropy Score: {penalized_entropy_f_res:.4f}")
        print(f"  Combined Fitness Score (PenalizedEnt + {eff_weight_eg}*Rho): {best_fitness_score_res:.4f}")

        # Channel non-decreasing check output
        scaled_stage_output_channels_final = []
        for j_idx in range(example_arch_config_8_stages['num_stages']):
            stage_width_multiplier_final = example_width_options[best_arch_res["w_indices"][j_idx+1]]
            scaled_c_out_final = make_divisible(
                example_arch_config_8_stages['original_stage_base_channels'][j_idx] * stage_width_multiplier_final,
                divisor=example_arch_config_8_stages['channel_divisible_by']
            )
            if scaled_c_out_final < example_arch_config_8_stages['channel_divisible_by']:
                 scaled_c_out_final = example_arch_config_8_stages['channel_divisible_by']
            scaled_stage_output_channels_final.append(scaled_c_out_final)
        print(f"  Scaled Stage Output Channels: {scaled_stage_output_channels_final}")
        is_non_decreasing_final = True
        for i_idx in range(len(scaled_stage_output_channels_final) - 1):
            if scaled_stage_output_channels_final[i_idx] > scaled_stage_output_channels_final[i_idx+1]:
                is_non_decreasing_final = False
                break
        print(f"  Channels Non-Decreasing?: {is_non_decreasing_final}")

    else:
        print("\nGA Search FAILED to find a suitable architecture.")