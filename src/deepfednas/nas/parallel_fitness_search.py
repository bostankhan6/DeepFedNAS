# entropy_maximizer_with_effectiveness.py (Corrected for Parallelization) v1_with_parallelization_added
import numpy as np
from tqdm import tqdm
import os
import sys
import multiprocessing
from functools import partial



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
    current_hw = initial_hw // stem_stride
    for i in range(num_stages):
        current_hw = current_hw // stage_downsample_factors[i]
        output_hws.append(current_hw)
    return np.array(output_hws)

def calculate_paper_H_j_for_stage(
    stage_idx, d_extra_block, e_ratios_for_this_stage,
    scaled_c_in_to_stage, scaled_c_out_of_stage,
    stage_output_hw_for_entropy, arch_config_params):
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
        if block_num == 0:
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
            stage_idx=j, d_extra_block=depth_vec[j], e_ratios_for_this_stage=e_ratios_for_current_stage,
            scaled_c_in_to_stage=current_scaled_c_in_to_stage, scaled_c_out_of_stage=scaled_c_out_of_stage,
            stage_output_hw_for_entropy=stage_output_hws_for_entropy[j], arch_config_params=arch_config_params
        )
        H_j_list.append(H_j)
        current_scaled_c_in_to_stage = scaled_c_out_of_stage
    weighted_H_sum = np.dot(alpha_weights[:num_stages], np.array(H_j_list))
    return weighted_H_sum, H_j_list

def calculate_L_and_avg_log_w(depth_vec, exp_vec_values, w_indices, width_mult_options, arch_config_params):
    L_total, sum_log_w = 0, 0.0
    num_stages = arch_config_params['num_stages']
    original_stem_out_channels = arch_config_params['original_stem_out_channels']
    original_stage_base_channels = arch_config_params['original_stage_base_channels']
    channel_divisible_by = arch_config_params['channel_divisible_by']
    max_extra_blocks_per_stage = arch_config_params['max_extra_blocks_per_stage']
    max_blocks_total_per_stage_supernet = max_extra_blocks_per_stage + 1
    stem_width_multiplier = width_mult_options[w_indices[0]]
    scaled_stem_cout = make_divisible(original_stem_out_channels * stem_width_multiplier, divisor=channel_divisible_by)
    c_in_stem = arch_config_params.get('initial_input_channels', 3.0)
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
    return L_total, sum_log_w / L_total

def calculate_effectiveness_rho(L_total, avg_log_w):
    if L_total == 0: return float('inf')
    bar_w = np.exp(avg_log_w)
    if bar_w < EPS: return float('inf')
    return L_total / bar_w

def decode_chromosome(chrom, num_depth_genes, num_exp_genes, num_width_idx_genes, exp_opt_values_list):
    depth_genes_end = num_depth_genes
    exp_genes_end = num_depth_genes + num_exp_genes
    d_vec = chrom[:depth_genes_end].astype(int).tolist()
    e_indices = chrom[depth_genes_end:exp_genes_end].astype(int)
    w_indices = chrom[exp_genes_end:exp_genes_end + num_width_idx_genes].astype(int).tolist()
    e_indices = np.clip(e_indices, 0, len(exp_opt_values_list) - 1)
    e_vec_values = [exp_opt_values_list[i] for i in e_indices]
    return {"d": d_vec, "e": e_vec_values, "w_indices": w_indices}

def calculate_fitness(chromosome, mac_limit, rho0_constraint, width_mult_options, exp_opt_values_list, num_depth_genes, num_exp_genes, num_width_idx_genes, arch_config_params_ga, effectiveness_weight=0.1, verbose_debug=False):
    arch = decode_chromosome(chromosome, num_depth_genes, num_exp_genes, num_width_idx_genes, exp_opt_values_list)
    num_stages_fit = arch_config_params_ga['num_stages']
    original_stage_base_channels_fit = arch_config_params_ga['original_stage_base_channels']
    channel_divisible_by_fit = arch_config_params_ga['channel_divisible_by']
    scaled_stage_output_channels = []
    for j in range(num_stages_fit):
        stage_width_multiplier = width_mult_options[arch["w_indices"][j+1]]
        scaled_c_out_of_stage = make_divisible(original_stage_base_channels_fit[j] * stage_width_multiplier, divisor=channel_divisible_by_fit)
        if scaled_c_out_of_stage < channel_divisible_by_fit: scaled_c_out_of_stage = channel_divisible_by_fit
        scaled_stage_output_channels.append(scaled_c_out_of_stage)
    quantified_violation = 0.0
    if arch_config_params_ga.get('non_decreasing_channels', True):
        for i in range(len(scaled_stage_output_channels) - 1):
            if scaled_stage_output_channels[i] > scaled_stage_output_channels[i+1]:
                quantified_violation += (scaled_stage_output_channels[i] - scaled_stage_output_channels[i+1])
    current_macs, _ = subnet_macs(arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga)
    if current_macs > mac_limit:
        if verbose_debug: print(f"Failed MAC: {current_macs/1e6:.2f}M > {mac_limit/1e6:.2f}M")
        return -1e9
    L_total, avg_log_w = calculate_L_and_avg_log_w(arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga)
    rho = calculate_effectiveness_rho(L_total, avg_log_w)
    if rho > rho0_constraint:
        if verbose_debug: print(f"Failed Rho: {rho:.4f} > {rho0_constraint:.4f}")
        return -1e8
    raw_paper_entropy_score, _ = calculate_paper_deepmad_entropy_objective(arch["d"], arch["e"], arch["w_indices"], width_mult_options, arch_config_params_ga)
    actual_stage_depths = [1 + d_extra for d_extra in arch["d"]]
    variance_of_depths = np.var(actual_stage_depths)
    Q_penalty_term = np.exp(variance_of_depths)
    beta_depth_penalty = arch_config_params_ga.get('beta_depth_penalty', 10.0)
    penalized_entropy_score = raw_paper_entropy_score - beta_depth_penalty * Q_penalty_term
    combined_fitness = penalized_entropy_score + effectiveness_weight * rho
    if arch_config_params_ga.get('non_decreasing_channels', True) and quantified_violation > 0:
        channel_penalty_coeff = arch_config_params_ga.get('non_decreasing_penalty_coeff', 1.0)
        combined_fitness -= channel_penalty_coeff * quantified_violation
        if verbose_debug:
            print(f"    Applied Non-Decreasing Channel Penalty. ViolationSum: {quantified_violation:.2f}, Coeff: {channel_penalty_coeff}, PenaltyValue: {channel_penalty_coeff * quantified_violation:.2f}")
    if verbose_debug:
        print(f"PASSED: MACs={current_macs/1e6:.2f}M, ρ={rho:.4f}, StageDepths={actual_stage_depths}, VarDepths={variance_of_depths:.3f}, QPenalty={Q_penalty_term:.3f}, RawEntropy={raw_paper_entropy_score:.4f}, PenalizedEntropy={penalized_entropy_score:.4f}, CombinedFitness={combined_fitness:.4f}")
    return combined_fitness

def _calculate_fitness_worker(fixed_args, chromosome):
    """ Worker function for multiprocessing. Receives chromosome as the first arg. """
    mac_limit, rho0_constraint, width_mult_options, exp_opt_values_list, \
    num_depth_genes, num_exp_genes, num_width_idx_genes, \
    arch_config_params_ga, effectiveness_weight = fixed_args
    
    return calculate_fitness(chromosome, mac_limit, rho0_constraint,
                             width_mult_options, exp_opt_values_list,
                             num_depth_genes, num_exp_genes, num_width_idx_genes,
                             arch_config_params_ga, effectiveness_weight)

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
    # ... (rest of the print statements are fine) ...

    # --- Prepare for Parallelization ---
    # Package all the fixed arguments that won't change per-individual
    fixed_args = (
        mac_budget, rho0_constraint, width_mult_options, exp_opt_values,
        num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga,
        arch_config_params, effectiveness_fitness_weight
    )
    # Use functools.partial to create a new function where fixed_args is the first argument,
    # and the chromosome will be the second argument (the one iterated by pool.map)
    worker_func = partial(_calculate_fitness_worker, fixed_args)

    with multiprocessing.Pool() as pool:
        # --- Parallel Initialization ---
        pop = []
        pbar_init = tqdm(total=pop_size, desc="Initializing Population", ncols=120)
        init_attempts_count = 0
        while len(pop) < pop_size and init_attempts_count < max_init_attempts_factor * pop_size:
            # Generate a batch of random chromosomes
            batch_size_init = max(1, (pop_size - len(pop)) * 5) # Ensure batch_size is at least 1
            candidate_chromosomes = []
            for _ in range(batch_size_init):
                init_attempts_count += 1
                depth_g = rng.choice(depth_choices, num_depth_genes_ga)
                exp_g_indices = rng.integers(0, len(exp_opt_values), num_exp_genes_ga)
                w_g_indices = rng.integers(0, len(width_mult_options), num_width_idx_genes_ga)
                candidate_chromosomes.append(np.concatenate([depth_g, exp_g_indices, w_g_indices]))
            
            # Evaluate the batch in parallel
            fitness_scores = pool.map(worker_func, candidate_chromosomes)
            
            # Filter for valid chromosomes
            valid_chromosomes = [chromo for chromo, score in zip(candidate_chromosomes, fitness_scores) if score > -1e7]
            
            needed = pop_size - len(pop)
            pop.extend(valid_chromosomes[:needed])
            pbar_init.update(len(pop) - pbar_init.n)
            pbar_init.set_postfix_str(f"Generated: {init_attempts_count}, Found: {len(pop)}")
        
        pbar_init.close()

        if not pop:
            print(f"ERROR: Could not initialize ANY valid individuals after {init_attempts_count} attempts.")
            return None, -float('inf')
        
        pop = np.stack(pop)
        best_overall_fitness = -float('inf')
        best_overall_chromosome = None

        # --- GA Generations with Parallel Fitness Evaluation ---
        for gen in tqdm(range(generations), desc="GA Generations", ncols=120):
            # Evaluate fitness of the entire population in parallel
            current_pop_fitness_scores = np.array(pool.map(worker_func, pop))

            gen_best_fitness_idx = np.argmax(current_pop_fitness_scores)
            if current_pop_fitness_scores[gen_best_fitness_idx] > best_overall_fitness:
                best_overall_fitness = current_pop_fitness_scores[gen_best_fitness_idx]
                best_overall_chromosome = pop[gen_best_fitness_idx].copy()
            
            # ... (The rest of your GA logic: selection, crossover, mutation, elitism) ...
            # ... This part is fast and can remain serial ...
            parents = [] # Selection logic here
            num_parents_to_select = len(pop)
            for _ in range(num_parents_to_select):
                competitor_indices = rng.choice(len(pop), 3, replace=False)
                winner_idx = competitor_indices[np.argmax(current_pop_fitness_scores[competitor_indices])]
                parents.append(pop[winner_idx])
            parents = np.array(parents)

            children = []
            for i in range(0, len(parents), 2):
                if i + 1 >= len(parents):
                    children.append(parents[i])
                    continue
                p1, p2 = parents[i], parents[i+1]
                cx_point = rng.integers(1, chromosome_len_ga)
                children.append(np.concatenate((p1[:cx_point], p2[cx_point:])))
                children.append(np.concatenate((p2[:cx_point], p1[cx_point:])))
            
            pop = np.array(children[:len(pop)])

            for i in range(len(pop)):
                if rng.random() < mutate_p:
                    gene_idx = rng.integers(chromosome_len_ga)
                    if gene_idx < num_depth_genes_ga: pop[i, gene_idx] = rng.choice(depth_choices)
                    elif gene_idx < num_depth_genes_ga + num_exp_genes_ga: pop[i, gene_idx] = rng.integers(len(exp_opt_values))
                    else: pop[i, gene_idx] = rng.integers(len(width_mult_options))

            if best_overall_chromosome is not None and len(pop) > 0:
                # Re-evaluate fitness of current population after mutation for elitism if needed
                # or ensure best_overall_chromosome is always present in the next generation.
                # For simplicity here, we ensure the best chromosome survives.
                # This assumes the best_overall_chromosome will remain valid and its fitness won't change
                # in a way that makes it worse than a newly generated child after mutation,
                # which might not always be true if constraints are tight.
                # A more robust approach might be to evaluate its fitness again or ensure it passes.
                if len(pop) > 0: # Ensure there are children to replace
                     # It's better to explicitly calculate the fitness of the best_overall_chromosome
                     # and ensure it's still good.
                    best_chromo_fitness = calculate_fitness(
                        best_overall_chromosome, mac_budget, rho0_constraint, width_mult_options, exp_opt_values,
                        num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga, arch_config_params, effectiveness_fitness_weight
                    )
                    child_fitness_scores_for_elitism = np.array(pool.map(worker_func, pop))
                    if child_fitness_scores_for_elitism.size > 0:
                        worst_child_idx = np.argmin(child_fitness_scores_for_elitism)
                        if best_chromo_fitness > child_fitness_scores_for_elitism[worst_child_idx]:
                            pop[worst_child_idx] = best_overall_chromosome.copy()

            tqdm.write(f"Gen {gen+1}/{generations} - Overall Best Fitness: {best_overall_fitness:.4f}", end='\r')

    # Final check for best chromosome if the loop finished without finding one
    if best_overall_chromosome is None and len(pop) > 0:
        final_scores = np.array(pool.map(worker_func, pop))
        if final_scores.size > 0:
            best_overall_chromosome = pop[np.argmax(final_scores)]
            best_overall_fitness = np.max(final_scores)
    
    if best_overall_chromosome is None:
        print("ERROR: GA failed to find any valid solution.")
        return None, -float('inf')
            
    print(f"\nEvolutionary search completed. Best overall combined fitness: {best_overall_fitness:.4f}")
    final_arch = decode_chromosome(best_overall_chromosome, num_depth_genes_ga, num_exp_genes_ga, num_width_idx_genes_ga, exp_opt_values)
    return final_arch, best_overall_fitness