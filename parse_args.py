import argparse
import json

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # ------------------ Model & Supernetwork Architecture Settings ------------------
    parser.add_argument(
        "--model",
        type=str,
        default="ofaresnet_generic",
        metavar="N",
        help="Defines the model type to be used. 'ofaresnet_generic' activates the new configurable framework.",
    )
    parser.add_argument(
        "--supernet_num_stages",
        type=int,
        default=4,
        help="Number of stages in the supernetwork.",
    )
    parser.add_argument(
        "--supernet_initial_input_hw",
        type=int,
        default=32,
        help="Initial height/width of the input (e.g., 32 for CIFAR-10).",
    )
    parser.add_argument(
        "--supernet_initial_input_channels",
        type=int,
        default=3,
        help="Number of input channels to the network (e.g., 3 for RGB).",
    )
    parser.add_argument(
        "--supernet_stem_stride",
        type=int,
        default=1,
        help="Stride of the initial stem convolution.",
    )
    parser.add_argument(
        "--supernet_original_stem_out_channels",
        type=int,
        default=64,
        help="Base output channels for the stem layer (before width multiplication).",
    )
    parser.add_argument(
        "--supernet_original_stage_base_channels",
        type=json.loads,
        default='[256, 512, 1024, 2048]',
        help="JSON list of base output channels for each stage. Length must equal num_stages.",
    )
    parser.add_argument(
        "--supernet_stage_downsample_factors",
        type=json.loads,
        default='[1, 2, 2, 2]',
        help="JSON list of downsampling factors (stride) for the first block of each stage. Length must equal num_stages.",
    )
    parser.add_argument(
        "--supernet_max_extra_blocks_per_stage",
        type=int,
        default=2,
        help="Maximum number of *additional* blocks a stage can have. Choices for 'd' will be [0, ..., this_value].",
    )
    parser.add_argument(
        "--supernet_channel_divisible_by",
        type=int,
        default=8,
        help="Ensures channel counts are divisible by this number.",
    )
    parser.add_argument(
        "--supernet_width_multiplier_choices",
        type=json.loads,
        default='[0.1, 0.14, 0.18, 0.22, 0.25]',
        help="JSON list of available width multipliers.",
    )
    parser.add_argument(
        "--supernet_expansion_ratio_choices",
        type=json.loads,
        default='[0.1, 0.14, 0.18, 0.22, 0.25]',
        help="JSON list of available expansion ratios for blocks.",
    )

    # ------------------ DeepFedNAS-related NAS Sampler Settings ------------------
    parser.add_argument(
        "--supernet_alpha_weights",
        type=json.loads,
        default='[1.0, 1.0, 1.0, 1.0]',
        help="JSON list of alpha weights for entropy calculation per stage.",
    )
    parser.add_argument(
        "--supernet_beta_depth_penalty",
        type=float,
        default=10.0,
        help="Beta penalty for depth variance in entropy calculation.",
    )
    parser.add_argument(
        "--supernet_rho0_constraint",
        type=float,
        default=0.55,
        help="Effectiveness (rho0) constraint for entropy-based sampling.",
    )
    parser.add_argument(
        "--supernet_effectiveness_fitness_weight",
        type=float,
        default=1000,
        help="Weight of effectiveness term in fitness for entropy-based sampling.",
    )
    parser.add_argument(
        "--supernet_non_decreasing_penalty_coeff",
        type=float,
        default=100.0,
        help="Penalty coefficient if non-decreasing channel constraint is violated during entropy-based sampling.",
    )

    #  Genetic Algorithm (GA) Sampler Settings
    parser.add_argument(
        "--ga_pop_size",
        type=int,
        default=128,
        help="Population size for the genetic algorithm.",
    )
    parser.add_argument(
        "--ga_generations",
        type=int,
        default=100,
        help="Number of generations for the genetic algorithm.",
    )
    parser.add_argument(
        "--ga_mutate_p",
        type=float,
        default=0.3,
        help="Mutation probability for the genetic algorithm.",
    )

    # ------------------ Federated Learning and Training Settings ------------------
    parser.add_argument(
        "--dataset", type=str, default="cifar10", help="dataset used for training"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./../../../data/cifar10", help="data directory"
    )
    parser.add_argument(
        "--partition_method", type=str, default="homo", help="how to partition the dataset"
    )
    parser.add_argument(
        "--partition_alpha", type=float, default=1.0, help="partition alpha"
    )
    parser.add_argument(
        "--client_num_in_total", type=int, default=20, help="number of workers in a distributed cluster"
    )
    parser.add_argument(
        "--client_num_per_round", type=int, default=8, help="number of workers"
    )
    parser.add_argument(
        "--comm_round", type=int, default=1501, help="how many round of communications"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="how many epochs will be trained locally"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="input batch size for training"
    )
    parser.add_argument(
        "--client_optimizer", type=str, default="sgd", help="client optimizer (e.g., sgd, adam)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate"
    )
    parser.add_argument(
        "--wd", help="weight decay parameter", type=float, default=5e-4
    )
    parser.add_argument(
        "--max_norm", type=float, default=10.0, help="gradient clipping max_norm"
    )
    parser.add_argument(
        "--bn_gamma_zero_init", action="store_true", default=False, help="Use gamma=0 in last BN of residual blocks"
    )

    # ------------------ Subnet Distribution and Sampling ------------------
    parser.add_argument(
        "--subnet_dist_type",
        type=str,
        default="TS_all_random",
        choices=['static', 'TS_optimal_path', 'TS_cached_entropy_maximizer', 'TS_entropy_maximizer', 'dynamic', 'all_random', 'TS_all_random', 'sandwich_all_random', 'TS_compound', 'max_sample_count', 'multi_sandwich', 'TS_KD', 'PS'],
        help="Subnetwork selection strategy for every round"
    )
    
    parser.add_argument(
        "--subnet_cache_path",
        type=str,
        default='fitness_prediction_dataset.csv',
        help="Path to the cached entropy maximizer data. If None, it will not use cached data.",
    )
    parser.add_argument(
        "--diverse_subnets",
        type=json.loads,
        default='{"0":{"d":[0,0,0,0],"e":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],"w_indices":[0,0,0,0,0]}}',
        help="Subnets for static sampling. 'd' len=num_stages, 'e' len=num_stages*(max_extra+1), 'w_indices' len=num_stages+1"
    )
    parser.add_argument(
        "--top_k_maxnet", type=int, default=1, help="How many max subnets to sample per round"
    )
    parser.add_argument(
        "--bottom_k_maxnet", type=int, default=1, help="How many min subnets to sample per round"
    )

    # ------------------ Multi-Architecture and KD Settings (Legacy but Required) ------------------
    parser.add_argument(
        "--multi", action="store_true", help="Pass on multiple networks to a single client"
    )
    parser.add_argument(
        "--num_multi_archs", type=int, default=1, help="Number of archs for multi-net run"
    )
    parser.add_argument(
        "--kd_ratio",
        type=float,
        default=0.0,
        help="Knowledge Distillation ratio."
    )
    parser.add_argument(
        "--teacher_ckpt_name",
        type=str,
        default=None,
        help="Name of teacher model to load from W&B."
    )
    parser.add_argument(
        "--teacher_run_path",
        type=str,
        default=None,
        help="Run path of teacher model to load from W&B."
    )

    # ------------------ System and Logging Settings ------------------
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU ID to use"
    )
    parser.add_argument(
        "--init_seed", type=int, default=0, help="Initial seed for reproducibility"
    )
    parser.add_argument(
        "--wandb_project_name", type=str, default="deepfednas", help="Project Name for wandb logging"
    )
    parser.add_argument(
        "--wandb_group", type=str, default="validation_runs", help="W&B group name for organizing runs"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="generic_run_1", help="Run Name for wandb logging"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="bostankhan6-m-lardalens-university", help="W&B entity (username or team)"
    )
    
    parser.add_argument(
        "--wandb_watch",
        action="store_true",
        help="Enable wandb.watch() to log model gradients and parameters."
    )
    parser.add_argument(
        "--wandb_watch_freq",
        type=int,
        default=100, # A sensible default
        help="Frequency (in steps) for logging gradients with wandb.watch()"
    )
    
    # ------------------ Model Loading/Saving and Resuming ------------------
    parser.add_argument(
        "--local_model_ckpt_path",
        type=str,
        default=None,
        help="Direct local filesystem path to a model checkpoint (.pt file) to load.",
    )
    parser.add_argument(
        "--model_ckpt_name", type=str, default=None, help="Name of model to load from W&B."
    )
    parser.add_argument(
        "--run_path", type=str, default=None, help="Run path of model to load from W&B."
    )
    parser.add_argument(
        "--wandb_run_id_resume",
        type=str,
        default=None,
        help="W&B run ID to resume logging to an existing run. If None, a new run is created."
    )
    parser.add_argument(
        "--resume_round", type=int, default=0, help="Round to resume training from (requires a valid checkpoint path)."
    )
    parser.add_argument(
        "--cli_subnet_track",
        type=json.loads,
        default=None,
        help="Subnet tracker state for resuming run, typically loaded from checkpoint.",
    )
    
    # ------------------ Testing and Evaluation Frequency ------------------
    parser.add_argument(
        "--frequency_of_the_test", type=int, default=10, help="frequency of testing"
    )
    
    parser.add_argument('--val_batch_size', type=int, default=512, 
                        help='Batch size for validation and testing (default: 512).')

    # --- Other Legacy Arguments from original file for compatibility ---
    parser.add_argument(
        "--ps_depth_only", type=int, default=750, help="Rounds for depth-only sampling in PS"
    )
    parser.add_argument(
        "--weighted_avg_schedule", type=json.loads, default=None, help="Dynamic weighted avg hyper-parameters"
    )
    
    # ------------------ Missing Arguments from Old ------------------
    parser.add_argument(
        "--supernet_non_decreasing_channels_constraint",
        type=bool,
        default=True,
        help="Whether to enforce non-decreasing channels guideline in entropy calculation if used for sampling.",
    )
    parser.add_argument(
        "--skip_train_largest",
        action="store_true",
        help="Pass on multiple networks to a single client",
    )
    parser.add_argument(
        "--multi_drop_largest",
        action="store_true",
        help="(Needs to be reworked!) drops largest model and smallest to allow single middle model to be aggregated only",
    )
    parser.add_argument(
        "--cli_supernet",
        action="store_true",
        help="Pass on entire supernetwork to a single client",
    )
    parser.add_argument(
        "--cli_supernet_ps",
        action="store_true",
        help="Pass on entire supernetwork to a single client and perform PS on each client",
    )
    parser.add_argument(
        "--inplace_kd",
        action="store_true",
        help="In place knowledge distillation",
    )
    parser.add_argument(
        "--multi_disable_rest_bn",
        action="store_true",
        help="Multi setting Disable BN for rest of the networks apart from the largest one",
    )
    parser.add_argument(
        "--warmup_init_lr",
        type=float,
        default=0,
        help="warmup learning rate init (default: 0)",
    )
    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=0,
        help="Number of rounds to perform LR warmup",
    )
    parser.add_argument(
        "--largest_subnet_wd",
        help="weight decay parameter for largest subnet",
        type=float,
        default=0,
    )
    parser.add_argument(
        '--ci', 
        type=int, 
        default=0, 
        help='CI'
    )
    parser.add_argument(
        "--ckpt_subnets",
        type=json.loads,
        default=None,
        help="List of subnets used to checkpoint best model on interval",
    )
    parser.add_argument(
        "--use_bn",
        action="store_true",
        help="Use batchnorm",
    )
    parser.add_argument(
        "--reset_bn_stats",
        action="store_true",
        help="Resets bn mean and variance before test and training",
        default=False,
    )
    parser.add_argument(
        "--reset_bn_stats_test",
        action="store_true",
        help="Resets bn mean and variance before testing",
    )
    parser.add_argument(
        "--efficient_test",
        action="store_true",
        help="Test efficiently",
    )
    parser.add_argument(
        "--reset_bn_sample_size",
        type=float,
        default=0.2,
        help="percentage of local train data to use to reset bn stats",
    )
    parser.add_argument(
        "--ofa_config",
        type=json.loads,
        default=None,
        help="OFA ResNet configuration",
    )
    parser.add_argument(
        "--kd_type",
        type=str,
        default="ce",
        choices=["ce", "mse"],
        help="knowledge Distillation using pretrained model",
    )
    parser.add_argument(
        "--lr_schedule",
        type=json.loads,
        default=None,
        help="Weighted random"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just simulate distribution of subnetworks and log number of parameters and gflops of each subnet",
    )
    parser.add_argument(
        "--model_checkpoint_freq",
        type=int,
        default=500,
        help="frequency at which model is checkpointed",
    )
    parser.add_argument(
        "--best_model_freq",
        type=int,
        default=1000,
        help="intervals at which to track best model",
    )
    parser.add_argument(
        "--custom_config",
        type=json.loads,
        default=None,
        help="Allows user to specify custom config.yaml for crashed runs or runs missing config.yaml",
    )
    parser.add_argument(
        "--optim_step_more",
        action="store_true",
        help="Take optimizer step after each gradient in cli supernet",
    )
    parser.add_argument(
        "--largest_step_more",
        action="store_true",
        help="Take optimizer step after largest subnet gradient in cli supernet",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log in more detail",
    )
    parser.add_argument(
        "--verbose_test",
        action="store_true",
        help="Log testing in more detail",
    )
    parser.add_argument(
        "--feddyn",
        action="store_true",
        help="Use FedDyn Optimization",
    )
    parser.add_argument(
        "--feddyn_alpha",
        type=float,
        default=0.01,
        help="FedDyn optimization hyperparameter alpha. Generally one of [0.1, 0.01, 0.001]",
    )
    parser.add_argument(
        "--feddyn_max_norm",
        type=float,
        default=10,
        help="FedDyn optimization gradient clipping max_norm",
    )
    parser.add_argument(
        "--feddyn_no_wd_modifier",
        action="store_true",
        help="Don't use feddyn modified wd",
    )
    parser.add_argument(
        "--feddyn_override_wd",
        type=float,
        default=-1,
        help="Override feddyn wd if > 0",
    )
    parser.add_argument(
        "--mod_wd_dyn",
        action="store_true",
        help="Use FedDyn modified wd in non-feddyn runs",
    )
    parser.add_argument(
        "--weight_dataset",
        action="store_true",
        help="Include dataset size into weighted averaging",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.45,
        help="Elastic TCN dropout",
    )
    parser.add_argument(
        "--emb_dropout",
        type=float,
        default=0.25,
        help="Elastic TCN embedding dropout",
    )
    parser.add_argument(
        "--ksize",
        type=int,
        default=3,
        help="Elastic TCN kernel size",
    )
    parser.add_argument(
        "--emsize",
        type=int,
        default=600,
        help="Elastic TCN size of word embeddings (default: 600)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Elastic TCN # of levels (default: 4)",
    )
    parser.add_argument(
        "--nhid",
        type=int,
        default=600,
        help="Elastic TCN number of hidden units per layer (default: 600)",
    )
    parser.add_argument(
        "--tied",
        action="store_false",
        help="Include dataset tie the encoder-decoder weights (default: True)",
    )
    parser.add_argument(
        "--validseqlen",
        type=int,
        default=40,
        help="Elastic TCN valid sequence length (default: 40)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=80,
        help="Elastic TCN total sequence length, including effective history (default: 80)",
    )
    parser.add_argument(
        "--skip_train_test",
        action="store_true",
        help="skip train ppl calculations",
    )
    parser.add_argument(
        "--init_channel_size",
        type=int,
        default=128,
        help="Initial number of channels in convolution of darts model",
    )
    parser.add_argument(
        "--darts_layers",
        type=int,
        default=20,
        help="Number of layers in darts model",
    )
    parser.add_argument(
        "--PS_with_largest",
        action="store_true",
        help="During client supernet training using PS, select the largest subnetwork once and sample k-1 random subnets based on phase",
    )
    parser.add_argument(
        "--use_train_pkl",
        action="store_true",
        help="Use train dataset generated after splitting original train dataset into train and val datasets",
    )


    return parser

def add_args_test(parser): # Keep this function as is for now, or update similarly if needed for your testing script.
    parser.add_argument(
        "--model",
        type=str,
        default="resnet56",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="deepfednas",
        metavar="N",
        help="Project Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="deepfednas",
        metavar="N",
        help="Run Name of wandb for logging",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="deepfednas",
        metavar="N",
        help="helps in creating a project under teams or personal username",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./../../../data/cifar10",
        help="data directory",
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="homo",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=1,
        metavar="PA",
        help="partition alpha (default: 1)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=20,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument(
        "--model_ckpt_name", type=str, default=None, help="Name of model to load",
    )

    parser.add_argument(
        "--run_path", type=str, default=None, help="Run path of model to load",
    )

    parser.add_argument(
        "--init_seed",
        type=int,
        default=0,
        help="Initial seed for intializing model weights and partitioning dataset amongst clients",
    )

    parser.add_argument(
        "--test_subnets",
        type=json.loads,
        default=None,
        help="Subnets to test as a list",
    )

    parser.add_argument(
        "--nas", action="store_true", help="Perform NAS",
    )

    parser.add_argument(
        "--nas_constraints",
        type=json.loads,
        default=None,
        help="List of constraints to be used during nas",
    )

    parser.add_argument(
        "--mutate_prob", type=float, default=0.1, help="nas mutation probability",
    )

    parser.add_argument(
        "--population_size", type=int, default=100, help="nas population size",
    )

    parser.add_argument(
        "--max_time_budget", type=int, default=500, help="nas max time taken to sample",
    )

    parser.add_argument(
        "--parent_ratio", type=float, default=0.25, help="nas parent ratio",
    )

    parser.add_argument(
        "--mutation_ratio", type=float, default=0.5, help="nas mutation ratio",
    )

    parser.add_argument(
        "--multi_seed_test",
        action="store_true",
        help="Perform test on multiple seed and save results for plotting",
    )

    parser.add_argument(
        "--seed_list",
        type=json.loads,
        default=None,
        help="List of init seed to be used during multi seed test",
    )

    parser.add_argument(
        "--multi_seed_run_paths",
        type=json.loads,
        default=None,
        help="run path for model to test as a list corresponding to seed_list",
    )

    parser.add_argument(
        "--multi_seed_model_ckpt_names",
        type=json.loads,
        default=None,
        help="names of model to load from run paths",
    )

    parser.add_argument(
        "--multi_seed_test_subnets",
        type=json.loads,
        default=None,
        help="Subnets to test as a list for multi seed testing",
    )

    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="path and name of csv to save test data",
    )

    parser.add_argument(
        "--feddyn", action="store_true", help="Use FedDyn Optimization",
    )

    parser.add_argument(
        "--feddyn_alpha",
        type=float,
        default=0.1,
        help="FedDyn optimization hyperparameter alpha. Generally one of [0.1, 0.01, 0.001]",
    )

    return parser