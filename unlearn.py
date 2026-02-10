from recbole.quick_start import unlearn_recbole
from recbole.config import Config
import argparse

import numpy as np
import random
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="amazon_reviews", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")

    # Distributed training configuration
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    # General training configuration
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of training epochs",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="gpu to run on",
    )

    # Task configuration
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["CF", "SBR", "NBR"],
        help="Recommendation task",
        default="SBR",
    )
    parser.add_argument(
        "--ind",
        type=int,
        default=0,
        help="Index for the keyset and model version"
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs='+',
        default=[10, 20],
        help="Top-k parameters (space-separated list, e.g., --topk 10 20)"
    )
    parser.add_argument(
        "--training",
        type=int,
        default=None,
        help="1 for retraining, 2 for unlearning"
    )

    # Data split configuration
    parser.add_argument(
        "--temporal_split",
        action="store_true",
        help="Temporal split flag. If not set do user split"
    )
    parser.add_argument(
        "--LOCAL",
        action="store_true",
        help="Local flag. If not set assume we run on cluster"
    )

    # Unlearning configuration
    parser.add_argument(
        "--unlearning_fraction",
        type=float,
        default=0.00001,
        help="percentage of training data to unlearn"
    )
    parser.add_argument(
        "--unlearning_sample_selection_method",
        type=str,
        default="popular",
        # choices=["random", "popular", "unpopular", "sensitive_category_meat", "sensitive_category_baby", "sensitive_category_alcohol"],
        help="how should the forget set be chosen (what type of interactions appear there)"
    )
    parser.add_argument(
        "--popular_percentage",
        type=float,
        default=0.1,
        help="Fraction of most/least popular items to consider"
    )
    parser.add_argument(
        "--unlearning_algorithm",
        type=str,
        default="scif",
        choices=["scif", "kookmin", "fanchuan", "gif", "ceu", "idea", "seif"],
        help="what unlearning algorithm to use",
    )
    parser.add_argument(
        "--sensitive_category",
        type=str,
        default=None,
        # choices=["baby", "meat", "alcohol"],
        help="When choosing sensitive items to unlearn, choose which category"
    )
    parser.add_argument(
        "--retrain_checkpoint_idx_to_match",
        type=int,
        default=None,
        help="which unlearning checkpoint should be taken as example to create the unlearning set (only take a subset of the unlearning set given)"
    )
    parser.add_argument(
        "--unlearning_batchsize",
        type=int,
        default=1,
        help="number of users to unlearn in a single batch (default: 1 for sequential processing)"
    )
    parser.add_argument(
        "--max_training_hours",
        type=float,
        default=None,
        help="maximum total time for unlearning in hours (default: None for no time limit). Time limit per batch = max_training_hours / num_batches"
    )

    # SCIF-specific parameters
    parser.add_argument(
        "--max_norm",
        type=float,
        default=None,
        help="if not None, scif update step clipping will be applied with this value",
    )
    parser.add_argument(
        "--lissa_train_pair_count_scif",
        type=int,
        default=1024,
        help="how many samples are used for lissa hessian estimation"
    )
    parser.add_argument(
        "--retain_samples_used_for_update",
        type=int,
        default=128,
        help="how many samples are used in the HVP Hv inside v (v is the avg of the gradients of the unlearn sample, the cleaned one and some retain samples)"
    )

    # Kookmin-specific parameters
    parser.add_argument(
        "--kookmin_init_rate",
        type=float,
        default=0.01,
        help="fraction of params to re-initialize with kookmin unlearning",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.01,
        help="damping parameter for conjugate gradient solver in SCIF (lambda in (H + lambda*I)x = v)",
    )

    # GIF-specific parameters
    parser.add_argument(
        "--gif_damping",
        type=float,
        default=0.01,
        help="damping factor (lambda) for GIF Hessian approximation convergence",
    )
    parser.add_argument(
        "--gif_scale_factor",
        type=float,
        default=1000,
        help="scaling factor for GIF Hessian to ensure convergence",
    )
    parser.add_argument(
        "--gif_iterations",
        type=int,
        default=100,
        help="number of iterations for GIF Hessian inverse approximation",
    )
    parser.add_argument(
        "--gif_k_hops",
        type=int,
        default=2,
        help="number of hops for influenced neighbors in GIF (k-hop neighborhood)",
    )
    parser.add_argument(
        "--gif_use_true_khop",
        action="store_true",
        help="use true k-hop neighbor computation instead of random sampling (default: True for sparse graphs)",
    )
    parser.add_argument(
        "--gif_retain_samples",
        type=int,
        default=None,
        help="number of retain samples for GIF fine-tuning (default: 128 * forget_size)",
    )

    # CEU-specific parameters
    parser.add_argument(
        "--ceu_lambda",
        type=float,
        default=0.01,
        help="regularization rate (lambda) for CEU loss function",
    )
    parser.add_argument(
        "--ceu_sigma",
        type=float,
        default=0.1,
        help="standard deviation (sigma) for Gaussian noise in CEU",
    )
    parser.add_argument(
        "--ceu_epsilon",
        type=float,
        default=0.1,
        help="epsilon parameter for (epsilon, delta)-certified unlearning guarantee",
    )
    parser.add_argument(
        "--ceu_cg_iterations",
        type=int,
        default=100,
        help="number of conjugate gradient iterations for inverse Hessian approximation in CEU",
    )
    parser.add_argument(
        "--ceu_hessian_samples",
        type=int,
        default=1024,
        help="number of samples used for Hessian computation in CEU",
    )

    # IDEA-specific parameters
    parser.add_argument(
        "--idea_damping",
        type=float,
        default=0.01,
        help="damping factor (lambda) for IDEA Hessian approximation (for strong convexity)",
    )
    parser.add_argument(
        "--idea_sigma",
        type=float,
        default=0.1,
        help="standard deviation (sigma) for Gaussian noise in IDEA certification",
    )
    parser.add_argument(
        "--idea_epsilon",
        type=float,
        default=0.1,
        help="epsilon parameter for (epsilon, delta)-certified unlearning guarantee in IDEA",
    )
    parser.add_argument(
        "--idea_delta",
        type=float,
        default=0.01,
        help="delta parameter for (epsilon, delta)-certified unlearning guarantee in IDEA",
    )
    parser.add_argument(
        "--idea_iterations",
        type=int,
        default=100,
        help="number of iterations for stochastic Hessian inverse estimation in IDEA",
    )
    parser.add_argument(
        "--idea_hessian_samples",
        type=int,
        default=1024,
        help="number of samples used for Hessian computation in IDEA",
    )

    # SEIF-specific parameters
    parser.add_argument(
        "--seif_erase_std",
        type=float,
        default=0.6,
        help="standard deviation for Gaussian noise applied in SEIF erase phase",
    )
    parser.add_argument(
        "--seif_erase_std_final",
        type=float,
        default=0.005,
        help="standard deviation for final robustness noise in SEIF (applied before last epoch)",
    )
    parser.add_argument(
        "--seif_repair_epochs",
        type=int,
        default=4,
        help="number of repair epochs for SEIF fine-tuning on retain set",
    )
    parser.add_argument(
        "--seif_forget_class_weight",
        type=float,
        default=0.05,
        help="weight for forget class in SEIF weighted cross-entropy loss (lower = more forgetting)",
    )
    parser.add_argument(
        "--seif_learning_rate",
        type=float,
        default=0.0007,
        help="learning rate for SEIF repair phase",
    )
    parser.add_argument(
        "--seif_momentum",
        type=float,
        default=0.9,
        help="momentum for SGD optimizer in SEIF",
    )
    parser.add_argument(
        "--seif_weight_decay",
        type=float,
        default=5e-4,
        help="weight decay for SGD optimizer in SEIF",
    )

    # Spam/attack configuration
    parser.add_argument(
        "--spam",
        action="store_true",
        help="flag if we are in the spam setting",
    )
    parser.add_argument(
        "--n_target_items",
        type=int,
        default=10,
        help="number of target items the attacker wants to boost in the spam setting",
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="if set, only evaluate instead of unlearning and then evaluating",
    )
    parser.add_argument(
        "--sensitive_eval_only",
        action="store_true",
        help="if set, only evaluate sensitive items and skip normal evaluation metrics (must be used with --eval_only)",
    )
    parser.add_argument(
        "--dont_save_interaction_probabilities",
        action="store_true",
        help="if set, skip saving model interaction probabilities to pickle file (saves storage)",
    )

    # Model storage
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Directory where the models are saved"
    )

    # RULI Privacy evaluation configuration
    parser.add_argument(
        "--ruli_privacy_evaluation",
        action="store_true",
        help="Enable RULI Privacy (Game 2) MIA evaluation"
    )
    parser.add_argument(
        "--ruli_privacy_beta_threshold",
        type=float,
        default=0.5,
        help="Decision threshold for membership inference (default: 0.5)"
    )
    parser.add_argument(
        "--ruli_privacy_n_population_samples",
        type=int,
        default=2500,
        help="Number of population samples for comparison (default: 2500)"
    )
    parser.add_argument(
        "--ruli_privacy_shadow_models_dir",
        type=str,
        default=None,
        help="Directory where shadow models are stored (optional, uses default if not specified)"
    )
    parser.add_argument(
        "--ruli_privacy_k",
        type=int,
        default=8,
        help="Number of shadow models to use (default: 8, must match shadow model computation)"
    )
    parser.add_argument(
        "--ruli_privacy_d_target_seed",
        type=int,
        default=None,
        help="Random seed for D_target construction (optional, uses config seed if not specified)"
    )
    parser.add_argument(
        "--ruli_privacy_unlearning_algorithm",
        type=str,
        default=None,
        choices=["scif", "kookmin", "fanchuan", "gif", "ceu", "idea", "seif"],
        help="Unlearning algorithm to use for Qh shadow models (if multiple available, defaults to first available or matches --unlearning_algorithm)"
    )

    args, _ = parser.parse_known_args()

    if args.seed is not None:
        set_seed(args.seed)

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    config_dict = {
        "dataset": args.dataset,
        "unlearning_fraction": args.unlearning_fraction,
        "unlearning_sample_selection_method": args.unlearning_sample_selection_method,
        "model": args.model,
        "n_target_items": args.n_target_items,
        "unlearn_sample_selection_seed": args.seed,
        "seed": args.seed,
        "spam": args.spam,
        "unlearning_algorithm": args.unlearning_algorithm,
        "gpu_id": args.gpu_id,
        "eval_only": args.eval_only,
        "sensitive_eval_only": args.sensitive_eval_only,
        "save_interaction_probabilities": not args.dont_save_interaction_probabilities,
        "task_type": args.task_type,
        "ind": args.ind,
        "topk": args.topk,
        "training": args.training,
        "temporal_split": args.temporal_split,
        "LOCAL": args.LOCAL,
        "popular_percentage": args.popular_percentage,
        "sensitive_category": args.sensitive_category,
        "retrain_checkpoint_idx_to_match": args.retrain_checkpoint_idx_to_match,
        "unlearning_batchsize": args.unlearning_batchsize,
        "max_norm": args.max_norm,
        "lissa_train_pair_count_scif": args.lissa_train_pair_count_scif,
        "retain_samples_used_for_update": args.retain_samples_used_for_update,
        "kookmin_init_rate": args.kookmin_init_rate,
        "model_dir": args.model_dir,
        "damping": args.damping,
        "gif_damping": args.gif_damping,
        "gif_scale_factor": args.gif_scale_factor,
        "gif_iterations": args.gif_iterations,
        "gif_k_hops": args.gif_k_hops,
        "gif_use_true_khop": args.gif_use_true_khop,
        "ceu_lambda": args.ceu_lambda,
        "ceu_sigma": args.ceu_sigma,
        "ceu_epsilon": args.ceu_epsilon,
        "ceu_cg_iterations": args.ceu_cg_iterations,
        "ceu_hessian_samples": args.ceu_hessian_samples,
        "idea_damping": args.idea_damping,
        "idea_sigma": args.idea_sigma,
        "idea_epsilon": args.idea_epsilon,
        "idea_delta": args.idea_delta,
        "idea_iterations": args.idea_iterations,
        "idea_hessian_samples": args.idea_hessian_samples,
        "seif_erase_std": args.seif_erase_std,
        "seif_erase_std_final": args.seif_erase_std_final,
        "seif_repair_epochs": args.seif_repair_epochs,
        "seif_forget_class_weight": args.seif_forget_class_weight,
        "seif_learning_rate": args.seif_learning_rate,
        "seif_momentum": args.seif_momentum,
        "seif_weight_decay": args.seif_weight_decay,
        "ruli_privacy_evaluation": args.ruli_privacy_evaluation,
        "ruli_privacy_beta_threshold": args.ruli_privacy_beta_threshold,
        "ruli_privacy_n_population_samples": args.ruli_privacy_n_population_samples,
        "ruli_privacy_shadow_models_dir": args.ruli_privacy_shadow_models_dir,
        "ruli_privacy_k": args.ruli_privacy_k,
        "ruli_privacy_d_target_seed": args.ruli_privacy_d_target_seed if args.ruli_privacy_d_target_seed is not None else args.seed,
        "ruli_privacy_unlearning_algorithm": args.ruli_privacy_unlearning_algorithm,
    }
    if args.gif_retain_samples is not None:
        config_dict["gif_retain_samples"] = args.gif_retain_samples
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs

    if args.spam:
        base_model_path = f"./saved/model_{args.model}_seed_{args.seed}_dataset_{args.dataset}_unlearning_fraction_{args.unlearning_fraction}_n_target_items_{args.n_target_items}_best.pth"
    else:
        base_model_path = f"./saved/model_{args.model}_seed_{args.seed}_dataset_{args.dataset}_best.pth"

    unlearn_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
        saved=True,
        unlearning_algorithm=args.unlearning_algorithm,
        max_norm=args.max_norm,
        base_model_path=base_model_path,
        kookmin_init_rate=args.kookmin_init_rate,
        spam=args.spam,
        unlearning_batchsize=args.unlearning_batchsize,
        damping=args.damping,
        gif_damping=args.gif_damping,
        gif_scale_factor=args.gif_scale_factor,
        gif_iterations=args.gif_iterations,
        gif_k_hops=args.gif_k_hops,
        gif_retain_samples=args.gif_retain_samples,
        ceu_lambda=args.ceu_lambda,
        ceu_sigma=args.ceu_sigma,
        ceu_epsilon=args.ceu_epsilon,
        ceu_cg_iterations=args.ceu_cg_iterations,
        ceu_hessian_samples=args.ceu_hessian_samples,
        idea_damping=args.idea_damping,
        idea_sigma=args.idea_sigma,
        idea_epsilon=args.idea_epsilon,
        idea_delta=args.idea_delta,
        idea_iterations=args.idea_iterations,
        idea_hessian_samples=args.idea_hessian_samples,
        seif_erase_std=args.seif_erase_std,
        seif_erase_std_final=args.seif_erase_std_final,
        seif_repair_epochs=args.seif_repair_epochs,
        seif_forget_class_weight=args.seif_forget_class_weight,
        seif_learning_rate=args.seif_learning_rate,
        seif_momentum=args.seif_momentum,
        seif_weight_decay=args.seif_weight_decay,
        max_training_hours=args.max_training_hours,
    )

if __name__ == "__main__":
    main()