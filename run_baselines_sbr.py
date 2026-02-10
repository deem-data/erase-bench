#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run SKNN_SBR and Pop_SBR baseline models on multiple datasets

This script runs the baseline models (SKNN_SBR and Pop_SBR) for session-based
recommendation on the specified datasets: amazon_reviews_books, 30music, and nowp.

Usage:
    python run_baselines_sbr.py --dataset amazon_reviews_books --model SKNN_SBR
    python run_baselines_sbr.py --dataset 30music --model Pop_SBR --gpu_id 1

    # Run all combinations:
    python run_baselines_sbr.py --run_all
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Configuration
DATASETS = ["amazon_reviews_books", "30music", "nowp"]
MODELS = ["SKNN_SBR", "Pop_SBR"]
UNLEARNING_FRACTIONS = [0.0001]
DEFAULT_EPOCHS = 1  # Memory-based models don't need training epochs
DEFAULT_GPU_ID = 0

# Map datasets to their sensitive categories
DATASET_TO_SENSITIVE_CATEGORY = {
    "amazon_reviews_books": "health",
    "30music": "explicit",
    "nowp": "explicit"
}


def run_single_experiment(model, dataset, gpu_id=0, seed=None, config_files=None, additional_args=None):
    """
    Run a single experiment with the given configuration.

    Args:
        model: Model name (SKNN_SBR or Pop_SBR)
        dataset: Dataset name
        gpu_id: GPU device ID
        seed: Random seed
        config_files: Additional config files (space-separated string)
        additional_args: Dictionary of additional arguments
    """
    cmd = [
        "python", "run_recbole.py",
        "--model", model,
        "--dataset", dataset,
        "--gpu_id", str(gpu_id),
        "--task_type", "SBR",  # Session-based recommendation
    ]

    # Add seed if specified
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # Add config files if specified
    if config_files:
        cmd.extend(["--config_files", config_files])

    # Add epochs (1 for memory-based models)
    cmd.extend(["--epochs", str(DEFAULT_EPOCHS)])

    # Add additional arguments
    if additional_args:
        for key, value in additional_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

    # Extract sensitive category and unlearning fraction for display
    sensitive_info = ""
    if additional_args:
        if 'sensitive_category' in additional_args:
            sensitive_info += f" [sensitive: {additional_args['sensitive_category']}]"
        if 'unlearning_fraction' in additional_args:
            sensitive_info += f" [uf: {additional_args['unlearning_fraction']}]"

    print(f"\n{'=' * 80}")
    print(f"Running: {model} on {dataset}{sensitive_info}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Successfully completed: {model} on {dataset}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {model} on {dataset}")
        print(f"Error: {e}\n")
        return False


def run_all_experiments(gpu_id=0, seed=None, config_files=None, unlearning_fractions=None):
    """
    Run all combinations of models, datasets, and unlearning fractions.

    Args:
        gpu_id: GPU device ID
        seed: Random seed
        config_files: Additional config files
        unlearning_fractions: List of unlearning fractions (default: UNLEARNING_FRACTIONS)
    """
    if unlearning_fractions is None:
        unlearning_fractions = UNLEARNING_FRACTIONS

    total = len(MODELS) * len(DATASETS) * len(unlearning_fractions)
    completed = 0
    failed = 0

    print(f"\n{'=' * 80}")
    print(f"Running {total} experiments ({len(MODELS)} models × {len(DATASETS)} datasets × {len(unlearning_fractions)} unlearning fractions)")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Unlearning fractions: {', '.join(map(str, unlearning_fractions))}")
    print(f"{'=' * 80}\n")

    results = []

    for model in MODELS:
        for dataset in DATASETS:
            # Get sensitive category for this dataset
            sensitive_category = DATASET_TO_SENSITIVE_CATEGORY.get(dataset)

            for unlearning_fraction in unlearning_fractions:
                # Prepare additional arguments
                additional_args = {}
                if sensitive_category:
                    additional_args['sensitive_category'] = sensitive_category
                if unlearning_fraction is not None:
                    additional_args['unlearning_fraction'] = unlearning_fraction

                success = run_single_experiment(
                    model=model,
                    dataset=dataset,
                    gpu_id=gpu_id,
                    seed=seed,
                    config_files=config_files,
                    additional_args=additional_args
                )

                results.append({
                    'model': model,
                    'dataset': dataset,
                    'sensitive_category': sensitive_category,
                    'unlearning_fraction': unlearning_fraction,
                    'success': success
                })

                if success:
                    completed += 1
                else:
                    failed += 1

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total experiments: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"\nDetailed results:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        sensitive_info = f"(sensitive: {r['sensitive_category']}, uf: {r['unlearning_fraction']})" if r['sensitive_category'] else ""
        print(f"  {status} {r['model']:15s} on {r['dataset']:30s} {sensitive_info}")
    print(f"{'=' * 80}\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run SKNN_SBR and Pop_SBR baseline models on multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single model on a single dataset
  python run_baselines_sbr.py --model SKNN_SBR --dataset amazon_reviews_books

  # Run all combinations
  python run_baselines_sbr.py --run_all

  # Run all with specific GPU and seed
  python run_baselines_sbr.py --run_all --gpu_id 1 --seed 2023

  # Run with custom config file
  python run_baselines_sbr.py --model Pop_SBR --dataset 30music --config_files "config.yaml"
"""
    )

    # Main arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=MODELS,
        help="Model to run (SKNN_SBR or Pop_SBR)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=DATASETS,
        help="Dataset to use"
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all model-dataset combinations"
    )

    # Optional arguments
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=DEFAULT_GPU_ID,
        help=f"GPU device ID (default: {DEFAULT_GPU_ID})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--config_files",
        type=str,
        default=None,
        help="Additional config files (space-separated)"
    )
    parser.add_argument(
        "--sensitive_category",
        type=str,
        default=None,
        help="Sensitive category for evaluation (will override dataset default)"
    )
    parser.add_argument(
        "--unlearning_fraction",
        type=float,
        default=None,
        help="Unlearning fraction for evaluation (e.g., 0.0001)"
    )
    parser.add_argument(
        "--unlearning_fractions",
        type=str,
        default=None,
        help="Comma-separated list of unlearning fractions for --run_all mode (e.g., '0.0001,0.001')"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.run_all:
        # Parse unlearning fractions if provided
        unlearning_fractions = UNLEARNING_FRACTIONS
        if args.unlearning_fractions:
            unlearning_fractions = [float(x.strip()) for x in args.unlearning_fractions.split(',')]

        # Run all combinations
        success = run_all_experiments(
            gpu_id=args.gpu_id,
            seed=args.seed,
            config_files=args.config_files,
            unlearning_fractions=unlearning_fractions
        )
        sys.exit(0 if success else 1)

    elif args.model and args.dataset:
        # Prepare additional arguments
        additional_args = {}

        # Get sensitive category (from arg or dataset default)
        sensitive_category = args.sensitive_category
        if not sensitive_category:
            sensitive_category = DATASET_TO_SENSITIVE_CATEGORY.get(args.dataset)

        if sensitive_category:
            additional_args['sensitive_category'] = sensitive_category
        if args.unlearning_fraction is not None:
            additional_args['unlearning_fraction'] = args.unlearning_fraction

        # Run single experiment
        success = run_single_experiment(
            model=args.model,
            dataset=args.dataset,
            gpu_id=args.gpu_id,
            seed=args.seed,
            config_files=args.config_files,
            additional_args=additional_args
        )
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        print("\nError: Either specify --run_all or provide both --model and --dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()
