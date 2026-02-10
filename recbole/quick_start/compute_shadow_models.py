"""
Shadow Model Computation Pipeline for RULI Privacy Evaluation

This module provides functionality to compute and save shadow models independently
from the evaluation pipeline. Shadow models are trained on different data partitions
and can be reused across multiple evaluation runs.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataloader import UnlearnTrainDataLoader
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.quick_start.quick_start import k_subsets_exact_np


logger = getLogger()


def _get_eligible_users(dataset_obj, min_interactions=4):
    """
    Get users that have at least min_interactions interactions remaining.
    This ensures we only create shadow models for users with sufficient data.
    
    Args:
        dataset_obj: Dataset object with filtered interactions
        min_interactions: Minimum number of interactions required per user
        
    Returns:
        numpy array of eligible user IDs (internal IDs)
    """
    uid_field = dataset_obj.uid_field
    user_ids = dataset_obj.inter_feat[uid_field].to_numpy()
    
    # Count interactions per user
    unique_users, counts = np.unique(user_ids, return_counts=True)
    
    # Filter to users with >= min_interactions
    eligible_mask = counts >= min_interactions
    eligible_users = unique_users[eligible_mask]
    
    logger.info(f"Found {len(eligible_users)} eligible users (>= {min_interactions} interactions) "
               f"out of {len(unique_users)} total users")
    
    return eligible_users


def compute_shadow_models(
    model: str,
    dataset: str,
    config_file_list: Optional[List[str]] = None,
    config_dict: Optional[Dict] = None,
    k: int = 8,
    model_dir: str = "./models",
    gpu_id: Optional[int] = None,
    saved: bool = True,
    sensitive_category: Optional[str] = None,
    unlearning_fraction: Optional[float] = None,
    unlearning_sample_selection_seed: Optional[int] = None,
    task_type: str = "CF",
    create_unlearned_models: bool = False,
    unlearning_algorithm: Optional[str] = None,
    unlearning_algorithms: Optional[List[str]] = None,
    max_norm: Optional[float] = None,
    kookmin_init_rate: float = 0.01,
    damping: float = 0.01,
    **unlearning_kwargs,
):
    """
    Compute and save shadow models for RULI Privacy evaluation.
    
    Shadow models are trained on the retain set (dataset excluding forget set),
    then partitioned using k-subsets for MIA evaluation.
    
    Optionally creates unlearned versions of shadow models (Qh distribution)
    by applying unlearning to each trained shadow model.
    
    Args:
        model: Model name
        dataset: Dataset name
        config_file_list: List of config file paths
        config_dict: Configuration dictionary
        k: Number of shadow models to create (must be even)
        model_dir: Directory to save shadow models
        gpu_id: GPU ID to use
        saved: Whether to save models
        sensitive_category: Sensitive category for forget set (if applicable)
        unlearning_fraction: Unlearning fraction for forget set (if applicable)
        unlearning_sample_selection_seed: Seed for forget set selection (if applicable)
        task_type: Task type (CF, SBR, or NBR)
        create_unlearned_models: Whether to create unlearned versions of shadow models (Qh)
        unlearning_algorithm: Single unlearning algorithm to use (deprecated, use unlearning_algorithms instead)
        unlearning_algorithms: List of unlearning algorithms to use (creates unlearned models for each)
        max_norm: Max norm for SCIF algorithm
        kookmin_init_rate: Initial rate for Kookmin algorithm
        damping: Damping parameter for GIF/CEU/IDEA algorithms
        **unlearning_kwargs: Additional arguments for unlearning algorithm
    
    Returns:
        Tuple of (saved_models, metadata, all_saved_unlearned_models) where:
        - saved_models: List of regular shadow model paths
        - metadata: Metadata dictionary
        - all_saved_unlearned_models: Dict mapping algorithm -> list of unlearned model paths
    """
    if k % 2 != 0:
        raise ValueError(f"k (number of shadow models) must be even, got {k}")
    
    # Initialize config
    if config_dict is None:
        config_dict = {}
    
    config_dict.update({
        "model": model,
        "dataset": dataset,
        "task_type": task_type,
    })
    if gpu_id is not None:
        config_dict["gpu_id"] = gpu_id
    
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(f"Computing {k} shadow models for {model} on {dataset}")
    
    # Load full dataset
    dataset_obj = create_dataset(config)
    logger.info(dataset_obj)
    
    # Remove forget set if provided (shadow models should be trained on retain set)
    if sensitive_category is not None and unlearning_fraction is not None:
        logger.info(f"Removing forget set (sensitive_category={sensitive_category}, "
                   f"unlearning_fraction={unlearning_fraction})")
        
        # Load forget set
        unlearning_samples_path = os.path.join(
            config["data_path"],
            f"{dataset}_unlearn_pairs_sensitive_category_{sensitive_category}"
            f"_seed_{unlearning_sample_selection_seed or config['seed']}"
            f"_unlearning_fraction_{float(unlearning_fraction)}.inter"
        )
        
        if not os.path.exists(unlearning_samples_path):
            raise FileNotFoundError(
                f"Forget set not found: {unlearning_samples_path}. "
                f"Please generate forget set first."
            )
        
        # Load forget set based on task type
        uid_field = dataset_obj.uid_field
        iid_field = dataset_obj.iid_field
        
        if config["task_type"] == "CF":
            forget_set_df = pd.read_csv(
                unlearning_samples_path,
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"],
                header=0,
            )
        elif config["task_type"] == "SBR":
            with open(unlearning_samples_path, 'r') as f:
                header_line = f.readline().strip()
            column_names = [col.split(':')[0] for col in header_line.split('\t')]
            forget_set_df = pd.read_csv(
                unlearning_samples_path,
                sep="\t",
                names=column_names,
                header=0,
            )
        else:
            raise ValueError(f"Unsupported task_type for shadow models: {config['task_type']}")
        
        logger.info(f"Loaded forget set with {len(forget_set_df)} interactions")
        
        # Create lookup set for forget interactions
        # Get user_ids and item_ids from dataset (these are already internal IDs)
        user_ids = dataset_obj.inter_feat[uid_field].to_numpy()
        item_ids = dataset_obj.inter_feat[iid_field].to_numpy()
        
        forget_pairs = set()
        for _, row in forget_set_df.iterrows():
            try:
                user_token = str(row[uid_field])
                item_token = str(row[iid_field])
                user_id = dataset_obj.token2id(uid_field, user_token)
                item_id = dataset_obj.token2id(iid_field, item_token)
                # Store as tuples with consistent types
                forget_pairs.add((int(user_id), int(item_id)))
            except (ValueError, KeyError):
                pass
        
        logger.info(f"Created forget_pairs set with {len(forget_pairs)} unique (user_id, item_id) pairs")
        
        # Remove forget set interactions from dataset
        # Create boolean mask: True for interactions to keep (not in forget set)
        user_ids_int = user_ids.astype(int)
        item_ids_int = item_ids.astype(int)
        
        # Create boolean mask: True for interactions to keep (not in forget set)
        removed_mask = np.array([
            (int(user_ids_int[i]), int(item_ids_int[i])) not in forget_pairs
            for i in range(len(user_ids_int))
        ])
        
        n_forget_removed = (~removed_mask).sum()
        dataset_obj = dataset_obj.copy(dataset_obj.inter_feat[removed_mask])
        logger.info(f"Removed {n_forget_removed} forget interactions (out of {len(forget_pairs)} in forget set). "
                   f"Retain set has {len(dataset_obj.inter_feat)} interactions")
    
    # Get eligible users (>= 4 interactions) and create k subsets
    # Only use eligible users to ensure each partition has sufficient data
    uid_field = dataset_obj.uid_field
    eligible_users = _get_eligible_users(dataset_obj, min_interactions=4)
    
    if len(eligible_users) == 0:
        raise ValueError(
            "No eligible users found with >= 4 interactions after filtering. "
            "Cannot create shadow models."
        )
    
    logger.info(f"Creating {k} user subsets from {len(eligible_users)} eligible users")
    user_subsets = k_subsets_exact_np(eligible_users, k=k)
    logger.info(f"Created {k} user subsets using k_subsets_exact_np")
    
    # Create shadow model directory structure: ./saved/shadow_models/{dataset}/{model}/
    # This matches the plan specification and avoids conflicts between different datasets/models
    shadow_models_base_dir = os.path.join("saved", "shadow_models")
    shadow_models_dir = os.path.join(shadow_models_base_dir, dataset, model)
    os.makedirs(shadow_models_dir, exist_ok=True)
    
    saved_models = []
    # Track unlearned models per algorithm: {algorithm: [model_paths]}
    all_saved_unlearned_models = {}
    metadata = {
        "model": model,
        "dataset": dataset,
        "seed": config["seed"],
        "k": k,
        "create_unlearned_models": create_unlearned_models,
        "unlearning_algorithms": [],
        "partitions": []
    }
    
    # Determine which algorithms to use
    if create_unlearned_models:
        if unlearning_algorithms is not None:
            algorithms_to_use = unlearning_algorithms
        elif unlearning_algorithm is not None:
            # Backward compatibility: single algorithm
            algorithms_to_use = [unlearning_algorithm]
        else:
            # Default to scif
            algorithms_to_use = ["scif"]
        metadata["unlearning_algorithms"] = algorithms_to_use
        logger.info(f"Will create unlearned shadow models for algorithms: {', '.join(algorithms_to_use)}")
    else:
        algorithms_to_use = []
    
    # Store forget set for unlearning if needed
    forget_set_for_unlearning = None
    if create_unlearned_models:
        if sensitive_category is None or unlearning_fraction is None:
            raise ValueError(
                "create_unlearned_models=True requires sensitive_category and unlearning_fraction"
            )
        # Load forget set for unlearning
        unlearning_samples_path = os.path.join(
            config["data_path"],
            f"{dataset}_unlearn_pairs_sensitive_category_{sensitive_category}"
            f"_seed_{unlearning_sample_selection_seed or config['seed']}"
            f"_unlearning_fraction_{float(unlearning_fraction)}.inter"
        )
        if not os.path.exists(unlearning_samples_path):
            raise FileNotFoundError(
                f"Forget set not found for unlearning: {unlearning_samples_path}"
            )
        
        uid_field = dataset_obj.uid_field
        iid_field = dataset_obj.iid_field
        
        if config["task_type"] == "CF":
            forget_set_for_unlearning = pd.read_csv(
                unlearning_samples_path,
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"],
                header=0,
            )
        elif config["task_type"] == "SBR":
            with open(unlearning_samples_path, 'r') as f:
                header_line = f.readline().strip()
            column_names = [col.split(':')[0] for col in header_line.split('\t')]
            forget_set_for_unlearning = pd.read_csv(
                unlearning_samples_path,
                sep="\t",
                names=column_names,
                header=0,
            )
        else:
            raise ValueError(f"Unsupported task_type for unlearned shadow models: {config['task_type']}")
        
        logger.info(f"Loaded forget set for unlearning: {len(forget_set_for_unlearning)} interactions")
    
    # Log full dataset structure - all shadow models will have this same architecture
    # because dataset.copy() preserves field2id_token mappings
    logger.info(f"All shadow models will have the same architecture: "
               f"{dataset_obj.num(uid_field)} users, "
               f"{dataset_obj.num(iid_field)} items")
    logger.info(f"Each shadow model will be trained on roughly half of the data "
               f"(by excluding a subset of users)")
    
    # Train shadow models for each partition
    for partition_idx in range(k):
        logger.info(f"Training shadow model {partition_idx + 1}/{k}")
        
        # Create dataset excluding users in this partition (OUT model)
        # Using dataset.copy() with a mask preserves the full user/item embedding structure
        # (field2id_token mappings), so all shadow models have the same architecture
        users_to_drop = user_subsets[partition_idx]  # Already a list from k_subsets_exact_np
        # Get user_ids from current dataset_obj (after forget set removal if applicable)
        current_user_ids = dataset_obj.inter_feat[uid_field].to_numpy()
        removed_mask = ~np.isin(current_user_ids, users_to_drop)
        
        # Copy full dataset but mask out interactions we don't want for training
        # This preserves the full user/item embedding structure while training on subset
        partition_dataset = dataset_obj.copy(dataset_obj.inter_feat[removed_mask])
        
        # Verify partition is valid: check that remaining users still have enough interactions
        partition_user_ids = partition_dataset.inter_feat[uid_field].to_numpy()
        partition_unique_users, partition_counts = np.unique(partition_user_ids, return_counts=True)
        users_with_sufficient_data = partition_unique_users[partition_counts >= 4]
        
        if len(users_with_sufficient_data) == 0:
            logger.warning(f"Partition {partition_idx}: No users with >= 4 interactions after exclusion. "
                          f"Skipping this partition.")
            continue
        
        logger.info(f"Partition {partition_idx}: {len(users_to_drop)} users excluded, "
                   f"{len(partition_dataset.inter_feat)} interactions remaining, "
                   f"{len(users_with_sufficient_data)} users with >= 4 interactions")
        
        # Prepare data with same split as main experiment
        train_data, valid_data, test_data = data_preparation(config, partition_dataset)
        
        # Initialize model with partition_dataset - it has the same architecture as full dataset
        # because dataset.copy() preserves field2id_token mappings (same n_users, n_items)
        # but we train on the masked subset (roughly half the data)
        init_seed(config["seed"] + partition_idx, config["reproducibility"])
        shadow_model = get_model(config["model"])(config, partition_dataset).to(config["device"])
        
        # Initialize trainer
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, shadow_model)
        
        # Train model
        logger.info(f"Training shadow model partition {partition_idx}...")
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, verbose=True, saved=saved, show_progress=False
        )
        
        # Save model
        if saved:
            # Include sensitive_category and unlearning_fraction in filename to avoid conflicts
            # when training shadow models with different forget sets
            filename_parts = [
                f"shadow_model_{model}",
                f"seed_{config['seed']}",
                f"dataset_{dataset}",
            ]
            if sensitive_category is not None:
                filename_parts.append(f"category_{sensitive_category}")
            if unlearning_fraction is not None:
                filename_parts.append(f"unlearning_fraction_{unlearning_fraction}")
            filename_parts.append(f"partition_{partition_idx}")
            model_filename = "_".join(filename_parts) + ".pth"
            model_path = os.path.join(shadow_models_dir, model_filename)
            
            # Save checkpoint
            checkpoint = {
                "state_dict": shadow_model.state_dict(),
                "other_parameter": shadow_model.other_parameter(),
                "config": config,
                "partition_idx": partition_idx,
            }
            torch.save(checkpoint, model_path)
            saved_models.append(model_path)
            logger.info(f"Saved shadow model to {model_path}")
            
            # Store metadata
            partition_metadata = {
                "partition_idx": partition_idx,
                "excluded_users": users_to_drop,
                "n_excluded_users": len(users_to_drop),
                "model_path": model_path,
                "best_valid_score": best_valid_score,
            }
            
            # Create unlearned versions for all requested algorithms
            if create_unlearned_models and forget_set_for_unlearning is not None:
                partition_unlearned_paths = {}
                
                for algo in algorithms_to_use:
                    logger.info(f"Creating unlearned shadow model for partition {partition_idx} using {algo}...")
                    try:
                        # Prepare algorithm-specific kwargs
                        algo_kwargs = {}
                        if algo == "scif" and max_norm is not None:
                            algo_kwargs["max_norm"] = max_norm
                        elif algo == "kookmin":
                            algo_kwargs["kookmin_init_rate"] = kookmin_init_rate
                        elif algo in ["gif", "ceu", "idea"]:
                            algo_kwargs["damping"] = damping
                        
                        # Merge with any additional kwargs
                        algo_kwargs.update(unlearning_kwargs)
                        
                        # Create a fresh copy of the shadow model for this algorithm
                        # partition_dataset has the same architecture as full dataset (same n_users, n_items)
                        # because dataset.copy() preserves field2id_token mappings
                        algo_unlearned_model = get_model(config["model"])(
                            config, partition_dataset
                        ).to(config["device"])
                        algo_unlearned_model.load_state_dict(shadow_model.state_dict())
                        algo_unlearned_model.load_other_parameter(shadow_model.other_parameter())
                        
                        # Create trainer for this model
                        algo_trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, algo_unlearned_model)
                        
                        unlearned_model = _create_unlearned_shadow_model(
                            shadow_model=algo_unlearned_model,
                            trainer=algo_trainer,
                            partition_dataset=partition_dataset,
                            forget_set_df=forget_set_for_unlearning,
                            config=config,
                            partition_idx=partition_idx,
                            unlearning_algorithm=algo,
                            **algo_kwargs
                        )
                        
                        # Save unlearned model
                        # Include sensitive_category and unlearning_fraction in filename to avoid conflicts
                        filename_parts = [
                            f"shadow_model_unlearned_{model}",
                            f"seed_{config['seed']}",
                            f"dataset_{dataset}",
                        ]
                        if sensitive_category is not None:
                            filename_parts.append(f"category_{sensitive_category}")
                        if unlearning_fraction is not None:
                            filename_parts.append(f"unlearning_fraction_{unlearning_fraction}")
                        filename_parts.extend([
                            f"partition_{partition_idx}",
                            f"algorithm_{algo}"
                        ])
                        unlearned_model_filename = "_".join(filename_parts) + ".pth"
                        unlearned_model_path = os.path.join(shadow_models_dir, unlearned_model_filename)
                        
                        checkpoint = {
                            "state_dict": unlearned_model.state_dict(),
                            "other_parameter": unlearned_model.other_parameter(),
                            "config": config,
                            "partition_idx": partition_idx,
                            "unlearning_algorithm": algo,
                        }
                        torch.save(checkpoint, unlearned_model_path)
                        
                        # Track per algorithm
                        if algo not in all_saved_unlearned_models:
                            all_saved_unlearned_models[algo] = []
                        all_saved_unlearned_models[algo].append(unlearned_model_path)
                        partition_unlearned_paths[algo] = unlearned_model_path
                        
                        logger.info(f"Saved unlearned shadow model ({algo}) to {unlearned_model_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create unlearned shadow model ({algo}) for partition {partition_idx}: {e}")
                        logger.warning(f"Continuing without {algo} unlearned model for this partition")
                
                # Store all unlearned model paths in metadata
                if partition_unlearned_paths:
                    partition_metadata["unlearned_model_paths"] = partition_unlearned_paths
                    # For backward compatibility, also store first algorithm's path
                    first_algo = list(partition_unlearned_paths.keys())[0]
                    partition_metadata["unlearned_model_path"] = partition_unlearned_paths[first_algo]
            
            metadata["partitions"].append(partition_metadata)
    
    # Save metadata
    # Include dataset, model, sensitive_category, and unlearning_fraction in metadata filename to avoid conflicts
    metadata_path = None
    if saved:
        metadata_filename_parts = [
            f"shadow_models_metadata_{model}",
            f"seed_{config['seed']}",
            f"dataset_{dataset}",
        ]
        if sensitive_category is not None:
            metadata_filename_parts.append(f"category_{sensitive_category}")
        if unlearning_fraction is not None:
            metadata_filename_parts.append(f"unlearning_fraction_{unlearning_fraction}")
        metadata_filename = "_".join(metadata_filename_parts) + ".json"
        metadata_path = os.path.join(shadow_models_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved shadow models metadata to {metadata_path}")
        metadata["metadata_path"] = metadata_path
    
    logger.info(f"Completed computing {k} shadow models")
    if create_unlearned_models:
        total_unlearned = sum(len(models) for models in all_saved_unlearned_models.values())
        logger.info(f"Created {total_unlearned} unlearned shadow models across {len(all_saved_unlearned_models)} algorithms:")
        for algo, models in all_saved_unlearned_models.items():
            logger.info(f"  {algo}: {len(models)} models")
    
    return saved_models, metadata, all_saved_unlearned_models


def _create_unlearned_shadow_model(
    shadow_model: torch.nn.Module,
    trainer,
    partition_dataset,
    forget_set_df,
    config,
    partition_idx: int,
    unlearning_algorithm: str = "scif",
    **unlearning_kwargs
) -> torch.nn.Module:
    """
    Create an unlearned version of a shadow model.
    
    This applies unlearning to a trained shadow model to create the "held-out" distribution (Qh).
    
    Args:
        shadow_model: Trained shadow model to unlearn
        trainer: Trainer instance
        partition_dataset: Dataset used for training the shadow model
        forget_set_df: DataFrame containing forget set interactions
        config: Configuration object
        partition_idx: Partition index
        unlearning_algorithm: Unlearning algorithm to use
        **unlearning_kwargs: Additional arguments for unlearning algorithm
    
    Returns:
        Unlearned shadow model
    """
    logger = getLogger()
    
    # Create a copy of the model for unlearning
    unlearned_model = shadow_model
    
    # Prepare forget data
    # Filter forget set to only include interactions that exist in partition_dataset
    uid_field = partition_dataset.uid_field
    iid_field = partition_dataset.iid_field
    
    # Get interactions from partition dataset
    partition_user_ids = partition_dataset.inter_feat[uid_field].to_numpy()
    partition_item_ids = partition_dataset.inter_feat[iid_field].to_numpy()
    partition_pairs = set(zip(partition_user_ids.astype(int), partition_item_ids.astype(int)))
    
    # Filter forget set to only include pairs that are in the partition
    forget_pairs_in_partition = []
    for _, row in forget_set_df.iterrows():
        try:
            user_token = str(row[uid_field])
            item_token = str(row[iid_field])
            user_id = partition_dataset.token2id(uid_field, user_token)
            item_id = partition_dataset.token2id(iid_field, item_token)
            if (int(user_id), int(item_id)) in partition_pairs:
                forget_pairs_in_partition.append((int(user_id), int(item_id)))
        except (ValueError, KeyError):
            continue
    
    if len(forget_pairs_in_partition) == 0:
        logger.warning(
            f"No forget pairs found in partition {partition_idx}. "
            f"Returning original model without unlearning."
        )
        return unlearned_model
    
    # Create forget dataset
    user_ids = partition_dataset.inter_feat[uid_field].to_numpy().astype(int)
    item_ids = partition_dataset.inter_feat[iid_field].to_numpy().astype(int)
    
    forget_mask = np.array([
        (int(user_ids[i]), int(item_ids[i])) in forget_pairs_in_partition
        for i in range(len(user_ids))
    ])
    
    forget_dataset = partition_dataset.copy(partition_dataset.inter_feat[forget_mask])
    logger.info(
        f"Partition {partition_idx}: {len(forget_pairs_in_partition)} forget pairs, "
        f"{len(forget_dataset.inter_feat)} forget interactions"
    )
    
    # Create retain dataset (partition dataset minus forget set)
    retain_mask = ~forget_mask
    retain_dataset = partition_dataset.copy(partition_dataset.inter_feat[retain_mask])
    
    # Prepare data loaders
    _, _, _ = data_preparation(config, retain_dataset)
    retain_train_data, _, _ = data_preparation(config, retain_dataset)
    
    # Create forget data loader
    # For unlearning, we need a special data loader
    forget_data = UnlearnTrainDataLoader(
        config=config,
        dataset=forget_dataset,
        sampler=None,
        shuffle=False
    )
    
    # Create clean forget data (same as forget_data for most cases)
    clean_forget_data = forget_data
    
    # Apply unlearning
    logger.info(
        f"Applying {unlearning_algorithm} unlearning to shadow model partition {partition_idx}..."
    )
    
    try:
        trainer.unlearn(
            epoch_idx=0,
            forget_data=forget_data,
            clean_forget_data=clean_forget_data,
            retain_train_data=retain_train_data,
            retain_valid_data=None,
            retain_test_data=None,
            unlearning_algorithm=unlearning_algorithm,
            saved=False,
            show_progress=False,
            verbose=False,
            **unlearning_kwargs
        )
        logger.info(f"Successfully unlearned shadow model partition {partition_idx}")
    except Exception as e:
        logger.error(f"Error during unlearning: {e}")
        raise
    
    return unlearned_model

