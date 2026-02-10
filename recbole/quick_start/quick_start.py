# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import sys
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger
import collections
import json
import pickle
from pathlib import Path

from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.interaction import Interaction
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
    load_model_from_path,
    parse_hf_model_path,
)

import pandas as pd
import os
import torch
import numpy as np
import gc
import time
import tempfile
import glob
import re
import traceback
from copy import deepcopy


def save_sensitive_results_to_csv(sensitive_results, config, retrain_flag=False, unlearning_algorithm=None):
    """
    Save sensitive item evaluation results to a CSV file.
    
    Args:
        sensitive_results (list): List of dictionaries containing sensitive item evaluation results
        config (Config): Configuration object with model/dataset info
        retrain_flag (bool): Whether this is a retraining evaluation
        unlearning_algorithm (str): Name of the unlearning algorithm (for unlearning models)
    """
    if not sensitive_results or len(sensitive_results) == 0:
        return
    
    # Flatten results for CSV
    csv_rows = []
    
    for result_dict in sensitive_results:
        # Base metadata
        base_row = {
            "dataset": config["dataset"],
            "model": config["model"],
            "seed": config.get("seed", None),
            "sensitive_category": result_dict.get("sensitive_category", config.get("sensitive_category", None)),
            "checkpoint_idx": result_dict.get("checkpoint_idx", None),
            "eval_type": "retraining" if result_dict.get("is_retrained", False) else "unlearning" if result_dict.get("is_unlearned", False) else "original",
            "unlearning_algorithm": unlearning_algorithm if result_dict.get("is_unlearned", False) else None,
            "unlearning_fraction": config.get("unlearning_fraction", None),
            "topk": result_dict.get("topk", None),
        }
        
        # Sensitive item metrics
        base_row.update({
            "users_with_sensitive": result_dict.get("users_with_sensitive_in_topk", 0),
            "total_users": result_dict.get("total_unlearned_users", 0),
            "pct_users_with_sensitive": result_dict.get("pct_users_with_sensitive", 0.0),
            "avg_sensitive_per_user": result_dict.get("avg_sensitive_per_user", 0.0),
            "min_sensitive_per_user": result_dict.get("min_sensitive_per_user", 0),
            "max_sensitive_per_user": result_dict.get("max_sensitive_per_user", 0),
            "total_sensitive_in_topk": result_dict.get("total_sensitive_in_topk", 0),
        })
        
        csv_rows.append(base_row)
    
    # Create DataFrame
    df = pd.DataFrame(csv_rows)
    
    # Construct output filename
    model_name = config["model"]
    dataset_name = config["dataset"]
    seed = config.get("seed", "unknown")
    sensitive_category = config.get("sensitive_category", "unknown")
    
    if retrain_flag:
        eval_type_str = "retrain"
        checkpoint_idx = config.get("retrain_checkpoint_idx_to_match", "unknown")
        output_file = f"sensitive_eval_{dataset_name}_{model_name}_seed_{seed}_category_{sensitive_category}_checkpoint_{checkpoint_idx}_{eval_type_str}.csv"
    elif unlearning_algorithm:
        eval_type_str = "unlearn"
        checkpoint_idx = config.get("retrain_checkpoint_idx_to_match", "unknown")
        output_file = f"sensitive_eval_{dataset_name}_{model_name}_seed_{seed}_category_{sensitive_category}_checkpoint_{checkpoint_idx}_{eval_type_str}_{unlearning_algorithm}.csv"
    else:
        output_file = f"sensitive_eval_{dataset_name}_{model_name}_seed_{seed}_category_{sensitive_category}_original.csv"
    
    # Save to CSV
    output_path = os.path.join(".", output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\nSensitive evaluation results saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    
    return output_path


def run(
    model,
    dataset,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
    retrain_flag=False,
    unlearning_fraction=None,
    unlearning_sample_selection_method=None,
    retrain_checkpoint_idx_to_match=None,
    spam=False,
    sensitive_category=None,
):
    if nproc == 1 and world_size <= 0:
        res = run_recbole(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
            retrain_flag=retrain_flag,
            unlearning_fraction=unlearning_fraction,
            unlearning_sample_selection_method=unlearning_sample_selection_method,
            retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match,
            spam=spam,
            sensitive_category=sensitive_category,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            run_recboles,
            args=(model, dataset, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def k_subsets_exact_np(user_list, k=8):
    """
    Efficiently create k subsets where each sample appears in exactly k/2 subsets.
    Optimized for when k << n.
    
    Args:
        samples: list or array of samples
        k: number of subsets (must be even)
    
    Returns:
        list of k subsets
    """

    user_list = np.asarray(user_list)

    old_state = np.random.get_state()
    np.random.seed(2)

    assert k % 2 == 0, f"k (number of total reference models) must be even. currently k = {k}"
    
    n = len(user_list)
    half_k = k // 2
        
    inclusion_matrix = np.zeros((n, k), dtype=bool)
    random_selections = np.random.rand(n, k).argsort(axis=1)[:, :half_k]
    
    rows = np.arange(n)[:, None]
    inclusion_matrix[rows, random_selections] = True
    
    subsets = [user_list[inclusion_matrix[:, j]].tolist() for j in range(k)]
    
    np.random.set_state(old_state)

    return subsets
    
    


def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
    retrain_flag=False,
    unlearning_fraction=None,
    unlearning_sample_selection_method=None,
    retrain_checkpoint_idx_to_match=None,
    spam=False,
    sensitive_category=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """

    total_start_time = time.time()

    # configurations initialization
    # Set default topk to [10, 20] if not specified
    if config_dict is None:
        config_dict = {}
    if 'topk' not in config_dict:
        config_dict['topk'] = [10, 20]

    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering - always load base dataset
    # Note: fraud sessions are now injected inside create_dataset() before ID remapping
    dataset = create_dataset(config, unlearning=False, spam=spam)

    logger.info(dataset)

    # Keep reference to original dataset for consistent ID mappings
    original_dataset = dataset

    # Remove forget set if retraining OR if evaluating an unlearned model (need to use retain dataset)
    remove_forget_set = retrain_flag or (
        unlearning_fraction is not None and
        unlearning_sample_selection_method is not None and
        retrain_checkpoint_idx_to_match is not None
    )
    
    if remove_forget_set:
        # remove unlearned interactions
        if "spam" in config and config["spam"]:
            # Use new naming convention (bandwagon_unpopular_ratio) - consistent with dataset loading
            # For NBR: use _fraud_baskets_ with .json, for SBR/CF: use _fraud_sessions_ with .inter
            if config.task_type == "NBR":
                unlearning_samples_path = os.path.join(
                    config["data_path"],
                    f"{config['dataset']}_fraud_baskets_bandwagon_unpopular_ratio_{config['unlearning_fraction']}_seed_{config['unlearn_sample_selection_seed']}.json"
                )
            else:
                unlearning_samples_path = os.path.join(
                    config["data_path"],
                    f"{config['dataset']}_fraud_sessions_bandwagon_unpopular_ratio_{config['unlearning_fraction']}_seed_{config['unlearn_sample_selection_seed']}.inter"
                )
        elif "sensitive_category" in config and config['sensitive_category'] is not None:
            # Handle sensitive category unlearning
            sensitive_category = config['sensitive_category']
            unlearning_samples_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_unlearn_pairs_sensitive_category_{sensitive_category}"
                f"_seed_{config['unlearn_sample_selection_seed']}"
                f"_unlearning_fraction_{float(config['unlearning_fraction'])}.inter"
            )
        else:
            unlearning_samples_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_unlearn_pairs_{config['unlearning_sample_selection_method']}"
                f"_seed_{config['unlearn_sample_selection_seed']}"
                f"_unlearning_fraction_{float(config['unlearning_fraction'])}.inter"
            )

        # Load unlearning samples based on task type
        if config.task_type == "SBR":
            # Read header to infer column names dynamically
            with open(unlearning_samples_path, 'r') as f:
                header_line = f.readline().strip()

            # Extract column names from header (e.g., "user_id:token" -> "user_id")
            column_names = [col.split(':')[0] for col in header_line.split('\t')]

            unlearning_samples = pd.read_csv(
                unlearning_samples_path,
                sep="\t",
                names=column_names,
                header=0,
            )
        elif config.task_type == "CF":
            unlearning_samples = pd.read_csv(
                unlearning_samples_path,
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"],
                header=0,
            )
        elif config.task_type == "NBR":
            # For NBR, we need to handle this differently:
            # 1. Load user IDs from pickle/json file (not user-item pairs)
            # 2. Reload dataset with sensitive items removed from baskets
            # This is handled in the NBR-specific dataset recreation below
            pass

        # For NBR task, we need to recreate the dataset with cleaned baskets
        if config.task_type == "NBR":
            # Construct path to unlearning user IDs file (pkl or json)
            # Handle both .inter and .json base file extensions
            if unlearning_samples_path.endswith('.json'):
                unlearning_users_path_pkl = unlearning_samples_path.replace('.json', '.pkl')
                unlearning_users_path_json = unlearning_samples_path  # Already .json
            else:
                unlearning_users_path_pkl = unlearning_samples_path.replace('.inter', '.pkl')
                unlearning_users_path_json = unlearning_samples_path.replace('.inter', '.json')

            # Try to load user IDs from pickle or json
            unlearning_user_ids = None
            if os.path.exists(unlearning_users_path_pkl):
                logger.info(f"Loading NBR unlearning user IDs from: {unlearning_users_path_pkl}")
                with open(unlearning_users_path_pkl, 'rb') as f:
                    unlearning_user_ids = pickle.load(f)
            elif os.path.exists(unlearning_users_path_json):
                logger.info(f"Loading NBR unlearning user IDs from: {unlearning_users_path_json}")
                with open(unlearning_users_path_json, 'r') as f:
                    unlearning_user_ids = json.load(f)
            else:
                raise FileNotFoundError(
                    f"NBR unlearning user IDs file not found. Tried:\n"
                    f"  - {unlearning_users_path_pkl}\n"
                    f"  - {unlearning_users_path_json}\n"
                    f"Please create unlearning sets using create_nbr_unlearning_sets_user_ids.py"
                )

            # Apply checkpoint filtering if needed
            if retrain_checkpoint_idx_to_match is not None and unlearning_user_ids:
                unlearning_checkpoints = [
                    len(unlearning_user_ids) // 4,
                    len(unlearning_user_ids) // 2,
                    3 * len(unlearning_user_ids) // 4,
                    len(unlearning_user_ids) - 1
                ]
                users_to_unlearn = unlearning_checkpoints[retrain_checkpoint_idx_to_match]
                unlearning_user_ids = sorted(unlearning_user_ids)[:users_to_unlearn + 1]

            logger.info(f"NBR unlearning: {len(unlearning_user_ids)} users to clean")

            # Normalize user IDs to strings for comparison (JSON keys are always strings)
            unlearning_user_ids_set = set(str(uid) for uid in unlearning_user_ids)

            # Now we need to create a cleaned merged JSON file
            # Load the sensitive items to remove
            if "sensitive_category" in config and config['sensitive_category'] is not None:
                sensitive_category = config['sensitive_category']

                # Load sensitive items from file
                sensitive_items_file = os.path.join(
                    config["data_path"],
                    f"sensitive_asins_{sensitive_category}.txt"
                )

                # Try alternative naming conventions
                if not os.path.exists(sensitive_items_file):
                    sensitive_items_file = os.path.join(
                        config["data_path"],
                        f"sensitive_products_{sensitive_category}.txt"
                    )

                if not os.path.exists(sensitive_items_file):
                    raise FileNotFoundError(
                        f"Sensitive items file not found for category '{sensitive_category}'. Tried:\n"
                        f"  - sensitive_asins_{sensitive_category}.txt\n"
                        f"  - sensitive_products_{sensitive_category}.txt"
                    )

                logger.info(f"Loading sensitive items from: {sensitive_items_file}")
                with open(sensitive_items_file, 'r') as f:
                    sensitive_items = set(int(line.strip()) for line in f if line.strip())

                logger.info(f"Loaded {len(sensitive_items)} sensitive items for category '{sensitive_category}'")

                # Create cleaned merged JSON by removing sensitive items from baskets
                original_merged_path = os.path.join(
                    config["data_path"],
                    f"{config['dataset']}_merged.json"
                )

                logger.info(f"Loading original merged data from: {original_merged_path}")
                with open(original_merged_path, 'r') as f:
                    merged_data = json.load(f)
                
                # First, check which sensitive items exist in the original dataset (before removing sensitive items)
                # This ensures we only include items that survive normal filtering (e.g., min_item_inter_num)
                logger.info("Checking which sensitive items exist in original dataset (before removal)...")
                # Create a new config with the original merged file path
                # Use the original config_file_list to get all defaults, then override only the JSON path
                # This applies normal dataset filtering (MIN_NBR_BASKET_COUNT, user_inter_num_interval, etc.)
                original_config = Config(
                    model=model,  # Use function parameter
                    dataset=config["dataset"],  # Use dataset name string, not dataset object
                    config_file_list=config_file_list,  # Use original config files to get all defaults
                    config_dict={
                        "NEXT_BASKET_JSON": f"{config['dataset']}_merged.json",
                    }
                )
                # Create temporary dataset from original data to check item existence
                temp_original_dataset = create_dataset(original_config, unlearning=False, spam=False)
                original_item_tokens = set(temp_original_dataset.field2token_id[temp_original_dataset.iid_field].keys())
                logger.info(f"Original dataset has {len(original_item_tokens)} items after normal filtering")
                
                # Filter sensitive items to only those that exist in original dataset
                # These are items that survive normal filtering (min interactions, k-core, etc.)
                sensitive_items_in_original = []
                for item in sensitive_items:
                    item_token = str(item)
                    if item_token in original_item_tokens:
                        sensitive_items_in_original.append(item)
                
                logger.info(f"Found {len(sensitive_items_in_original)} sensitive items in original dataset (out of {len(sensitive_items)})")
                logger.info(f"These items survive normal dataset filtering and will be used in forget set")
                
                # Store the filtered sensitive items for later use in forget set creation
                # Convert back to set for the cleaning operation
                sensitive_items = set(sensitive_items_in_original)

                logger.info(f"Processing {len(merged_data)} users to create cleaned dataset...")

                # Clean baskets for users in unlearning set
                cleaned_data = {}
                users_filtered_out = 0

                for user_id, baskets in merged_data.items():
                    # user_id from JSON is always a string, so compare as string
                    if str(user_id) in unlearning_user_ids_set:
                        # Remove sensitive items from this user's baskets
                        clean_baskets = []
                        for basket in baskets:
                            clean_basket = [item for item in basket if item not in sensitive_items]
                            if len(clean_basket) > 0:  # Only keep non-empty baskets
                                clean_baskets.append(clean_basket)

                        # Keep user only if they still have >= 4 baskets
                        if len(clean_baskets) >= 4:
                            cleaned_data[user_id] = clean_baskets
                        else:
                            users_filtered_out += 1
                    else:
                        # User not in unlearning set, keep all baskets
                        if len(baskets) >= 4:
                            cleaned_data[user_id] = baskets

                logger.info(f"NBR cleaning: {users_filtered_out} users filtered out (< 4 baskets after removal)")
                logger.info(f"NBR retain dataset: {len(cleaned_data)} users")

                # Save cleaned data to temporary file
                temp_merged_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.json',
                    delete=False,
                    dir=config["data_path"]
                )
                json.dump(cleaned_data, temp_merged_file)
                temp_merged_file.close()

                logger.info(f"Created temporary cleaned merged file: {temp_merged_file.name}")

                # Update config to point to cleaned file
                config["NEXT_BASKET_JSON"] = temp_merged_file.name

                # Recreate dataset with cleaned data
                dataset = create_dataset(config, unlearning=False, spam=spam)
                
                # Store the filtered sensitive items (as list) for later use in forget set creation
                config["_filtered_sensitive_items"] = sensitive_items_in_original

                # Clean up temporary file
                os.unlink(temp_merged_file.name)
                logger.info("Cleaned up temporary merged file")
        else:
            # For CF and SBR, use the original logic
            uid_field, iid_field = original_dataset.uid_field, original_dataset.iid_field

            pairs_by_user = (
                unlearning_samples.groupby(uid_field)[iid_field]
                .agg(list)
                .to_dict()
            )

            unlearning_checkpoints = [len(pairs_by_user) // 4, len(pairs_by_user) // 2, 3 * len(pairs_by_user) // 4, len(pairs_by_user) - 1]
            users_unlearned = unlearning_checkpoints[retrain_checkpoint_idx_to_match]
            removed_mask = np.zeros(len(dataset.inter_feat), dtype=bool)
            user_ids = dataset.inter_feat[uid_field].to_numpy()
            item_ids = dataset.inter_feat[iid_field].to_numpy()

            pairs_by_user_unlearned = sorted(pairs_by_user.items())[:users_unlearned + 1]

            for unlearn_request_idx, (u_token, forget_items_tokens) in enumerate(pairs_by_user_unlearned):
                # Convert tokens to internal IDs using original_dataset to ensure consistent mappings
                try:
                    u_id = original_dataset.token2id(uid_field, str(u_token))
                except ValueError:
                    logger.warning(f"User token {u_token} not found in dataset, skipping")
                    continue

                forget_items_ids = [original_dataset.token2id(iid_field, str(item_token)) for item_token in forget_items_tokens]

                all_idx = np.where(user_ids == u_id)[0]
                mask = np.isin(item_ids[all_idx], forget_items_ids)
                # Mark the forget items (interactions to remove) as True
                removed_mask[all_idx[mask]] = True

            # Keep only interactions that are in the retain set
            dataset = dataset.copy(dataset.inter_feat[~removed_mask])

        print("retain dataset")
        logger.info(dataset)
    
    elif config["rmia_out_model_flag"]:
        uid_field = dataset.uid_field
        user_ids = dataset.inter_feat[uid_field].to_numpy()
        unique_users = np.sort(np.unique(user_ids))
        
        user_subsets = k_subsets_exact_np(unique_users, k=config["rmia_out_model_k"])

        users_to_drop = user_subsets[config["rmia_out_model_partition_idx"]].tolist()
        rmia_removed_mask = ~np.isin(user_ids, list(users_to_drop))

        dataset = dataset.copy(dataset.inter_feat[rmia_removed_mask])

        print(f"rmia dataset for OUT model with idx: {config['rmia_out_model_partition_idx']}")
        logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset, spam=spam)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    
    # Automatically compute inverse frequency weights for Sets2Sets model
    if hasattr(model, 'compute_inverse_freq_weights_from_dataset'):
        logger.info("Computing inverse frequency weights from training data...")
        model.compute_inverse_freq_weights_from_dataset(train_data)
        logger.info("Inverse frequency weights computed.")
    
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # Check if eval_only mode is enabled
    eval_only = "eval_only" in config and config["eval_only"]
    sensitive_eval_only = "sensitive_eval_only" in config and config["sensitive_eval_only"]
    
    if eval_only:
        # Skip training and load model from checkpoint
        logger.info("Eval-only mode: skipping training and loading model from checkpoint")
        
        # Check if a HuggingFace model path is specified
        if "hf_model_path" in config and config["hf_model_path"] is not None:
            # Load from HuggingFace Hub
            model_file_path = config["hf_model_path"]
            logger.info(f"Loading model from HuggingFace: {model_file_path}")
            
            # Use the HF loader utility
            checkpoint = load_model_from_path(
                model_file_path,
                map_location=config["device"],
                hf_token=config.get("hf_token", None),
                cache_dir=config.get("hf_cache_dir", None)
            )
        else:
            # Load from local path (original behavior)
            # Use the trainer's saved_model_file which is already constructed correctly
            model_file_path = trainer.saved_model_file
            
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"Model file not found: {model_file_path}. Expected path: {model_file_path}")
            
            logger.info(f"Loading model from local path: {model_file_path}")
            checkpoint = torch.load(model_file_path, map_location=config["device"])
        
        # Load the state dict into the model
        trainer.model.load_state_dict(checkpoint["state_dict"])
        if "other_parameter" in checkpoint:
            trainer.model.load_other_parameter(checkpoint["other_parameter"])
        trainer.model.eval()
        
        # Set dummy values for best_valid_score and best_valid_result since we didn't train
        best_valid_score = None
        best_valid_result = None
    else:
        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=saved,
            show_progress=False,
            retrain_flag=retrain_flag,
        )

    # model evaluation
    if sensitive_eval_only:
        # Skip normal evaluation, only do sensitive item evaluation later
        logger.info("Sensitive-eval-only mode: skipping normal evaluation metrics")
        test_result = {}
    else:
        test_result = trainer.evaluate(
            test_data, load_best_model=(saved and not eval_only) or eval_only, show_progress=False,
        )

        environment_tb = get_environment(config)
        logger.info(
            "The running environment of this training is as follows:\n"
            + environment_tb.draw()
        )

        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
        logger.info(set_color("test result", "yellow") + f": {test_result}")

    # For spam scenarios, also evaluate on unpoisoned data
    unpoisoned_test_result = None
    if spam and not remove_forget_set and not config["rmia_out_model_flag"] and not sensitive_eval_only:
        logger.info("\n" + "="*50)
        logger.info("Evaluating on unpoisoned (clean) data...")
        logger.info("="*50)

        # Create unpoisoned dataset by removing fraud sessions/baskets
        # Use new naming convention (bandwagon_unpopular_ratio) - consistent with dataset loading
        # For NBR: use _fraud_baskets_ with .json, for SBR/CF: use _fraud_sessions_ with .inter
        if config.task_type == "NBR":
            unlearning_samples_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_fraud_baskets_bandwagon_unpopular_ratio_{config['unlearning_fraction']}_seed_{config['unlearn_sample_selection_seed']}.json"
            )
        else:
            unlearning_samples_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_fraud_sessions_bandwagon_unpopular_ratio_{config['unlearning_fraction']}_seed_{config['unlearn_sample_selection_seed']}.inter"
            )

        if os.path.exists(unlearning_samples_path):
            # Get original dataset interactions
            orig_inter_feat = original_dataset.inter_feat
            uid_field = original_dataset.uid_field

            if config.task_type == "NBR":
                # For NBR, load fraud baskets JSON and extract fraud user IDs
                with open(unlearning_samples_path, 'r') as f:
                    fraud_baskets_data = json.load(f)
                
                # Extract fraud user IDs (keys are user IDs, convert to int for matching)
                fraud_user_ids = set(int(uid) for uid in fraud_baskets_data.keys())
                logger.info(f"Found {len(fraud_user_ids)} fraud users in NBR dataset")
                
                # Get original user IDs
                if hasattr(orig_inter_feat, 'interaction'):
                    # It's an Interaction object
                    orig_user_ids = orig_inter_feat[uid_field].cpu().numpy()
                elif hasattr(orig_inter_feat, 'to_numpy'):
                    # It's a pandas DataFrame
                    orig_user_ids = orig_inter_feat[uid_field].to_numpy()
                else:
                    raise ValueError(f"Unexpected type for orig_inter_feat: {type(orig_inter_feat)}")
                
                # Create mask to remove fraud users
                if hasattr(orig_user_ids, 'item'):
                    # It's a tensor
                    mask = ~torch.isin(orig_user_ids, torch.tensor(list(fraud_user_ids), dtype=orig_user_ids.dtype))
                else:
                    # It's a numpy array
                    mask = ~np.isin(orig_user_ids, list(fraud_user_ids))
            else:
                # For SBR/CF, read fraud sessions from .inter file
                # Read header to infer column names dynamically
                with open(unlearning_samples_path, 'r') as f:
                    header_line = f.readline().strip()

                # Extract column names from header (e.g., "user_id:token" -> "user_id")
                column_names = [col.split(':')[0] for col in header_line.split('\t')]

                spam_sessions_df = pd.read_csv(
                    unlearning_samples_path,
                    sep="\t",
                    names=column_names,
                    header=0,
                )

                # Create mask to remove spam sessions from original dataset
                spam_session_ids = set(spam_sessions_df['session_id'].unique())

                session_field = original_dataset.uid_field  # In SBR, uid_field is session_id

                # Handle both Interaction objects and DataFrames
                if hasattr(orig_inter_feat, 'interaction'):
                    # It's an Interaction object
                    orig_session_ids = orig_inter_feat[session_field].cpu().numpy()
                elif hasattr(orig_inter_feat, 'to_numpy'):
                    # It's a pandas DataFrame
                    orig_session_ids = orig_inter_feat[session_field].to_numpy()
                else:
                    raise ValueError(f"Unexpected type for orig_inter_feat: {type(orig_inter_feat)}")

                # Create mask for unpoisoned data
                unpoisoned_mask = ~pd.Series(orig_session_ids).isin(spam_session_ids).values

                logger.info(f"Original dataset: {len(orig_session_ids)} interactions")
                logger.info(f"Spam sessions: {len(spam_session_ids)} sessions")
                logger.info(f"Unpoisoned dataset: {unpoisoned_mask.sum()} interactions")

            # Create unpoisoned dataset by filtering using the mask
            # For Interaction objects, use indexing; for DataFrames, use iloc
            if config.task_type == "NBR":
                # For NBR, mask is already created above
                if hasattr(orig_inter_feat, 'interaction'):
                    # It's an Interaction object - use boolean indexing
                    unpoisoned_indices = np.where(mask)[0]
                    unpoisoned_inter_feat = orig_inter_feat[unpoisoned_indices]
                elif hasattr(orig_inter_feat, 'iloc'):
                    # It's a DataFrame
                    unpoisoned_inter_feat = orig_inter_feat.iloc[mask]
                else:
                    raise ValueError(f"Unexpected type for orig_inter_feat: {type(orig_inter_feat)}")
            else:
                # For SBR/CF, use the unpoisoned_mask created above
                if hasattr(orig_inter_feat, 'interaction'):
                    # It's an Interaction object - use boolean indexing
                    unpoisoned_indices = np.where(unpoisoned_mask)[0]
                    unpoisoned_inter_feat = orig_inter_feat[unpoisoned_indices]
                elif hasattr(orig_inter_feat, 'iloc'):
                    # It's a DataFrame
                    unpoisoned_inter_feat = orig_inter_feat.iloc[unpoisoned_mask]
                else:
                    logger.warning("Unknown inter_feat type, cannot filter. Skipping unpoisoned evaluation.")
                    unpoisoned_test_result = test_result
                    unpoisoned_inter_feat = None
        else:
            # File doesn't exist, skip unpoisoned evaluation
            logger.warning(f"Fraud file not found: {unlearning_samples_path}. Skipping unpoisoned evaluation.")
            unpoisoned_inter_feat = None
            unpoisoned_test_result = test_result

        if unpoisoned_inter_feat is not None:
                unpoisoned_dataset = original_dataset.copy(unpoisoned_inter_feat)
                logger.info(unpoisoned_dataset)

                # Create dataloaders for unpoisoned dataset
                _, _, unpoisoned_test_data = data_preparation(config, unpoisoned_dataset, spam=False)

                # Evaluate on unpoisoned data
                unpoisoned_test_result = trainer.evaluate(
                    unpoisoned_test_data, load_best_model=(saved and not eval_only) or eval_only, show_progress=False,
                )

                logger.info(set_color("test result (unpoisoned)", "yellow") + f": {unpoisoned_test_result}")
        else:
            logger.warning(f"Spam sessions file not found: {unlearning_samples_path}")
            logger.warning("Skipping unpoisoned evaluation")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    # Add unpoisoned test result if available
    if unpoisoned_test_result is not None:
        result["test_result_unpoisoned"] = unpoisoned_test_result

    # Sensitive item evaluation
    if "sensitive_category" in config and config['sensitive_category'] is not None:
        # Determine if this is a retrained or unlearned model
        is_unlearned_model = (unlearning_fraction is not None and 
                              unlearning_sample_selection_method is not None and 
                              retrain_checkpoint_idx_to_match is not None)
        
        if retrain_flag:
            print("\nSensitive Item Evaluation for Retrained Model")
        elif is_unlearned_model:
            print("\nSensitive Item Evaluation for Unlearned Model")
        else:
            print("\nSensitive Item Evaluation for Original Model")

        sensitive_category = config['sensitive_category']
        sensitive_items_path = os.path.join(config['data_path'], f"sensitive_asins_{sensitive_category}.txt")

        if os.path.exists(sensitive_items_path):
            # For NBR retrain, use pre-filtered sensitive items that were verified against original dataset
            # This ensures we're checking the right items (those that exist before any filtering)
            if config.task_type == "NBR" and "_filtered_sensitive_items" in config:
                sensitive_asins = set(str(item) for item in config["_filtered_sensitive_items"])
                print(f"Using {len(sensitive_asins)} pre-filtered sensitive items (verified against original dataset)")
            else:
                with open(sensitive_items_path, 'r') as f:
                    sensitive_asins = set(line.strip() for line in f if line.strip())
                print(f"Loaded {len(sensitive_asins)} sensitive items from {sensitive_items_path}")

            # Map ASINs to internal item IDs using the cleaned dataset
            # The model predicts on the cleaned dataset, so we need IDs in that space
            # Note: For NBR retrain, sensitive items are removed from user baskets but still exist in vocabulary
            sensitive_item_ids = set()
            iid_field = dataset.iid_field
            for asin in sensitive_asins:
                try:
                    item_id = dataset.token2id(iid_field, asin)
                    sensitive_item_ids.add(item_id)
                except ValueError:
                    # ASIN not in cleaned dataset vocabulary (completely removed or filtered out)
                    pass

            print(f"Mapped to {len(sensitive_item_ids)} sensitive internal item IDs in cleaned dataset (out of {len(sensitive_asins)} ASINs)")
            print(f"[Debug] Sample sensitive item IDs: {sorted(list(sensitive_item_ids))[:10]}")
            print(f"[Debug] Cleaned dataset has {dataset.item_num} items, original_dataset has {original_dataset.item_num} items")

            # Determine which users to evaluate
            # For retrained or unlearned models, evaluate the users that were supposed to be unlearned
            if retrain_flag or (unlearning_fraction is not None and unlearning_sample_selection_method is not None and retrain_checkpoint_idx_to_match is not None):
                # For retrained models, evaluate the users that were supposed to be unlearned
                # Build base path for unlearning samples (without extension)
                unlearning_base_path = os.path.join(
                    config["data_path"],
                    f"{config['dataset']}_unlearn_pairs_sensitive_category_{sensitive_category}"
                    f"_seed_{config['unlearn_sample_selection_seed']}"
                    f"_unlearning_fraction_{float(config['unlearning_fraction'])}"
                )

                if config.task_type == "NBR":
                    # For NBR, forget sets are stored as user ID lists in .pkl or .json
                    unlearning_users_path_pkl = unlearning_base_path + ".pkl"
                    unlearning_users_path_json = unlearning_base_path + ".json"

                    unlearning_user_ids_raw = None
                    if os.path.exists(unlearning_users_path_pkl):
                        logger.info(f"Loading NBR unlearning user IDs from: {unlearning_users_path_pkl}")
                        with open(unlearning_users_path_pkl, "rb") as f:
                            unlearning_user_ids_raw = pickle.load(f)
                    elif os.path.exists(unlearning_users_path_json):
                        logger.info(f"Loading NBR unlearning user IDs from: {unlearning_users_path_json}")
                        with open(unlearning_users_path_json, "r") as f:
                            unlearning_user_ids_raw = json.load(f)
                    else:
                        raise FileNotFoundError(
                            f"NBR unlearning user IDs file not found. Tried:\n"
                            f"  - {unlearning_users_path_pkl}\n"
                            f"  - {unlearning_users_path_json}\n"
                            f"Please create unlearning sets using create_nbr_unlearning_sets_user_ids.py"
                        )

                    # Apply checkpoint-based subset selection if needed
                    checkpoint_idx = (
                        retrain_checkpoint_idx_to_match
                        if retrain_checkpoint_idx_to_match is not None
                        else 3
                    )
                    total_users = len(unlearning_user_ids_raw)
                    unlearning_checkpoints = [
                        total_users // 4,
                        total_users // 2,
                        3 * total_users // 4,
                        total_users - 1,
                    ]
                    users_unlearned = unlearning_checkpoints[checkpoint_idx]

                    # Map the external user tokens to internal user IDs using the cleaned dataset
                    # We need to use the same dataset that we'll get history from for predictions
                    uid_field = dataset.uid_field
                    sorted_users = sorted(unlearning_user_ids_raw)[: users_unlearned + 1]
                    unlearned_user_ids = []
                    skipped_users = []
                    for u in sorted_users:
                        try:
                            user_id = dataset.token2id(uid_field, str(u))
                            unlearned_user_ids.append(user_id)
                        except ValueError:
                            # User token doesn't exist in cleaned dataset, skip
                            skipped_users.append(str(u))
                            continue

                    if skipped_users:
                        logger.warning(
                            f"Skipped {len(skipped_users)} users that don't exist in cleaned dataset: "
                            f"{skipped_users[:5]}{'...' if len(skipped_users) > 5 else ''}"
                        )

                    print(f"\nEvaluating {len(unlearned_user_ids)} users unlearned up to checkpoint {checkpoint_idx}")
                else:
                    # Non-NBR tasks: load unlearning samples from .inter file
                    unlearning_samples_path = unlearning_base_path + ".inter"

                    if config.task_type == "SBR":
                        # Read header to infer column names dynamically
                        with open(unlearning_samples_path, "r") as f:
                            header_line = f.readline().strip()

                        # Extract column names from header (e.g., "user_id:token" -> "user_id")
                        column_names = [
                            col.split(":")[0] for col in header_line.split("\t")
                        ]

                        unlearning_samples_for_eval = pd.read_csv(
                            unlearning_samples_path,
                            sep="\t",
                            names=column_names,
                            header=0,
                        )
                    elif config.task_type == "CF":
                        unlearning_samples_for_eval = pd.read_csv(
                            unlearning_samples_path,
                            sep="\t",
                            names=["user_id", "item_id", "rating", "timestamp"],
                            header=0,
                        )
                    else:
                        unlearning_samples_for_eval = pd.read_csv(
                            unlearning_samples_path,
                            sep="\t",
                            names=["user_id", "item_id", "timestamp"],
                            header=0,
                        )

                    # Use uid_field and iid_field from original_dataset to handle SBR (session_id) vs CF (user_id)
                    uid_field_eval = original_dataset.uid_field
                    iid_field_eval = original_dataset.iid_field
                    pairs_by_user = (
                        unlearning_samples_for_eval.groupby(uid_field_eval)[
                            iid_field_eval
                        ]
                        .agg(list)
                        .to_dict()
                    )

                    # Calculate how many users were unlearned at this checkpoint
                    checkpoint_idx = (
                        retrain_checkpoint_idx_to_match
                        if retrain_checkpoint_idx_to_match is not None
                        else 3
                    )
                    unlearning_checkpoints = [
                        len(pairs_by_user) // 4,
                        len(pairs_by_user) // 2,
                        3 * len(pairs_by_user) // 4,
                        len(pairs_by_user) - 1,
                    ]
                    users_unlearned = unlearning_checkpoints[checkpoint_idx]

                    # Get the actual user IDs that were unlearned
                    # pairs_by_user.keys() contains tokens from CSV (ints or strings depending on dataset)
                    # Convert tokens to strings then to internal IDs using original_dataset to ensure consistent mappings
                    uid_field = original_dataset.uid_field
                    sorted_users = sorted(pairs_by_user.keys())[
                        : users_unlearned + 1
                    ]
                    unlearned_user_ids = []
                    skipped_users = []
                    for u in sorted_users:
                        try:
                            # Use original_dataset to ensure we have full token-to-ID mappings
                            user_id = original_dataset.token2id(uid_field, str(u))
                            unlearned_user_ids.append(user_id)
                        except ValueError:
                            # User token doesn't exist in original dataset, skip
                            skipped_users.append(str(u))
                            continue

                    if skipped_users:
                        logger.warning(
                            f"Skipped {len(skipped_users)} users that don't exist in original dataset: "
                            f"{skipped_users[:5]}{'...' if len(skipped_users) > 5 else ''}"
                        )

                    print(
                        f"\nEvaluating {len(unlearned_user_ids)} users unlearned up to checkpoint {checkpoint_idx}"
                    )
            else:
                # For original models, evaluate all users
                uid_field = dataset.uid_field
                user_ids_data = dataset.inter_feat[uid_field]

                # Handle both pandas Series and PyTorch tensor
                if hasattr(user_ids_data, 'to_numpy'):
                    # pandas Series
                    user_ids = user_ids_data.to_numpy()
                else:
                    # PyTorch tensor
                    user_ids = user_ids_data.cpu().numpy()

                unlearned_user_ids = np.unique(user_ids).tolist()

                print(f"\nEvaluating all {len(unlearned_user_ids)} users in the dataset")

            # Get list of k values to evaluate
            topk_list = config['topk'] if isinstance(config['topk'], list) else [config['topk']]
            max_k = max(topk_list)

            # Get top-max_k predictions once for all users
            all_user_topk_items = {}
            skipped_users_no_interactions = []
            trainer.model.eval()
            # Use test_data._dataset to access the underlying dataset after data_preparation
            eval_dataset = test_data._dataset
            with torch.no_grad():
                for user_id in unlearned_user_ids:
                    # Create interaction for this user based on model type
                    # Check if model is traditional (doesn't need sequences even for SBR)
                    from recbole.utils import ModelType
                    is_traditional = config['MODEL_TYPE'] == ModelType.TRADITIONAL

                    if config['task_type'] == 'SBR' and not is_traditional:
                        # For sequential models, we need to provide item sequence
                        # Get the user's interaction data from the dataset
                        # The dataset.inter_feat is sorted by uid and time, and after augmentation
                        # each row contains the history up to that point
                        user_mask = eval_dataset.inter_feat[eval_dataset.uid_field] == user_id
                        user_indices = torch.where(user_mask)[0]

                        if len(user_indices) == 0:
                            # User has no interactions, skip
                            skipped_users_no_interactions.append(user_id)
                            continue

                        # Get the last interaction which contains the longest/complete sequence
                        last_idx = user_indices[-1].item()

                        # Get the item sequence fields
                        item_seq_field = trainer.model.ITEM_SEQ
                        item_seq_len_field = trainer.model.ITEM_SEQ_LEN

                        interaction = Interaction({
                            eval_dataset.uid_field: torch.tensor([user_id], device=trainer.device),
                            item_seq_field: eval_dataset.inter_feat[item_seq_field][last_idx].unsqueeze(0).to(trainer.device),
                            item_seq_len_field: eval_dataset.inter_feat[item_seq_len_field][last_idx].unsqueeze(0).to(trainer.device)
                        })
                    elif config['task_type'] == 'NBR':
                        # For NBR models, we need to provide history baskets
                        user_mask = eval_dataset.inter_feat[eval_dataset.uid_field] == user_id
                        user_indices = torch.where(user_mask)[0]

                        if len(user_indices) == 0:
                            # User has no interactions, skip
                            skipped_users_no_interactions.append(user_id)
                            continue

                        # Get the last interaction which contains the longest/complete history
                        last_idx = user_indices[-1].item()

                        # Get the NBR history fields (these are set by NextBasketDataset)
                        history_items_field = eval_dataset.history_items_field  # 'history_item_matrix'
                        history_length_field = eval_dataset.history_length_field  # 'history_basket_length'
                        history_item_len_field = eval_dataset.history_item_len_field  # 'history_item_length_per_basket'

                        interaction = Interaction({
                            eval_dataset.uid_field: torch.tensor([user_id], device=trainer.device),
                            history_items_field: eval_dataset.inter_feat[history_items_field][last_idx].unsqueeze(0).to(trainer.device),
                            history_length_field: eval_dataset.inter_feat[history_length_field][last_idx].unsqueeze(0).to(trainer.device),
                            history_item_len_field: eval_dataset.inter_feat[history_item_len_field][last_idx].unsqueeze(0).to(trainer.device)
                        })
                    else:
                        # For CF models and traditional models, only user_id is needed
                        interaction = Interaction({
                            eval_dataset.uid_field: torch.tensor([user_id], device=trainer.device)
                        })

                    # Get predictions
                    scores = trainer.model.full_sort_predict(interaction)

                    # Get top-max_k items (we'll slice this for different k values)
                    _, topk_items = torch.topk(scores, k=max_k, dim=-1)
                    # Handle both 1D and 2D tensor outputs
                    topk_items_np = topk_items.cpu().numpy()
                    if topk_items_np.ndim > 1:
                        topk_items_np = topk_items_np[0]
                    all_user_topk_items[user_id] = topk_items_np

            if skipped_users_no_interactions:
                print(f"Skipped {len(skipped_users_no_interactions)} users with no interactions in retain dataset")

            # Debug: Check first user's predictions
            if len(all_user_topk_items) > 0:
                first_user_id = list(all_user_topk_items.keys())[0]
                first_user_topk = all_user_topk_items[first_user_id]
                print(f"[Debug] First user ({first_user_id}) top-20 predictions: {first_user_topk[:20].tolist()}")
                sensitive_in_first = [item for item in first_user_topk[:20] if item in sensitive_item_ids]
                print(f"[Debug] Sensitive items in first user's top-20: {sensitive_in_first}")

            # Evaluate for each k value
            sensitive_results = []
            for k in topk_list:
                users_with_sensitive_in_topk = 0
                total_sensitive_in_topk = 0
                sensitive_counts_per_user = []

                for user_id in all_user_topk_items.keys():
                    # Get top-k items for this user
                    topk_items = all_user_topk_items[user_id][:k]

                    # Check for sensitive items in top-k
                    sensitive_in_topk = [item for item in topk_items if item in sensitive_item_ids]
                    num_sensitive = len(sensitive_in_topk)

                    sensitive_counts_per_user.append(num_sensitive)

                    if num_sensitive > 0:
                        users_with_sensitive_in_topk += 1
                        total_sensitive_in_topk += num_sensitive

                # Compute metrics
                pct_users_with_sensitive = 100 * users_with_sensitive_in_topk / len(unlearned_user_ids)
                avg_sensitive_per_user = total_sensitive_in_topk / len(unlearned_user_ids)
                min_sensitive_per_user = min(sensitive_counts_per_user)
                max_sensitive_per_user = max(sensitive_counts_per_user)

                print(f"\n[Top-{k}] Users with sensitive items: {users_with_sensitive_in_topk}/{len(unlearned_user_ids)} ({pct_users_with_sensitive:.2f}%)")
                print(f"[Top-{k}] Sensitive items per user - Avg: {avg_sensitive_per_user:.4f}, Min: {min_sensitive_per_user}, Max: {max_sensitive_per_user}")
                print(f"[Top-{k}] Total sensitive items in predictions: {total_sensitive_in_topk}")

                checkpoint_idx = retrain_checkpoint_idx_to_match if retrain_checkpoint_idx_to_match is not None else None
                is_unlearned_model = (unlearning_fraction is not None and 
                                      unlearning_sample_selection_method is not None and 
                                      retrain_checkpoint_idx_to_match is not None)
                sensitive_results.append({
                    "sensitive_category": sensitive_category,
                    "checkpoint_idx": checkpoint_idx if (retrain_flag or is_unlearned_model) else None,
                    "is_retrained": retrain_flag,
                    "is_unlearned": is_unlearned_model,
                    "users_with_sensitive_in_topk": users_with_sensitive_in_topk,
                    "total_unlearned_users": len(unlearned_user_ids),
                    "pct_users_with_sensitive": pct_users_with_sensitive,
                    "avg_sensitive_per_user": avg_sensitive_per_user,
                    "min_sensitive_per_user": min_sensitive_per_user,
                    "max_sensitive_per_user": max_sensitive_per_user,
                    "total_sensitive_in_topk": total_sensitive_in_topk,
                    "topk": k,
                })

            result["sensitive_item_evaluation"] = sensitive_results
            
            # Save to CSV if sensitive_eval_only mode is enabled
            if sensitive_eval_only:
                save_sensitive_results_to_csv(
                    sensitive_results, 
                    config, 
                    retrain_flag=retrain_flag,
                    unlearning_algorithm=None
                )

        else:
            print(f"Warning: Sensitive items file not found at {sensitive_items_path}")

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process



def unlearn_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
    unlearning_algorithm="scif",
    max_norm=None,
    base_model_path=None,
    kookmin_init_rate=0.01,
    spam=False,
    damping=0.01,
    gif_damping=0.01,
    gif_scale_factor=1000,
    gif_iterations=100,
    gif_k_hops=2,
    gif_use_true_khop=False,
    gif_retain_samples=None,
    ceu_lambda=0.01,
    ceu_sigma=0.1,
    ceu_epsilon=0.1,
    ceu_cg_iterations=100,
    ceu_hessian_samples=1024,
    idea_damping=0.01,
    idea_sigma=0.1,
    idea_epsilon=0.1,
    idea_delta=0.01,
    idea_iterations=100,
    idea_hessian_samples=1024,
    seif_erase_std=0.6,
    seif_erase_std_final=0.005,
    seif_repair_epochs=4,
    seif_forget_class_weight=0.05,
    seif_learning_rate=0.0007,
    seif_momentum=0.9,
    seif_weight_decay=5e-4,
    unlearning_batchsize=1,
    max_training_hours=None,
    sensitive_eval_only=False,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """

    # configurations initialization
    if config_dict is None:
        config_dict = {}
    
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    retrain_checkpoint_idx_to_match = 0
    config["retrain_checkpoint_idx_to_match"] = retrain_checkpoint_idx_to_match
    # different batch_size for unlearning
    # TODO: make params
    config["train_batch_size"] = 256
    config["eval_batch_size"] = 512
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # For NBR tasks with sensitive_category, we need to check sensitive items against original dataset
    # before creating the cleaned dataset. This ensures we only include items that survive normal filtering.
    if config.task_type == "NBR" and "sensitive_category" in config and config['sensitive_category'] is not None:
        sensitive_category = config['sensitive_category']
        
        # Load sensitive items from file
        sensitive_items_file = os.path.join(
            config["data_path"],
            f"sensitive_asins_{sensitive_category}.txt"
        )
        
        if not os.path.exists(sensitive_items_file):
            sensitive_items_file = os.path.join(
                config["data_path"],
                f"sensitive_products_{sensitive_category}.txt"
            )
        
        if os.path.exists(sensitive_items_file):
            logger.info(f"Pre-checking sensitive items against original dataset...")
            with open(sensitive_items_file, 'r') as f:
                sensitive_items_raw = set(int(line.strip()) for line in f if line.strip())
            
            logger.info(f"Loaded {len(sensitive_items_raw)} raw sensitive items for pre-checking")
            
            # Create temporary dataset from original data to check item existence
            original_merged_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_merged.json"
            )
            
            if os.path.exists(original_merged_path):
                # Create a new config with the original merged file path
                # Use the original config_file_list to get all defaults, then override only the JSON path
                # This applies normal dataset filtering (MIN_NBR_BASKET_COUNT, user_inter_num_interval, etc.)
                original_config = Config(
                    model=config["model"], 
                    dataset=config["dataset"], 
                    config_file_list=config_file_list,  # Use original config files to get all defaults
                    config_dict={
                        "NEXT_BASKET_JSON": f"{config['dataset']}_merged.json",
                    }
                )
                
                # Create temporary dataset from original data to check item existence
                temp_original_dataset = create_dataset(original_config, unlearning=False, spam=False)
                original_item_tokens = set(temp_original_dataset.field2token_id[temp_original_dataset.iid_field].keys())
                logger.info(f"Original dataset has {len(original_item_tokens)} items after normal filtering")
                
                # Filter sensitive items to only those that exist in original dataset
                sensitive_items_in_original = []
                for item in sensitive_items_raw:
                    item_token = str(item)
                    if item_token in original_item_tokens:
                        sensitive_items_in_original.append(item)
                
                logger.info(f"Found {len(sensitive_items_in_original)} sensitive items in original dataset (out of {len(sensitive_items_raw)})")
                logger.info("These items survive normal dataset filtering and will be used in forget set")
                
                # Store the filtered sensitive items for later use
                config["_filtered_sensitive_items"] = sensitive_items_in_original

    dataset = create_dataset(config, unlearning=False, spam=spam)
    # save orig_df for forget set creation because inter_feat gets converted to torch during dataset.build()
    orig_inter_df = dataset.inter_feat.copy()

    removed_mask = np.zeros(len(orig_inter_df), dtype=bool)
    logger.info(dataset)

    # Get field names for later use
    uid_field, iid_field = dataset.uid_field, dataset.iid_field

    target_items = None
    if spam:
        # Use new naming convention (bandwagon_unpopular_ratio) - consistent with dataset loading
        # For NBR: use _fraud_baskets_ with .json, for SBR/CF: use _fraud_sessions_ with .inter
        if config.task_type == "NBR":
            unlearning_samples_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_fraud_baskets_bandwagon_unpopular_ratio_{config['unlearning_fraction']}_seed_{config['unlearn_sample_selection_seed']}.json"
            )
        else:
            unlearning_samples_path = os.path.join(
                config["data_path"],
                f"{config['dataset']}_fraud_sessions_bandwagon_unpopular_ratio_{config['unlearning_fraction']}_seed_{config['unlearn_sample_selection_seed']}.inter"
            )
        
        # Use new naming convention for metadata - consistent with dataset loading
        unlearning_samples_metadata_path = os.path.join(
            config["data_path"],
            f"{config['dataset']}_fraud_metadata_bandwagon_unpopular_ratio_{config['unlearning_fraction']}_seed_{config['unlearn_sample_selection_seed']}.json"
        )
        
        # Check if metadata file exists before trying to open it
        if os.path.exists(unlearning_samples_metadata_path):
            with open(unlearning_samples_metadata_path, "r") as f:
                metadata = json.load(f)
                # Convert target items from original tokens to internal IDs
                # Handle both numeric tokens (from JSON as int) and UUID strings (from JSON as str)
                # token2id expects string tokens, so convert to string if needed
                target_items_tokens = metadata["target_items"]
                target_items = []
                for item_token in target_items_tokens:
                    try:
                        # Convert to string if not already (handles both numeric and UUID tokens)
                        item_token_str = str(item_token) if not isinstance(item_token, str) else item_token
                        # Convert token to internal ID using dataset's mapping
                        item_id = dataset.token2id(iid_field, item_token_str)
                        target_items.append(item_id)
                    except (ValueError, KeyError):
                        # Item not in dataset (shouldn't happen for spam items, but handle gracefully)
                        logger.warning(f"Target item token {item_token} not found in dataset, skipping")
                        continue
                target_items = np.array(target_items, dtype=np.int64)
        else:
            logger.warning(
                f"Spam metadata file not found: {unlearning_samples_metadata_path}. "
                f"Target items will not be available for probability collection. "
                f"Unlearning will continue but some evaluation features may be limited."
            )
            target_items = None
    else:
        unlearning_samples_path = os.path.join(
            config["data_path"],
            f"{config['dataset']}_unlearn_pairs_{config['unlearning_sample_selection_method']}"
            f"_seed_{config['unlearn_sample_selection_seed']}"
            f"_unlearning_fraction_{float(config['unlearning_fraction'])}.inter"
        )

    print("loaded dataset")

    if config.task_type == "SBR":
        # Read header to infer column names dynamically
        with open(unlearning_samples_path, 'r') as f:
            header_line = f.readline().strip()

        # Extract column names from header (e.g., "user_id:token" -> "user_id")
        column_names = [col.split(':')[0] for col in header_line.split('\t')]

        unlearning_samples = pd.read_csv(
            unlearning_samples_path,
            sep="\t",
            names=column_names,
            header=0,
        )
    elif config.task_type == "CF":
        unlearning_samples = pd.read_csv(
            unlearning_samples_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            header=0,
        )
    elif config.task_type == "NBR":
        # For NBR, load user IDs from pickle/json file
        # Handle both .inter and .json base file extensions
        if unlearning_samples_path.endswith('.json'):
            unlearning_users_path_pkl = unlearning_samples_path.replace('.json', '.pkl')
            unlearning_users_path_json = unlearning_samples_path  # Already .json
        else:
            unlearning_users_path_pkl = unlearning_samples_path.replace('.inter', '.pkl')
            unlearning_users_path_json = unlearning_samples_path.replace('.inter', '.json')

        unlearning_user_ids = None
        if os.path.exists(unlearning_users_path_pkl):
            logger.info(f"Loading NBR unlearning user IDs from: {unlearning_users_path_pkl}")
            with open(unlearning_users_path_pkl, 'rb') as f:
                unlearning_user_ids = pickle.load(f)
        elif os.path.exists(unlearning_users_path_json):
            logger.info(f"Loading NBR unlearning user IDs from: {unlearning_users_path_json}")
            with open(unlearning_users_path_json, 'r') as f:
                unlearning_user_ids = json.load(f)
        else:
            raise FileNotFoundError(
                f"NBR unlearning user IDs file not found. Tried:\n"
                f"  - {unlearning_users_path_pkl}\n"
                f"  - {unlearning_users_path_json}\n"
                f"Please create unlearning sets using create_nbr_unlearning_sets_user_ids.py"
            )

        logger.info(f"Loaded {len(unlearning_user_ids)} user IDs for NBR unlearning")

        # For spam unlearning, we just need to unlearn the fraud users (all their items)
        # For sensitive category unlearning, we need to unlearn specific items from those users
        if spam:
            # For spam unlearning, we need to find all items for these fraud users
            # Load the fraud baskets JSON to get all items for each fraud user
            with open(unlearning_samples_path, 'r') as f:
                fraud_baskets_data = json.load(f)
            
            # Convert user IDs to strings for matching (JSON keys are strings)
            fraud_user_ids_str = [str(uid) for uid in unlearning_user_ids]
            
            # Collect all items for each fraud user from their baskets
            user_items_dict = {}
            total_items_before_filter = 0
            for user_id_str in fraud_user_ids_str:
                if user_id_str in fraud_baskets_data:
                    # Get all items from all baskets for this user
                    all_items = []
                    for basket in fraud_baskets_data[user_id_str]:
                        all_items.extend(basket)
                    # Remove duplicates but keep order
                    unique_items = list(dict.fromkeys(all_items))
                    total_items_before_filter += len(unique_items)
                    
                    # Filter items to only include those that exist in the dataset
                    # The dataset may have filtered out some items due to minimum interaction thresholds
                    # Items in JSON can be int/float, token mapping keys might be int or str depending on how they were created
                    filtered_items = []
                    token_map = dataset.field2token_id[dataset.iid_field]
                    for item in unique_items:
                        # Try to find the item in the token mapping
                        # Token mapping keys might be strings or integers depending on how the dataset was constructed
                        item_found = False
                        # Try as string first (most common case)
                        if str(item) in token_map:
                            item_found = True
                        # Also try as integer in case keys are stored as integers
                        elif item in token_map:
                            item_found = True
                        # Also try converting to int if item is a string representation of a number
                        elif isinstance(item, str) and item.isdigit() and int(item) in token_map:
                            item_found = True
                        
                        if item_found:
                            filtered_items.append(item)
                    
                    user_items_dict[int(user_id_str)] = filtered_items
            
            # Create DataFrame with user_id and list of all their items
            unlearning_samples = pd.DataFrame({
                'user_id': list(user_items_dict.keys()),
                'item_id': list(user_items_dict.values())
            })
            total_items_after_filter = sum(len(items) for items in user_items_dict.values())
            logger.info(f"Created unlearning samples for {len(unlearning_user_ids)} spam users")
            logger.info(f"Total items to unlearn: {total_items_after_filter} (filtered out {total_items_before_filter - total_items_after_filter} items that don't exist in dataset)")
        elif "sensitive_category" in config and config['sensitive_category'] is not None:
            sensitive_category = config['sensitive_category']

            # Try different naming conventions for sensitive items file
            sensitive_items_file = os.path.join(
                config["data_path"],
                f"sensitive_asins_{sensitive_category}.txt"
            )

            if not os.path.exists(sensitive_items_file):
                sensitive_items_file = os.path.join(
                    config["data_path"],
                    f"sensitive_products_{sensitive_category}.txt"
                )

            if not os.path.exists(sensitive_items_file):
                raise FileNotFoundError(
                    f"Sensitive items file not found for category '{sensitive_category}'. Tried:\n"
                    f"  - sensitive_asins_{sensitive_category}.txt\n"
                    f"  - sensitive_products_{sensitive_category}.txt"
                )

            logger.info(f"Loading sensitive items from: {sensitive_items_file}")
            with open(sensitive_items_file, 'r') as f:
                sensitive_items_raw = [int(line.strip()) for line in f if line.strip()]

            logger.info(f"Loaded {len(sensitive_items_raw)} raw sensitive items for category '{sensitive_category}'")

            # For NBR tasks, use pre-filtered sensitive items that were checked against both
            # original dataset (normal filtering) and cleaned dataset (after removal)
            if config.task_type == "NBR" and "_filtered_sensitive_items" in config:
                sensitive_items = config["_filtered_sensitive_items"]
                logger.info(f"Using pre-filtered sensitive items for NBR: {len(sensitive_items)} items")
                logger.info("These items were verified to exist in original dataset (survive normal filtering)")
            else:
                # For other task types, filter sensitive items to only include those that exist in the dataset
                # The dataset may have filtered out some items due to minimum interaction thresholds
                sensitive_items = []
                for item in sensitive_items_raw:
                    try:
                        # Check if this item exists in the dataset by checking the token mapping
                        item_token = str(item)
                        # Verify it exists in the dataset's item vocabulary (token -> ID mapping)
                        if item_token in dataset.field2token_id[dataset.iid_field]:
                            sensitive_items.append(item)
                    except (ValueError, KeyError):
                        # Item doesn't exist in dataset, skip it
                        pass

                logger.info(f"Filtered to {len(sensitive_items)} sensitive items that exist in the dataset (removed {len(sensitive_items_raw) - len(sensitive_items)} items)")

            if len(sensitive_items) == 0:
                raise ValueError(
                    f"No sensitive items from category '{sensitive_category}' exist in the dataset. "
                    f"This could mean all sensitive items were filtered out during preprocessing."
                )

            # Create DataFrame with user_id and the list of sensitive items for each user
            # Each user will unlearn the same set of sensitive items
            unlearning_samples = pd.DataFrame({
                'user_id': unlearning_user_ids,
                'item_id': [sensitive_items for _ in unlearning_user_ids]
            })
        else:
            raise ValueError(
                "For NBR unlearning, either 'spam=True' or 'sensitive_category' must be specified in config"
            )
    else:
        raise ValueError(f"Unsupported task_type: {config.task_type}. Only 'SBR', 'CF', and 'NBR' are supported.")

    print("loaded unlearning samples")

    user_ids = dataset.inter_feat[uid_field].to_numpy()
    item_ids = dataset.inter_feat[iid_field].to_numpy()

    print("created user and item ids")


    rows_by_user = dict()

    # For NBR, the item_id column already contains lists, so we need to handle it differently
    if config.task_type == "NBR":
        # Create dict directly without aggregation to avoid nested lists
        # Each row already has a list of items
        pairs_by_user = {}
        for _, row in unlearning_samples.iterrows():
            user_id = row[uid_field]
            item_list = row['item_id']
            # item_list is already a list, so use it directly
            pairs_by_user[user_id] = item_list
    else:
        # For SBR/CF, use the original aggregation approach
        # Use uid_field (e.g., "session_id" for SBR, "user_id" for CF) instead of hardcoding
        pairs_by_user = (
            unlearning_samples.groupby(uid_field)[iid_field]
            .agg(list)
            .to_dict()
        )

    # model loading and initialization

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, dataset).to(config["device"])

    checkpoint = torch.load(base_model_path)

    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    logger.info(model)

    # trainer loading and initialization
    # scale down the lr for non-reset params (which are the majority) during training we will scale the lr for the reset params to the normal value
    if unlearning_algorithm == "kookmin":
        config["learning_rate"] *= 0.1
        
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    train_data, valid_data, test_data = data_preparation(
        config, dataset, spam=spam,
    )

    # sanity check
    # print("evaluation of original model:\n")
    # test_result = trainer.evaluate(
    #     test_data, load_best_model=True, show_progress=True, model_file=base_model_path, collect_target_probabilities=spam, target_items=target_items,
    # )
    # if spam:
    #     test_result, probability_data = test_result
    # print(test_result)

    # negative item sampler used for unlearning later which can sample unseen items for all users
    unlearning_sampler = train_data._sampler

    # Determine how many user sessions we need per request
    if unlearning_algorithm == "scif":
        retain_sessions_per_request = 32  # Number of complete user sessions
    elif unlearning_algorithm == "kookmin":
        retain_sessions_per_request = 32  # Number of complete user sessions
    elif unlearning_algorithm == "fanchuan":
        retain_sessions_per_request = 32  # Number of complete user sessions
        contrastive_samples_per_iteration = 8  # For contrastive learning iterations
    elif unlearning_algorithm == "gif":
        retain_sessions_per_request = 32  # Number of complete user sessions
    elif unlearning_algorithm == "ceu":
        retain_sessions_per_request = 32  # Number of complete user sessions
    elif unlearning_algorithm == "idea":
        retain_sessions_per_request = 32  # Number of complete user sessions
    elif unlearning_algorithm == "seif":
        retain_sessions_per_request = 32  # Number of complete user sessions
    
    # Pre-sample retain data for all unlearning requests
    total_unlearn_requests = len(pairs_by_user)
    
    # Get all unique users from orig_inter_df
    unique_users = orig_inter_df[uid_field].unique()
    user_pool = unique_users.tolist()
    
    # Create a mapping of user to their session indices
    user_to_indices = orig_inter_df.groupby(uid_field).indices
    
    # Create an oversized pool to account for filtering needs
    pool_size_multiplier = 3  # Make pool 3x larger to ensure we have enough replacements
    
    if unlearning_algorithm == "fanchuan":
        base_sessions_needed = (
            retain_sessions_per_request * total_unlearn_requests * 
            contrastive_samples_per_iteration
        )
    else:
        base_sessions_needed = retain_sessions_per_request * total_unlearn_requests
    
    total_pool_size = base_sessions_needed * pool_size_multiplier
    
    # Create the oversized pool of users
    user_sample_pool = []
    while len(user_sample_pool) < total_pool_size:
        shuffled_users = np.random.permutation(user_pool)
        user_sample_pool.extend(shuffled_users.tolist())
    
    # Track which pool indices we've used
    pool_cursor = 0
    
    def get_retain_sessions_excluding_unlearned_users(
        num_interactions_needed,
        unlearned_users_before,
        user_pool,
        cursor,
        user_to_indices,
    ):
        """
        Get a set of full user sessions until we have at least `num_interactions_needed` interactions,
        skipping unlearned users.
        """
        selected_indices = []
        selected_users = []
        current_cursor = cursor
        total_interactions = 0

        while total_interactions < num_interactions_needed:
            if current_cursor >= len(user_pool):
                current_cursor = 0
                np.random.shuffle(user_pool)

            user_id = user_pool[current_cursor]
            current_cursor += 1

            if user_id in unlearned_users_before:
                continue

            session_indices = user_to_indices[user_id]
            session_len = len(session_indices)

            selected_indices.extend(session_indices)
            selected_users.append(user_id)
            total_interactions += session_len

        return selected_indices, selected_users, current_cursor
    
    # Now in the unlearning loop:
    unlearned_users_before = []
    
    # Get batch size from config or parameter (default: 1 for sequential processing)
    batch_size = config.get("unlearning_batchsize", unlearning_batchsize)
    total_users = len(pairs_by_user)
    
    # Calculate checkpoints based on number of batches
    if batch_size > 1:
        total_batches = (total_users + batch_size - 1) // batch_size  # Ceiling division
        unlearning_checkpoints_batch = [total_batches // 4, total_batches // 2, 3 * total_batches // 4, total_batches - 1]
        # Convert batch indices to user indices (epoch numbers in saved files)
        # When batch_size > 1, unlearn_request_idx is the batch index (0, 1, 2, ...)
        # But we need to find the actual user index where the checkpoint was saved
        # The checkpoint is saved at unlearn_request_idx, which equals the batch index
        # So we need to search for files with retrain_checkpoint_idx_to_match, not specific epoch numbers
        # We'll use the batch indices directly for searching, but the actual saved files use batch indices as epochs
        unlearning_checkpoints = unlearning_checkpoints_batch  # These are batch indices, which match the epoch in saved files
    else:
        unlearning_checkpoints = [len(pairs_by_user) // 4, len(pairs_by_user) // 2, 3 * len(pairs_by_user) // 4, len(pairs_by_user) - 1]
    
    # Construct eval_files by searching for files matching the pattern
    # The actual saved files use the request index as epoch, not the checkpoint batch index
    # Optionally include unlearning_batchsize in the pattern for hyperparameter tests
    base_filename = trainer.saved_model_file[:-len('.pth')]
    batch_size = config.get("unlearning_batchsize", 1)
    # Only use batchsize suffix if it's explicitly set and not the default (for hyperparameter tests)
    use_batchsize_suffix = batch_size != 1
    eval_files = []
    for r in range(4):
        if use_batchsize_suffix:
            # Search for files matching: base_unlearn_epoch_*_retrain_checkpoint_idx_to_match_r_bs{batchsize}.pth
            pattern = f"{base_filename}_unlearn_epoch_*_retrain_checkpoint_idx_to_match_{r}_bs{batch_size}.pth"
        else:
            # Search for files matching: base_unlearn_epoch_*_retrain_checkpoint_idx_to_match_r.pth
            pattern = f"{base_filename}_unlearn_epoch_*_retrain_checkpoint_idx_to_match_{r}.pth"
        matching_files = glob.glob(os.path.join(config["checkpoint_dir"], pattern))
        if matching_files:
            # Sort by epoch number (extract from filename) and take the one closest to the checkpoint
            def extract_epoch(fname):
                match = re.search(r'_unlearn_epoch_(\d+)_', fname)
                return int(match.group(1)) if match else 0
            matching_files.sort(key=extract_epoch)
            # Take the file with epoch closest to the checkpoint index
            checkpoint_idx = unlearning_checkpoints[r] if r < len(unlearning_checkpoints) else unlearning_checkpoints[-1]
            closest_file = min(matching_files, key=lambda f: abs(extract_epoch(f) - checkpoint_idx))
            eval_files.append(closest_file)
        else:
            # Fallback to original pattern if no matches found
            e = unlearning_checkpoints[r] if r < len(unlearning_checkpoints) else unlearning_checkpoints[-1]
            if use_batchsize_suffix:
                eval_files.append(f"{base_filename}_unlearn_epoch_{e}_retrain_checkpoint_idx_to_match_{r}_bs{batch_size}.pth")
            else:
                eval_files.append(f"{base_filename}_unlearn_epoch_{e}_retrain_checkpoint_idx_to_match_{r}.pth")
    
    eval_masks = []
    # Track which users were unlearned at each checkpoint for sensitive item evaluation
    unlearned_users_at_checkpoint = {}
    
    unlearning_times = []
    total_start_time = time.time()
    uid_seen = set()
    
    # Convert pairs_by_user to a list for batching
    sorted_pairs = sorted(pairs_by_user.items())
    
    # Calculate number of batches and time limit per batch
    # Time limit per batch = max_training_hours/num_batches hours = (max_training_hours*3600)/num_batches seconds
    # This ensures total unlearning time is maximally max_training_hours hours
    if batch_size > 1:
        num_batches = (len(sorted_pairs) + batch_size - 1) // batch_size
    else:
        num_batches = len(sorted_pairs)
    
    # Time limit per batch in seconds (max_training_hours is in hours)
    if max_training_hours is not None:
        # Convert hours to seconds
        max_training_hours_seconds = max_training_hours * 3600
        time_limit_per_batch_seconds = max_training_hours_seconds / num_batches if num_batches > 0 else float('inf')
        print(f"\nTime limit per batch: {time_limit_per_batch_seconds/3600:.2f} hours ({time_limit_per_batch_seconds:.0f} seconds)")
        print(f"Total batches: {num_batches}, Maximum total time: {max_training_hours:.2f} hours ({max_training_hours_seconds:.0f} seconds)\n")
    else:
        time_limit_per_batch_seconds = None
        print(f"\nTotal batches: {num_batches}, No time limit set\n")
    
    # Process in batches if batch_size > 1, otherwise process one by one
    if batch_size > 1:
        # Process in batches
        for batch_idx in range(0, len(sorted_pairs), batch_size):
            batch = sorted_pairs[batch_idx:batch_idx + batch_size]
            current_batch_size = len(batch)
            unlearn_request_idx = batch_idx // batch_size  # Batch index
            
            print(f"\nUnlearning batch {unlearn_request_idx + 1}/{(len(sorted_pairs) + batch_size - 1) // batch_size} with {current_batch_size} users\n")
            request_start_time = time.time()
            
            # Collect all forget items from all users in the batch
            batch_user_ids = []
            batch_forget_indices = []
            batch_clean_forget_indices = []
            
            for u, forget_items in batch:
                # Convert user and item tokens to internal IDs
                u_id = dataset.token2id(uid_field, str(u))
                if u_id in uid_seen:
                    print(f"Warning: user {u} (internal ID {u_id}) has already been unlearned before.")
                    continue
                uid_seen.add(u_id)
                # Filter out items that don't exist in the dataset
                # Items from JSON can be int/float, token2id expects string tokens but mapping keys might vary
                forget_items_ids = []
                for item in forget_items:
                    try:
                        # token2id expects string tokens, but first check if item exists in mapping
                        # Try string conversion first (most common)
                        item_token = str(item) if not isinstance(item, str) else item
                        item_id = dataset.token2id(iid_field, item_token)
                        forget_items_ids.append(item_id)
                    except ValueError:
                        # Item doesn't exist in dataset, skip it
                        # This can happen if the item was filtered out during dataset construction
                        pass
                
                unlearned_users_before.append(u_id)
                batch_user_ids.append(u_id)
                
                rows_by_user[u_id] = np.where(user_ids == u_id)[0]
                all_idx = rows_by_user[u_id]
                mask = np.isin(item_ids[all_idx], forget_items_ids)
                removed_mask[all_idx[mask]] = True
                
                # Collect indices for forget and clean forget datasets
                batch_forget_indices.extend(all_idx.tolist())
                batch_clean_forget_indices.extend(all_idx[~mask].tolist())
            
            saved_checkpoint = unlearn_request_idx in unlearning_checkpoints
            
            if saved_checkpoint:
                eval_mask = removed_mask.copy()
                eval_masks.append(eval_mask)
                # Store which users have been unlearned at this checkpoint (before skipping)
                checkpoint_idx = retrain_checkpoint_idx_to_match
                unlearned_users_at_checkpoint[checkpoint_idx] = list(unlearned_users_before)
            
            if "eval_only" in config and config["eval_only"]:
                continue
            
            # Create combined forget dataset for the batch
            if len(batch_forget_indices) == 0:
                print(f"Warning: Batch {unlearn_request_idx + 1} has no forget interactions, skipping.")
                continue
                
            cur_forget_ds = dataset.copy(
                orig_inter_df.iloc[batch_forget_indices]
            )
            
            print(f"Unlearning {len(cur_forget_ds)} interactions from {len(batch_user_ids)} users")
            
            # For spam unlearning: skip clean_forget_data (no replacement samples), use only retain_data
            if spam:
                print(f"Note: Spam unlearning - skipping clean_forget_data (no replacement samples), using only retain_data")
                clean_forget_data = None
            else:
                # Create clean forget dataset (interactions that should remain)
                if len(batch_clean_forget_indices) == 0:
                    clean_forget_ds = None
                    clean_forget_data = None
                    print(f"Note: All interactions in batch are sensitive. Using empty clean_forget_data and more retain data instead.")
                else:
                    clean_forget_ds = dataset.copy(
                        orig_inter_df.iloc[batch_clean_forget_indices]
                    )
                    try:
                        clean_forget_data = data_preparation(config, clean_forget_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)
                    except ValueError as e:
                        if "num_samples should be a positive integer" in str(e):
                            print(f"Note: All interactions in batch became empty after filtering. Using empty clean_forget_data and more retain data instead.")
                            clean_forget_data = None
                        else:
                            raise
            
            # Always create forget_data (spam data to unlearn)
            # Handle case where forget_data becomes empty after filtering during data_preparation
            try:
                forget_data = data_preparation(config, cur_forget_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)
            except ValueError as e:
                if "num_samples should be a positive integer" in str(e):
                    print(f"Warning: forget_data for batch {unlearn_request_idx + 1} became empty after filtering (likely due to dataset filtering like min_user_inter_num). Skipping this batch.")
                    continue
                else:
                    raise
            
            retain_limit_absolute = int(0.1 * len(orig_inter_df))  # 10% of full dataset
            
            forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
            
            if unlearning_algorithm == "scif":
                retain_batch_size = 16
                samples_wanted_constant = 32
                retain_samples_used = 8
                retain_samples_used_for_update = 8 * forget_size
                
                total_samples_needed = max(
                    retain_samples_used * forget_size,
                    samples_wanted_constant
                )
                
                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)
                
            elif unlearning_algorithm == "kookmin":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
                
                samples_wanted_constant = 32
                neg_grad_retain_sample_size = 8 * forget_size
                retain_samples_used_for_update = 8 * forget_size
                
                total_samples_needed = max(
                    neg_grad_retain_sample_size,
                    retain_samples_used_for_update,
                    samples_wanted_constant
                )
                
                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)
                
            elif unlearning_algorithm == "fanchuan":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
                
                samples_wanted_constant = 32
                retain_samples_used_for_update = 8 * forget_size
                unlearn_iters_contrastive = 8
                
                total_samples_needed = max(
                    retain_samples_used_for_update * unlearn_iters_contrastive,
                    samples_wanted_constant
                )
                
                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)
                
            elif unlearning_algorithm == "gif":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
                
                # GIF: similar to kookmin approach
                retain_samples_used_for_update = config["retain_samples_used_for_update"] if "retain_samples_used_for_update" in config else 8 * forget_size
                hessian_sample_size = 1024
                
                total_samples_needed = max(
                    retain_samples_used_for_update,
                    hessian_sample_size
                )
                
                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)
                
            elif unlearning_algorithm == "ceu":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
                
                # CEU: needs samples for Hessian computation and influence estimation
                ceu_hessian_samples = config["ceu_hessian_samples"] if "ceu_hessian_samples" in config else 1024
                retain_samples_used_for_update = 8 * forget_size
                
                total_samples_needed = max(
                    retain_samples_used_for_update,
                    ceu_hessian_samples
                )
                
                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)
                
            elif unlearning_algorithm == "idea":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
                
                # IDEA: needs samples for Hessian computation and gradient estimation
                idea_hessian_samples = config["idea_hessian_samples"] if "idea_hessian_samples" in config else 1024
                retain_samples_used_for_update = 8 * forget_size
                
                total_samples_needed = max(
                    retain_samples_used_for_update,
                    idea_hessian_samples
                )
                
                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)
                
            elif unlearning_algorithm == "seif":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
                
                # SEIF: needs samples for repair phase fine-tuning
                # Use similar amount as other methods - enough for multiple epochs
                samples_wanted_constant = 32
                seif_repair_epochs = config["seif_repair_epochs"] if "seif_repair_epochs" in config else 4
                retain_samples_used_for_update = 8 * forget_size * seif_repair_epochs
                
                total_samples_needed = max(
                    retain_samples_used_for_update,
                    samples_wanted_constant
                )
                
                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)
            
            # Get complete sessions
            retain_indices, retain_users, pool_cursor = get_retain_sessions_excluding_unlearned_users(
                total_samples_needed,  # Pass total interactions needed, not sessions
                unlearned_users_before,
                user_sample_pool,
                pool_cursor,
                user_to_indices,
            )
            
            print(f"Selected {len(retain_users)} user sessions with {len(retain_indices)} total interactions")
            
            # Create dataset from selected indices
            current_retain_data = orig_inter_df.iloc[retain_indices]
            current_retain_ds = dataset.copy(current_retain_data)
            
            # Temporarily modify batch size for this retain loader
            tmp = config["train_batch_size"]
            config["train_batch_size"] = retain_batch_size
            current_retain_loader = data_preparation(config, current_retain_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)
            config["train_batch_size"] = tmp
            
            # model training
            # Only include batchsize in filename for hyperparameter tests (when batchsize != 1)
            unlearning_batchsize_for_filename = batch_size if batch_size != 1 else None
            trainer.unlearn(
                unlearn_request_idx,
                forget_data,
                clean_forget_data,
                retain_train_data=current_retain_loader,
                retain_valid_data=None,#valid_data,
                retain_test_data=None,#test_data,
                unlearning_algorithm=unlearning_algorithm,
                saved=saved_checkpoint,
                show_progress=False,  # no progress bar during unlearning as it is short either way
                max_norm=max_norm,
                unlearned_users_before=unlearned_users_before,
                kookmin_init_rate=kookmin_init_rate,
                retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match,
                task_type=config.task_type,
                damping=damping,
                unlearning_batchsize=unlearning_batchsize_for_filename,
                gif_damping=gif_damping,
                gif_scale_factor=gif_scale_factor,
                gif_iterations=gif_iterations,
                gif_k_hops=gif_k_hops,
                gif_use_true_khop=gif_use_true_khop,
                gif_retain_samples=gif_retain_samples,
                ceu_lambda=ceu_lambda,
                ceu_sigma=ceu_sigma,
                ceu_epsilon=ceu_epsilon,
                ceu_cg_iterations=ceu_cg_iterations,
                ceu_hessian_samples=ceu_hessian_samples,
                idea_damping=idea_damping,
                idea_sigma=idea_sigma,
                idea_epsilon=idea_epsilon,
                idea_delta=idea_delta,
                idea_iterations=idea_iterations,
                idea_hessian_samples=idea_hessian_samples,
                seif_erase_std=seif_erase_std,
                seif_erase_std_final=seif_erase_std_final,
                seif_repair_epochs=seif_repair_epochs,
                seif_forget_class_weight=seif_forget_class_weight,
                seif_learning_rate=seif_learning_rate,
                seif_momentum=seif_momentum,
                seif_weight_decay=seif_weight_decay,
                original_dataset=dataset,
                time_limit_per_batch=time_limit_per_batch_seconds,
                retain_samples_used_for_update=retain_samples_used_for_update if unlearning_algorithm == "scif" else None,
            )
            
            request_end_time = time.time()
            request_time = request_end_time - request_start_time
            unlearning_times.append(request_time)
            
            print(f"\n\nBatch {unlearn_request_idx + 1} completed in {request_time:.2f} seconds\n\n")
            
            # Increment checkpoint counter after unlearning completes
            if saved_checkpoint:
                retrain_checkpoint_idx_to_match += 1
                config["retrain_checkpoint_idx_to_match"] = retrain_checkpoint_idx_to_match
            
            sys.stdout.flush()
            gc.collect()
    else:
        # Original sequential processing when batch_size == 1
        for unlearn_request_idx, (u, forget_items) in enumerate(sorted_pairs):
            print(f"\nUnlearning request {unlearn_request_idx + 1}/{len(pairs_by_user)} for user {u}\n")
            request_start_time = time.time()

            # Convert user and item tokens to internal IDs
            # Convert to string first in case pandas read them as integers
            u_id = dataset.token2id(uid_field, str(u))
            if u_id in uid_seen:
                print(f"Warning: user {u} (internal ID {u_id}) has already been unlearned before.")
                continue
            uid_seen.add(u_id)
            # Filter out items that don't exist in the dataset
            # Items from JSON can be int/float, token2id expects string tokens but mapping keys might vary
            forget_items_ids = []
            for item in forget_items:
                try:
                    # token2id expects string tokens, but first check if item exists in mapping
                    # Try string conversion first (most common)
                    item_token = str(item) if not isinstance(item, str) else item
                    item_id = dataset.token2id(iid_field, item_token)
                    forget_items_ids.append(item_id)
                except ValueError:
                    # Item doesn't exist in dataset, skip it
                    # This can happen if the item was filtered out during dataset construction
                    pass

            unlearned_users_before.append(u_id)

            rows_by_user[u_id] = np.where(user_ids == u_id)[0]
            all_idx = rows_by_user[u_id]
            mask = np.isin(item_ids[all_idx], forget_items_ids)
            removed_mask[all_idx[mask]] = True

            saved_checkpoint = unlearn_request_idx in unlearning_checkpoints
            
            if saved_checkpoint:
                eval_mask = removed_mask.copy()
                eval_masks.append(eval_mask)
                # Store which users have been unlearned at this checkpoint (before skipping)
                checkpoint_idx = retrain_checkpoint_idx_to_match
                unlearned_users_at_checkpoint[checkpoint_idx] = list(unlearned_users_before)

            if "eval_only" in config and config["eval_only"]:
                continue

            cur_forget_ds = dataset.copy(
                orig_inter_df.iloc[all_idx]
            )

            print(f"Unlearning {len(cur_forget_ds)} interactions")

            # For spam unlearning: skip clean_forget_data (no replacement samples), use only retain_data
            if spam:
                print(f"Note: Spam unlearning - skipping clean_forget_data (no replacement samples), using only retain_data")
                clean_forget_data = None
            else:
                clean_forget_ds = dataset.copy(
                    orig_inter_df.iloc[all_idx[~mask]]
                )

                # Handle case where all user interactions are sensitive (clean_forget_ds is empty)
                # Check both before and after data_preparation, since filtering may occur during build()
                if len(clean_forget_ds) == 0:
                    print(f"Note: All {len(cur_forget_ds)} interactions for user {u} are sensitive. Using empty clean_forget_data and more retain data instead.")
                    # Set clean_forget_data to None since we can't create a dataloader from empty dataset
                    clean_forget_data = None
                else:
                    try:
                        clean_forget_data = data_preparation(config, clean_forget_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)
                    except ValueError as e:
                        if "num_samples should be a positive integer" in str(e):
                            print(f"Note: All {len(cur_forget_ds)} interactions for user {u} became empty after filtering (likely due to nbr_phase). Using empty clean_forget_data and more retain data instead.")
                            clean_forget_data = None
                        else:
                            raise

            # Handle case where forget_data becomes empty after filtering during data_preparation
            try:
                forget_data = data_preparation(config, cur_forget_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)
            except ValueError as e:
                if "num_samples should be a positive integer" in str(e):
                    print(f"Warning: forget_data for user {u} became empty after filtering (likely due to dataset filtering like min_user_inter_num). Skipping this user.")
                    continue
                else:
                    raise

            retain_limit_absolute = int(0.1 * len(orig_inter_df))  # 10% of full dataset

            avg_session_length = len(orig_inter_df) / len(unique_users)

            if unlearning_algorithm == "scif":
                retain_batch_size = 16
                samples_wanted_constant = 32
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)
                retain_samples_used = 8
                retain_samples_used_for_update = 8 * forget_size
                
                total_samples_needed = max(
                    retain_samples_used * forget_size,
                    samples_wanted_constant
                )

                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)

            elif unlearning_algorithm == "kookmin":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)

                samples_wanted_constant = 32
                neg_grad_retain_sample_size = 8 * forget_size
                retain_samples_used_for_update = 8 * forget_size

                total_samples_needed = max(
                    neg_grad_retain_sample_size,
                    retain_samples_used_for_update,
                    samples_wanted_constant
                )

                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)

            elif unlearning_algorithm == "fanchuan":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)

                samples_wanted_constant = 32
                retain_samples_used_for_update = 8 * forget_size
                unlearn_iters_contrastive = 8

                total_samples_needed = max(
                    retain_samples_used_for_update * unlearn_iters_contrastive,
                    samples_wanted_constant
                )

                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)

            elif unlearning_algorithm == "gif":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)

                # GIF: similar to kookmin approach
                retain_samples_used_for_update = config["retain_samples_used_for_update"] if "retain_samples_used_for_update" in config else 8 * forget_size
                hessian_sample_size = 32

                total_samples_needed = max(
                    retain_samples_used_for_update,
                    hessian_sample_size
                )

                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)

            elif unlearning_algorithm == "ceu":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)

                # CEU: needs samples for Hessian computation and influence estimation
                ceu_hessian_samples = config["ceu_hessian_samples"] if "ceu_hessian_samples" in config else 1024
                retain_samples_used_for_update = 8 * forget_size

                total_samples_needed = max(
                    retain_samples_used_for_update,
                    ceu_hessian_samples
                )

                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)

            elif unlearning_algorithm == "idea":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)

                # IDEA: needs samples for Hessian computation and gradient estimation
                idea_hessian_samples = config["idea_hessian_samples"] if "idea_hessian_samples" in config else 1024
                retain_samples_used_for_update = 8 * forget_size

                total_samples_needed = max(
                    retain_samples_used_for_update,
                    idea_hessian_samples
                )

                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)

            elif unlearning_algorithm == "seif":
                retain_batch_size = config["train_batch_size"]
                forget_size = len(forget_data[0].dataset) if isinstance(forget_data, tuple) else len(forget_data.dataset)

                # SEIF: needs samples for repair phase fine-tuning
                # Use similar amount as other methods - enough for multiple epochs
                samples_wanted_constant = 32
                seif_repair_epochs = config["seif_repair_epochs"] if "seif_repair_epochs" in config else 4
                retain_samples_used_for_update = 8 * forget_size * seif_repair_epochs

                total_samples_needed = max(
                    retain_samples_used_for_update,
                    samples_wanted_constant
                )

                # Cap to 10% of dataset
                total_samples_needed = min(total_samples_needed, retain_limit_absolute)
                # Note: We pass total_samples_needed directly as it represents interactions needed
                # The function will collect full user sessions until we have enough interactions

            # Get complete sessions
            retain_indices, retain_users, pool_cursor = get_retain_sessions_excluding_unlearned_users(
                total_samples_needed,  # Pass total interactions needed, not sessions
                unlearned_users_before,
                user_sample_pool,
                pool_cursor,
                user_to_indices,
            )
            
            print(f"Selected {len(retain_users)} user sessions with {len(retain_indices)} total interactions")
            
            # Create dataset from selected indices
            current_retain_data = orig_inter_df.iloc[retain_indices]
            current_retain_ds = dataset.copy(current_retain_data)
            
            # Temporarily modify batch size for this retain loader
            tmp = config["train_batch_size"]
            config["train_batch_size"] = retain_batch_size
            current_retain_loader = data_preparation(config, current_retain_ds, unlearning=True, spam=spam, sampler=unlearning_sampler)
            config["train_batch_size"] = tmp

            # model training
            # Only include batchsize in filename for hyperparameter tests (when batchsize != 1)
            unlearning_batchsize_for_filename = batch_size if batch_size != 1 else None
            trainer.unlearn(
                unlearn_request_idx,
                forget_data,
                clean_forget_data,
                retain_train_data=current_retain_loader,
                retain_valid_data=None,#valid_data,
                retain_test_data=None,#test_data,
                unlearning_algorithm=unlearning_algorithm,
                saved=saved_checkpoint,
                show_progress=False,  # no progress bar during unlearning as it is short either way
                max_norm=max_norm,
                unlearned_users_before=unlearned_users_before,
                kookmin_init_rate=kookmin_init_rate,
                retrain_checkpoint_idx_to_match=retrain_checkpoint_idx_to_match,
                task_type=config.task_type,
                damping=damping,
                unlearning_batchsize=unlearning_batchsize_for_filename,
                gif_damping=gif_damping,
                gif_scale_factor=gif_scale_factor,
                gif_iterations=gif_iterations,
                gif_k_hops=gif_k_hops,
                gif_use_true_khop=gif_use_true_khop,
                gif_retain_samples=gif_retain_samples,
                ceu_lambda=ceu_lambda,
                ceu_sigma=ceu_sigma,
                ceu_epsilon=ceu_epsilon,
                ceu_cg_iterations=ceu_cg_iterations,
                ceu_hessian_samples=ceu_hessian_samples,
                idea_damping=idea_damping,
                idea_sigma=idea_sigma,
                idea_epsilon=idea_epsilon,
                idea_delta=idea_delta,
                idea_iterations=idea_iterations,
                idea_hessian_samples=idea_hessian_samples,
                seif_erase_std=seif_erase_std,
                seif_erase_std_final=seif_erase_std_final,
                seif_repair_epochs=seif_repair_epochs,
                seif_forget_class_weight=seif_forget_class_weight,
                seif_learning_rate=seif_learning_rate,
                seif_momentum=seif_momentum,
                seif_weight_decay=seif_weight_decay,
                original_dataset=dataset,
                time_limit_per_batch=time_limit_per_batch_seconds,
                retain_samples_used_for_update=retain_samples_used_for_update if unlearning_algorithm == "scif" else None,
            )

            request_end_time = time.time()
            request_time = request_end_time - request_start_time
            unlearning_times.append(request_time)
            
            print(f"\n\nRequest {unlearn_request_idx + 1} completed in {request_time:.2f} seconds\n\n")

            # Increment checkpoint counter after unlearning completes
            if saved_checkpoint:
                retrain_checkpoint_idx_to_match += 1
                config["retrain_checkpoint_idx_to_match"] = retrain_checkpoint_idx_to_match

            sys.stdout.flush()
            gc.collect()


    if "eval_only" not in config or not config["eval_only"]:
        total_end_time = time.time()
        total_unlearning_time = total_end_time - total_start_time

        print(f"\nUnlearning Time Summary")
        print(f"Total unlearning time: {total_unlearning_time:.2f} seconds ({total_unlearning_time/60:.2f} minutes)")
        print(f"Average time per request: {np.mean(unlearning_times):.2f} seconds")
        print(f"Min time per request: {np.min(unlearning_times):.2f} seconds")
        print(f"Max time per request: {np.max(unlearning_times):.2f} seconds")
        print(f"Total requests processed: {len(unlearning_times)}")

    # eval
    # set eval batch size manually for now, as there is no parameter yet. change this in the future
    config["eval_batch_size"] = 256
    results = []

    # Dictionary to store per-model interaction data as lists of tuples
    model_interaction_probabilities = {}
    
    # Check if we should skip normal evaluation metrics
    sensitive_eval_only = "sensitive_eval_only" in config and config["sensitive_eval_only"]
    
    if not sensitive_eval_only:
        print("Original Model:")
        print(f"Evaluating model {base_model_path} on data with current mask\n")
        if not os.path.exists(base_model_path):
            print(f"Warning: Base model file not found: {base_model_path}, skipping evaluation")
            test_result = None
        else:
            test_result = trainer.evaluate(
                test_data, load_best_model=True, show_progress=False, model_file=base_model_path, collect_target_probabilities=spam, target_items=target_items,
            )
            if test_result is None:
                print(f"Warning: Failed to evaluate base model {base_model_path}")
            elif spam:
                test_result, probability_data = test_result
        if test_result is not None:
            print(test_result)
    else:
        print("Sensitive-eval-only mode: skipping normal evaluation metrics for all models")

    # First loop: evaluate each model on its corresponding masked data
    if not sensitive_eval_only:
        for i, (file, mask) in enumerate(zip(eval_files, eval_masks)):
            # Skip if checkpoint file doesn't exist
            if not os.path.exists(file):
                print(f"Skipping evaluation for {file} - checkpoint file not found")
                continue
                
            print(f"Evaluating model {file} on data with current mask\n")
            
            model_interaction_probabilities[file] = []
            
            # Clear GPU cache before each evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # test on data with just current poisoned data removed
            cur_eval_df = orig_inter_df.loc[~mask]
            cur_eval_dataset = dataset.copy(cur_eval_df)
            cur_train_data, cur_val_data, cur_test_data = data_preparation(config, cur_eval_dataset, spam=spam, sampler=unlearning_sampler)

            del cur_train_data, cur_val_data  # we only need test data for evaluation
            gc.collect()

            test_result = trainer.evaluate(
                cur_test_data, load_best_model=True, show_progress=False, model_file=file, collect_target_probabilities=spam, target_items=target_items,
            )
            if test_result is None:
                print(f"Skipping results for {file} - evaluation failed (checkpoint not found)")
                continue
            if spam:
                test_result, probability_data = test_result
                model_interaction_probabilities[file] = probability_data

            result = {
                "test_result": test_result,
                "model_file": file,
                "mask_type": "current",
            }
            results.append(result)
            print(f"Results for model {file} only removing currently poisoned data: {test_result}")
            
            # clear the current test data
            del cur_eval_df, cur_eval_dataset, cur_test_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("\n")
            sys.stdout.flush()

    # Only evaluate on unpoisoned data in spam scenario
    if spam and not sensitive_eval_only:
        print("Creating unpoisoned dataset for all models...")
        unpoisoned_eval_df = orig_inter_df.loc[~eval_masks[-1]]
        unpoisoned_eval_dataset = dataset.copy(unpoisoned_eval_df)
        unpoisoned_train_data, unpoisoned_val_data, unpoisoned_test_data = data_preparation(config, unpoisoned_eval_dataset, spam=spam)

        del unpoisoned_train_data, unpoisoned_val_data  # we only need test data for evaluation
        gc.collect()

        # Second loop: evaluate all models on the same unpoisoned data
        for file in eval_files:
            # Skip if checkpoint file doesn't exist
            if not os.path.exists(file):
                print(f"Skipping evaluation for {file} on unpoisoned data - checkpoint file not found")
                continue
                
            print(f"Evaluating model {file} on unpoisoned data\n")

            unpoisoned_key = f"{file}_unpoisoned"

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            unpoisoned_test_result = trainer.evaluate(
                unpoisoned_test_data, load_best_model=True, show_progress=False, model_file=file, collect_target_probabilities=spam, target_items=target_items,
            )
            if unpoisoned_test_result is None:
                print(f"Skipping results for {file} on unpoisoned data - evaluation failed (checkpoint not found)")
                continue
            if spam:
                unpoisoned_test_result, probability_data = unpoisoned_test_result
                model_interaction_probabilities[unpoisoned_key] = probability_data

            unpoisoned_result = {
                "test_result": unpoisoned_test_result,
                "model_file": file,
                "mask_type": "unpoisoned",
            }
            results.append(unpoisoned_result)

            print(f"Results for model {file} on unpoisoned data: {unpoisoned_test_result}")
            print("\n")
            sys.stdout.flush()

        del unpoisoned_eval_df, unpoisoned_eval_dataset, unpoisoned_test_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    

    # Save model interaction probabilities if spam mode (default: True, can be disabled with dont_save_interaction_probabilities)
    if spam and config.get("save_interaction_probabilities", True) and not sensitive_eval_only:
        # Save to the same directory as the model files
        # trainer.saved_model_file is like "saved/model_XXX.pth", so dirname gives us "saved"
        model_dir = os.path.dirname(trainer.saved_model_file) or config.get("checkpoint_dir", "saved")
        # Create directory if it doesn't exist (open() creates files but not directories)
        os.makedirs(model_dir, exist_ok=True)
        pickle_file = os.path.join(model_dir, "model_interaction_probabilities.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(model_interaction_probabilities, f)
        print(f"Saved model interaction probabilities to {pickle_file}")
    elif spam:
        print("Skipping model interaction probability saving (use --dont_save_interaction_probabilities to disable)")

    # Sensitive item evaluation for unlearned users
    if config.get('sensitive_category') is not None:
        print("Sensitive Item Evaluation")

        sensitive_category = config['sensitive_category']
        sensitive_items_path = os.path.join(config['data_path'], f"sensitive_asins_{sensitive_category}.txt")

        if os.path.exists(sensitive_items_path):
            # For NBR, use pre-filtered sensitive items that were verified against original dataset
            # This ensures we're checking the right items (those that exist before any filtering)
            if config.task_type == "NBR" and "_filtered_sensitive_items" in config:
                sensitive_asins = set(str(item) for item in config["_filtered_sensitive_items"])
                print(f"Using {len(sensitive_asins)} pre-filtered sensitive items (verified against original dataset)")
            else:
                with open(sensitive_items_path, 'r') as f:
                    sensitive_asins = set(line.strip() for line in f if line.strip())
                print(f"Loaded {len(sensitive_asins)} sensitive items from {sensitive_items_path}")

            # Map ASINs to internal item IDs
            # The dataset has item_id tokens that need to be mapped
            sensitive_item_ids = set()
            iid_field = dataset.iid_field
            for asin in sensitive_asins:
                try:
                    item_id = dataset.token2id(iid_field, asin)
                    sensitive_item_ids.add(item_id)
                except ValueError:
                    # ASIN not in dataset (e.g., filtered out or not in this subset)
                    pass

            print(f"Mapped to {len(sensitive_item_ids)} sensitive internal item IDs (out of {len(sensitive_asins)} ASINs)")

            # Evaluate each model (each corresponds to a different checkpoint)
            for checkpoint_idx, file in enumerate(eval_files):
                print(f"\nEvaluating sensitive item exposure for model: {file}")

                # Get list of users that were unlearned up to this checkpoint
                unlearned_user_ids = unlearned_users_at_checkpoint.get(checkpoint_idx, [])
                if not unlearned_user_ids:
                    print(f"  Warning: No unlearned users found for checkpoint {checkpoint_idx}")
                    continue

                print(f"  Evaluating {len(unlearned_user_ids)} users unlearned up to checkpoint {checkpoint_idx}")

                # Load model
                checkpoint = torch.load(file, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint['state_dict'])
                trainer.model.eval()

                # Get list of k values to evaluate
                topk_list = config['topk'] if isinstance(config['topk'], list) else [config['topk']]
                max_k = max(topk_list)

                # Get top-max_k predictions once for all users
                all_user_topk_items = {}
                # Use test_data._dataset to access the underlying dataset after data_preparation
                eval_dataset = test_data._dataset
                with torch.no_grad():
                    for user_id in unlearned_user_ids:
                        # Create interaction for this user based on task type
                        if config['task_type'] == 'SBR':
                            # For sequential models, we need to provide item sequence
                            # Get the user's interaction data from the dataset
                            # The dataset.inter_feat is sorted by uid and time, and after augmentation
                            # each row contains the history up to that point
                            user_mask = eval_dataset.inter_feat[eval_dataset.uid_field] == user_id
                            user_indices = torch.where(user_mask)[0]

                            if len(user_indices) == 0:
                                # User has no interactions, skip
                                continue

                            # Get the last interaction which contains the longest/complete sequence
                            last_idx = user_indices[-1].item()

                            # Get the item sequence fields
                            item_seq_field = trainer.model.ITEM_SEQ
                            item_seq_len_field = trainer.model.ITEM_SEQ_LEN

                            interaction = Interaction({
                                eval_dataset.uid_field: torch.tensor([user_id], device=trainer.device),
                                item_seq_field: eval_dataset.inter_feat[item_seq_field][last_idx].unsqueeze(0).to(trainer.device),
                                item_seq_len_field: eval_dataset.inter_feat[item_seq_len_field][last_idx].unsqueeze(0).to(trainer.device)
                            })
                        elif config['task_type'] == 'NBR':
                            # For NBR models, we need to provide history baskets
                            # Get the user's interaction data from the dataset
                            user_mask = eval_dataset.inter_feat[eval_dataset.uid_field] == user_id
                            user_indices = torch.where(user_mask)[0]

                            if len(user_indices) == 0:
                                # User has no interactions, skip
                                continue

                            # Get the last interaction which contains the longest/complete history
                            last_idx = user_indices[-1].item()

                            # Get the NBR history fields (these are set by NextBasketDataset)
                            history_items_field = eval_dataset.history_items_field  # 'history_item_matrix'
                            history_length_field = eval_dataset.history_length_field  # 'history_basket_length'
                            history_item_len_field = eval_dataset.history_item_len_field  # 'history_item_length_per_basket'

                            interaction = Interaction({
                                eval_dataset.uid_field: torch.tensor([user_id], device=trainer.device),
                                history_items_field: eval_dataset.inter_feat[history_items_field][last_idx].unsqueeze(0).to(trainer.device),
                                history_length_field: eval_dataset.inter_feat[history_length_field][last_idx].unsqueeze(0).to(trainer.device),
                                history_item_len_field: eval_dataset.inter_feat[history_item_len_field][last_idx].unsqueeze(0).to(trainer.device)
                            })
                        else:
                            # For CF models, only user_id is needed
                            interaction = Interaction({
                                'user_id': torch.tensor([user_id], device=trainer.device)
                            })

                        # Get predictions
                        scores = trainer.model.full_sort_predict(interaction)

                        # Get top-max_k items (we'll slice this for different k values)
                        _, topk_items = torch.topk(scores, k=max_k, dim=-1)
                        # Handle both 1D and 2D tensor outputs
                        topk_items_np = topk_items.cpu().numpy()
                        if topk_items_np.ndim > 1:
                            topk_items_np = topk_items_np[0]
                        all_user_topk_items[user_id] = topk_items_np

                # Evaluate for each k value
                for k in topk_list:
                    users_with_sensitive_in_topk = 0
                    total_sensitive_in_topk = 0
                    sensitive_counts_per_user = []

                    for user_id in unlearned_user_ids:
                        # Get top-k items for this user
                        topk_items = all_user_topk_items[user_id][:k]

                        # Check for sensitive items in top-k
                        sensitive_in_topk = [item for item in topk_items if item in sensitive_item_ids]
                        num_sensitive = len(sensitive_in_topk)

                        sensitive_counts_per_user.append(num_sensitive)

                        if num_sensitive > 0:
                            users_with_sensitive_in_topk += 1
                            total_sensitive_in_topk += num_sensitive

                    # Compute metrics
                    pct_users_with_sensitive = 100 * users_with_sensitive_in_topk / len(unlearned_user_ids)
                    avg_sensitive_per_user = total_sensitive_in_topk / len(unlearned_user_ids)
                    min_sensitive_per_user = min(sensitive_counts_per_user)
                    max_sensitive_per_user = max(sensitive_counts_per_user)

                    print(f"  [Top-{k}] Users with sensitive items: {users_with_sensitive_in_topk}/{len(unlearned_user_ids)} ({pct_users_with_sensitive:.2f}%)")
                    print(f"  [Top-{k}] Sensitive items per user - Avg: {avg_sensitive_per_user:.4f}, Min: {min_sensitive_per_user}, Max: {max_sensitive_per_user}")
                    print(f"  [Top-{k}] Total sensitive items in predictions: {total_sensitive_in_topk}")

                    # Add to results
                    results.append({
                        "model_file": file,
                        "sensitive_category": sensitive_category,
                        "checkpoint_idx": checkpoint_idx,
                        "users_with_sensitive_in_topk": users_with_sensitive_in_topk,
                        "total_unlearned_users": len(unlearned_user_ids),
                        "pct_users_with_sensitive": pct_users_with_sensitive,
                        "avg_sensitive_per_user": avg_sensitive_per_user,
                        "min_sensitive_per_user": min_sensitive_per_user,
                        "max_sensitive_per_user": max_sensitive_per_user,
                        "total_sensitive_in_topk": total_sensitive_in_topk,
                        "topk": k,
                    })
        else:
            print(f"Warning: Sensitive items file not found at {sensitive_items_path}")

    # RULI Privacy (Game 2) MIA Evaluation
    if config.get("ruli_privacy_evaluation", False) and config.get("sensitive_category") is not None:
        print("\n" + "="*60)
        print("RULI Privacy (Game 2) MIA Evaluation")
        print("="*60)
        
        try:
            from recbole.quick_start.ruli_privacy_evaluator import RULIPrivacyEvaluator
            
            sensitive_category = config['sensitive_category']
            
            # Initialize evaluator
            # Use the unlearning algorithm being evaluated to select matching Qh models
            evaluator = RULIPrivacyEvaluator(
                config=config,
                dataset=dataset,
                train_data=train_data,
                test_data=test_data,
                shadow_models_dir=config.get("ruli_privacy_shadow_models_dir"),
                k=config.get("ruli_privacy_k", 8),
                beta_threshold=config.get("ruli_privacy_beta_threshold", 0.5),
                n_population_samples=config.get("ruli_privacy_n_population_samples", 2500),
                unlearning_algorithm=config.get("ruli_privacy_unlearning_algorithm") or unlearning_algorithm,
            )
            
            # Load forget set if available
            forget_set_df = None
            if 'unlearning_samples' in locals() and unlearning_samples is not None:
                forget_set_df = unlearning_samples
            else:
                # Try to load forget set from file
                try:
                    unlearning_samples_path = os.path.join(
                        config["data_path"],
                        f"{config['dataset']}_unlearn_pairs_sensitive_category_{sensitive_category}"
                        f"_seed_{config.get('unlearn_sample_selection_seed', config.get('seed'))}"
                        f"_unlearning_fraction_{float(config.get('unlearning_fraction', 0.0001))}.inter"
                    )
                    if os.path.exists(unlearning_samples_path):
                        if config.task_type == "CF":
                            forget_set_df = pd.read_csv(
                                unlearning_samples_path,
                                sep="\t",
                                names=["user_id", "item_id", "rating", "timestamp"],
                                header=0,
                            )
                        elif config.task_type == "SBR":
                            with open(unlearning_samples_path, 'r') as f:
                                header_line = f.readline().strip()
                            column_names = [col.split(':')[0] for col in header_line.split('\t')]
                            forget_set_df = pd.read_csv(
                                unlearning_samples_path,
                                sep="\t",
                                names=column_names,
                                header=0,
                            )
                except Exception as e:
                    logger.warning(f"Could not load forget set: {e}")
            
            # Construct D_target
            d_target_seed = config.get("ruli_privacy_d_target_seed", config.get("seed"))
            d_target, overlap_flags = evaluator.construct_d_target(
                sensitive_category=sensitive_category,
                forget_set=forget_set_df,
                seed=d_target_seed,
            )
            
            print(f"Constructed D_target with {len(d_target)} samples")
            print(f"  - {sum(overlap_flags)} samples overlap with forget set (will be skipped)")
            print(f"  - {len(d_target) - sum(overlap_flags)} samples not in forget set")
            
            # Find retrained model path
            # For unlearning evaluation, we need a retrained baseline model
            # This should be trained separately without the forget set
            retrained_model_path = None
            
            # Try to find retrained model - check multiple possible locations
            possible_paths = [
                # Standard retrained model path
                os.path.join(
                    config.get("model_dir", "./saved"),
                    f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_retrained_best.pth"
                ),
                # Alternative location
                os.path.join(
                    "./saved",
                    f"model_{config['model']}_seed_{config['seed']}_dataset_{config['dataset']}_retrained_best.pth"
                ),
                # If retrain_flag is set, the current model might be the retrained one
                base_model_path if config.get("retrain_flag") else None,
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    retrained_model_path = path
                    break
            
            # Evaluate each unlearned model checkpoint
            # In eval_only mode, eval_files should still be defined (from before the unlearning loop)
            if 'eval_files' not in locals() or not eval_files:
                print("Warning: No unlearned model checkpoints found. Skipping RULI Privacy evaluation.")
                print("  Make sure unlearned model checkpoints exist, or run without --eval_only first.")
            else:
                for checkpoint_idx, unlearned_model_file in enumerate(eval_files):
                    print(f"\nEvaluating checkpoint {checkpoint_idx} (model: {unlearned_model_file})")
                    
                    # Check if file exists
                    if not os.path.exists(unlearned_model_file):
                        print(f"  Warning: Unlearned model file not found: {unlearned_model_file}")
                        print(f"  Skipping this checkpoint")
                        continue
                    
                    # Load unlearned model
                    unlearned_checkpoint = torch.load(unlearned_model_file, map_location=trainer.device)
                    trainer.model.load_state_dict(unlearned_checkpoint['state_dict'])
                    trainer.model.load_other_parameter(unlearned_checkpoint.get('other_parameter'))
                    trainer.model.eval()
                    unlearned_model = trainer.model
                    
                    # Load retrained model (baseline)
                    if retrained_model_path and os.path.exists(retrained_model_path):
                        retrained_checkpoint = torch.load(retrained_model_path, map_location=trainer.device)
                        # Create a copy of the model for retrained baseline
                        retrained_model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
                        retrained_model.load_state_dict(retrained_checkpoint['state_dict'])
                        retrained_model.load_other_parameter(retrained_checkpoint.get('other_parameter'))
                        retrained_model.eval()
                    else:
                        print(f"  Warning: Retrained model not found at {retrained_model_path}")
                        print(f"  Skipping RULI Privacy evaluation for this checkpoint")
                        continue
                    
                    # Run evaluation
                    eval_results, summary = evaluator.evaluate_unlearning(
                        unlearned_model=unlearned_model,
                        retrained_model=retrained_model,
                        d_target=d_target,
                        overlap_flags=overlap_flags,
                    )
                    
                    # Print summary
                    print(f"\n  RULI Privacy Evaluation Results:")
                    print(f"    Samples evaluated: {summary['n_samples_evaluated']}")
                    print(f"    Samples in forget set (skipped): {summary['n_samples_in_forget_set']}")
                    print(f"\n    Unlearned Model:")
                    print(f"      Mean MIA Score: {summary['unlearned_model']['mean_mia_score']:.4f}")
                    print(f"      Detection Rate: {summary['unlearned_model']['detection_rate']:.4f}")
                    print(f"\n    Retrained Model (Baseline):")
                    print(f"      Mean MIA Score: {summary['retrained_model']['mean_mia_score']:.4f}")
                    print(f"      Detection Rate: {summary['retrained_model']['detection_rate']:.4f}")
                    print(f"\n    Comparison:")
                    print(f"      Score Difference: {summary['comparison']['score_difference']:.4f}")
                    print(f"      Detection Rate Difference: {summary['comparison']['detection_rate_difference']:.4f}")
                    
                    # Store results
                    results.append({
                        "model_file": unlearned_model_file,
                        "checkpoint_idx": checkpoint_idx,
                        "evaluation_type": "ruli_privacy",
                        "sensitive_category": sensitive_category,
                        "summary": summary,
                        "detailed_results": eval_results,
                    })
                    
                    # Clean up
                    del retrained_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            print("\n" + "="*60)
            print("RULI Privacy Evaluation Complete")
            print("="*60)
            
        except ImportError as e:
            print(f"Warning: Could not import RULI Privacy Evaluator: {e}")
        except Exception as e:
            print(f"Error during RULI Privacy evaluation: {e}")
            traceback.print_exc()

    # Save sensitive results to CSV if sensitive_eval_only mode is enabled
    if sensitive_eval_only and config.get('sensitive_category') is not None:
        # Filter results to get only sensitive item evaluation results
        sensitive_results = [r for r in results if 'sensitive_category' in r and 'topk' in r]
        if sensitive_results:
            save_sensitive_results_to_csv(
                sensitive_results,
                config,
                retrain_flag=False,
                unlearning_algorithm=unlearning_algorithm
            )

    return results

def run_recboles(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole(
        *args[:3],
        **kwargs,
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    
    # Automatically compute inverse frequency weights for Sets2Sets model
    if hasattr(model, 'compute_inverse_freq_weights_from_dataset'):
        logger.info("Computing inverse frequency weights from training data...")
        model.compute_inverse_freq_weights_from_dataset(train_data)
        logger.info("Inverse frequency weights computed.")
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config, unlearning=False, spam=config["spam"])
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset, spam=config["spam"])

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    
    # Automatically compute inverse frequency weights for Sets2Sets model
    if hasattr(model, 'compute_inverse_freq_weights_from_dataset'):
        logger.info("Computing inverse frequency weights from training data...")
        model.compute_inverse_freq_weights_from_dataset(train_data)
        logger.info("Inverse frequency weights computed.")
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
