"""
RULI Privacy Evaluator for Game 2 MIA Evaluation

This module implements the RULI Privacy Evaluator based on Game 2 from the
usenixsecurity25-naderloui paper for Membership Inference Attack (MIA) evaluation
of unlearning.
"""

import os
import json
import numpy as np
import torch
import random
from typing import List, Dict, Optional, Tuple
from logging import getLogger
import pandas as pd
from collections import defaultdict

import torch.nn.functional as F

try:
    from sklearn.neighbors import KernelDensity
    KDE_AVAILABLE = True
except ImportError:
    KDE_AVAILABLE = False
    logger = getLogger()
    logger.warning("sklearn not available. KDE-based RULI will fall back to simplified approach.")


logger = getLogger()


class RULIPrivacyEvaluator:
    """
    RULI Privacy Evaluator for Game 2 MIA evaluation of unlearning.
    
    This evaluator uses shadow models to perform membership inference attacks
    on unlearned models, following the Game 2 methodology from the paper.
    """
    
    def __init__(
        self,
        config,
        dataset,
        train_data,
        test_data,
        shadow_models_dir: Optional[str] = None,
        k: int = 8,
        beta_threshold: float = 0.5,
        n_population_samples: int = 2500,
        unlearning_algorithm: Optional[str] = None,
    ):
        """
        Initialize RULI Privacy Evaluator.
        
        Args:
            config: Configuration object
            dataset: Dataset object
            train_data: Training data loader
            test_data: Test data loader
            shadow_models_dir: Directory containing shadow models (default: {model_dir}/shadow_models/)
            k: Number of shadow models (must match shadow model computation)
            beta_threshold: Decision threshold for membership inference
            n_population_samples: Number of population samples for calibration
            unlearning_algorithm: Specific unlearning algorithm to use for Qh (if multiple available)
        """
        self.config = config
        self.dataset = dataset
        self.train_data = train_data
        self.test_data = test_data
        self.k = k
        self.beta_threshold = beta_threshold
        self.n_population_samples = n_population_samples
        self.unlearning_algorithm = unlearning_algorithm
        
        # Determine shadow models directory
        # Default structure: ./saved/shadow_models/{dataset}/{model}/
        if shadow_models_dir is None:
            shadow_models_base = os.path.join("saved", "shadow_models")
            shadow_models_dir = os.path.join(
                shadow_models_base,
                self.config["dataset"],
                self.config["model"]
            )
        self.shadow_models_dir = shadow_models_dir
        
        # Load shadow models
        self.shadow_models = self._load_shadow_models()
        logger.info(f"Loaded {len(self.shadow_models)} shadow models from {shadow_models_dir}")
        
        # Load unlearned shadow models if available (for Qh distribution)
        self.unlearned_shadow_models = self._load_unlearned_shadow_models()
        if len(self.unlearned_shadow_models) > 0:
            logger.info(f"Loaded {len(self.unlearned_shadow_models)} unlearned shadow models")
        else:
            logger.warning("No unlearned shadow models found. Qh distribution will use Qout as approximation.")
        
        # Track partition-user relationships for smart Qh selection
        # This maps user_id -> list of partition indices that EXCLUDE this user
        # (these partitions can provide Qh observations for samples from this user)
        self.user_to_excluded_partitions = self._build_user_partition_map()
        
        # Store field names
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        
        # Observation cache for KDE: stores observations per target sample
        # Structure: {sample_id: {"qu": [...], "qh": [...], "qout": [...]}}
        self.observation_cache = defaultdict(lambda: {"qu": [], "qh": [], "qout": [], "qin": []})
        
        # KDE bandwidth parameter (can be tuned)
        self.kde_bandwidth = 0.1
        
        # Track if we need to collect observations
        self._observations_collected = False
    
    def _load_shadow_models(self) -> List[torch.nn.Module]:
        """Load pre-computed shadow models from disk."""
        shadow_models = []
        
        # Try to load metadata first
        # Include dataset and model in metadata filename to match save location
        metadata_path = os.path.join(
            self.shadow_models_dir,
            f"shadow_models_metadata_{self.config['model']}_seed_{self.config['seed']}_dataset_{self.config['dataset']}.json"
        )
        
        # If not found, try the old format (for backward compatibility)
        if not os.path.exists(metadata_path):
            old_metadata_path = os.path.join(
                self.shadow_models_dir,
                f"shadow_models_metadata_seed_{self.config['seed']}.json"
            )
            if os.path.exists(old_metadata_path):
                metadata_path = old_metadata_path
                logger.info(f"Using legacy metadata format: {old_metadata_path}")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for partition_info in metadata["partitions"]:
                model_path = partition_info["model_path"]
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.config["device"])
                    
                    # Recreate model
                    from recbole.utils import get_model
                    shadow_model = get_model(self.config["model"])(
                        self.config, self.train_data._dataset
                    ).to(self.config["device"])
                    shadow_model.load_state_dict(checkpoint["state_dict"])
                    shadow_model.load_other_parameter(checkpoint.get("other_parameter"))
                    shadow_model.eval()
                    shadow_models.append(shadow_model)
                else:
                    logger.warning(f"Shadow model not found: {model_path}")
        else:
            # Fallback: try to load models directly by naming convention
            logger.warning(f"Metadata not found at {metadata_path}, trying direct loading")
            for partition_idx in range(self.k):
                model_filename = (
                    f"shadow_model_{self.config['model']}_seed_{self.config['seed']}_"
                    f"dataset_{self.config['dataset']}_partition_{partition_idx}.pth"
                )
                model_path = os.path.join(self.shadow_models_dir, model_filename)
                
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.config["device"])
                    
                    from recbole.utils import get_model
                    shadow_model = get_model(self.config["model"])(
                        self.config, self.train_data._dataset
                    ).to(self.config["device"])
                    shadow_model.load_state_dict(checkpoint["state_dict"])
                    shadow_model.load_other_parameter(checkpoint.get("other_parameter"))
                    shadow_model.eval()
                    shadow_models.append(shadow_model)
                else:
                    logger.warning(f"Shadow model not found: {model_path}")
        
        if len(shadow_models) == 0:
            raise FileNotFoundError(
                f"No shadow models found in {self.shadow_models_dir}. "
                f"Please run shadow model computation first."
            )
        
        if len(shadow_models) != self.k:
            logger.warning(
                f"Expected {self.k} shadow models, but loaded {len(shadow_models)}. "
                f"Proceeding with available models."
            )
        
        return shadow_models
    
    def _load_unlearned_shadow_models(self) -> List[torch.nn.Module]:
        """Load pre-computed unlearned shadow models from disk (Qh distribution)."""
        unlearned_shadow_models = []
        
        # Try to load metadata first
        metadata_path = os.path.join(
            self.shadow_models_dir,
            f"shadow_models_metadata_{self.config['model']}_seed_{self.config['seed']}_dataset_{self.config['dataset']}.json"
        )
        
        # If not found, try the old format
        if not os.path.exists(metadata_path):
            old_metadata_path = os.path.join(
                self.shadow_models_dir,
                f"shadow_models_metadata_seed_{self.config['seed']}.json"
            )
            if os.path.exists(old_metadata_path):
                metadata_path = old_metadata_path
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if unlearned models were created
            if not metadata.get("create_unlearned_models", False):
                return unlearned_shadow_models
            
            # Get list of available algorithms
            available_algorithms = metadata.get("unlearning_algorithms", [])
            if not available_algorithms:
                # Fallback: try to infer from old format
                available_algorithms = [metadata.get("unlearning_algorithm")] if metadata.get("unlearning_algorithm") else []
            
            # Select algorithm to use
            if self.unlearning_algorithm:
                if self.unlearning_algorithm not in available_algorithms:
                    logger.warning(
                        f"Requested algorithm '{self.unlearning_algorithm}' not found in available algorithms: {available_algorithms}. "
                        f"Using first available: {available_algorithms[0] if available_algorithms else 'none'}"
                    )
                    algorithm_to_use = available_algorithms[0] if available_algorithms else None
                else:
                    algorithm_to_use = self.unlearning_algorithm
            else:
                # Use first available algorithm
                algorithm_to_use = available_algorithms[0] if available_algorithms else None
            
            if algorithm_to_use:
                logger.info(f"Loading unlearned shadow models for algorithm: {algorithm_to_use}")
            
            for partition_info in metadata["partitions"]:
                # Try new format first (multiple algorithms)
                unlearned_paths = partition_info.get("unlearned_model_paths", {})
                if unlearned_paths and algorithm_to_use:
                    unlearned_model_path = unlearned_paths.get(algorithm_to_use)
                else:
                    # Fallback to old format (single algorithm)
                    unlearned_model_path = partition_info.get("unlearned_model_path")
                
                if unlearned_model_path and os.path.exists(unlearned_model_path):
                    checkpoint = torch.load(unlearned_model_path, map_location=self.config["device"])
                    
                    from recbole.utils import get_model
                    unlearned_model = get_model(self.config["model"])(
                        self.config, self.train_data._dataset
                    ).to(self.config["device"])
                    unlearned_model.load_state_dict(checkpoint["state_dict"])
                    unlearned_model.load_other_parameter(checkpoint.get("other_parameter"))
                    unlearned_model.eval()
                    unlearned_shadow_models.append(unlearned_model)
                else:
                    logger.debug(f"Unlearned shadow model not found for partition {partition_info.get('partition_idx')} (algorithm: {algorithm_to_use})")
        else:
            # Fallback: try to load by naming convention
            # Determine which algorithm to try first
            algorithms_to_try = [self.unlearning_algorithm] if self.unlearning_algorithm else \
                               ["scif", "kookmin", "fanchuan", "gif", "ceu", "idea", "seif"]
            
            for partition_idx in range(self.k):
                found = False
                for algo in algorithms_to_try:
                    model_filename = (
                        f"shadow_model_unlearned_{self.config['model']}_seed_{self.config['seed']}_"
                        f"dataset_{self.config['dataset']}_partition_{partition_idx}_algorithm_{algo}.pth"
                    )
                    model_path = os.path.join(self.shadow_models_dir, model_filename)
                    
                    if os.path.exists(model_path):
                        checkpoint = torch.load(model_path, map_location=self.config["device"])
                        
                        from recbole.utils import get_model
                        unlearned_model = get_model(self.config["model"])(
                            self.config, self.train_data._dataset
                        ).to(self.config["device"])
                        unlearned_model.load_state_dict(checkpoint["state_dict"])
                        unlearned_model.load_other_parameter(checkpoint.get("other_parameter"))
                        unlearned_model.eval()
                        unlearned_shadow_models.append(unlearned_model)
                        found = True
                        break  # Found one for this partition
                
                if not found and self.unlearning_algorithm:
                    logger.warning(
                        f"No unlearned shadow model found for partition {partition_idx} "
                        f"with algorithm {self.unlearning_algorithm}"
                    )
        
        return unlearned_shadow_models
    
    def _build_user_partition_map(self) -> Dict[int, List[int]]:
        """
        Build a mapping from user_id to partition indices that exclude this user.
        
        This allows us to efficiently find which unlearned shadow models can provide
        Qh observations for a given target sample.
        
        Returns:
            Dictionary mapping user_id -> list of partition indices that exclude this user
        """
        user_to_partitions = {}
        
        # Try to load metadata to get partition information
        metadata_path = os.path.join(
            self.shadow_models_dir,
            f"shadow_models_metadata_{self.config['model']}_seed_{self.config['seed']}_dataset_{self.config['dataset']}.json"
        )
        
        if not os.path.exists(metadata_path):
            old_metadata_path = os.path.join(
                self.shadow_models_dir,
                f"shadow_models_metadata_seed_{self.config['seed']}.json"
            )
            if os.path.exists(old_metadata_path):
                metadata_path = old_metadata_path
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get all unique users from dataset
            all_users = set(self.dataset.inter_feat[self.uid_field].unique())
            
            # For each partition, track which users are excluded
            for partition_info in metadata.get("partitions", []):
                partition_idx = partition_info.get("partition_idx")
                excluded_users = partition_info.get("excluded_users", [])
                
                # For each excluded user, add this partition to their list
                for user_id in excluded_users:
                    if user_id not in user_to_partitions:
                        user_to_partitions[user_id] = []
                    user_to_partitions[user_id].append(partition_idx)
            
            # For users not in any excluded list, they appear in all partitions
            # So no partitions exclude them (they can't provide Qh)
            for user_id in all_users:
                if user_id not in user_to_partitions:
                    user_to_partitions[user_id] = []
        else:
            # Fallback: use k_subsets_exact_np logic
            # Each user appears in exactly k/2 partitions, so is excluded from k/2 partitions
            from recbole.quick_start.quick_start import k_subsets_exact_np
            unique_users = np.sort(np.unique(self.dataset.inter_feat[self.uid_field].to_numpy()))
            user_subsets = k_subsets_exact_np(unique_users, k=self.k)
            
            for partition_idx, excluded_users in enumerate(user_subsets):
                for user_id in excluded_users:
                    if user_id not in user_to_partitions:
                        user_to_partitions[user_id] = []
                    user_to_partitions[user_id].append(partition_idx)
        
        logger.debug(
            f"Built user-partition map: {len(user_to_partitions)} users, "
            f"average {np.mean([len(v) for v in user_to_partitions.values()]):.1f} excluded partitions per user"
        )
        
        return user_to_partitions
    
    def construct_d_target(
        self,
        sensitive_category: str,
        forget_set: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[Dict], List[bool]]:
        """
        Construct D_target from sensitive interactions.
        
        D_target can include samples from the forget set, but overlaps are tracked
        to avoid double-processing during unlearning.
        
        Args:
            sensitive_category: Sensitive category name
            forget_set: DataFrame containing forget set (D_forget) interactions
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (D_target interactions, overlap flags indicating which are in D_forget)
        """
        # Set seed for reproducibility
        old_np_state = None
        old_random_state = None
        if seed is not None:
            old_np_state = np.random.get_state()
            old_random_state = random.get_state()
            np.random.seed(seed)
            random.seed(seed)
        
        try:
            # Load sensitive items
            sensitive_items_path = os.path.join(
                self.config["data_path"],
                f"sensitive_asins_{sensitive_category}.txt"
            )
            
            if not os.path.exists(sensitive_items_path):
                sensitive_items_path = os.path.join(
                    self.config["data_path"],
                    f"sensitive_products_{sensitive_category}.txt"
                )
            
            if not os.path.exists(sensitive_items_path):
                raise FileNotFoundError(
                    f"Sensitive items file not found for category '{sensitive_category}'. "
                    f"Tried:\n  - sensitive_asins_{sensitive_category}.txt\n  "
                    f"- sensitive_products_{sensitive_category}.txt"
                )
            
            with open(sensitive_items_path, 'r') as f:
                sensitive_items_raw = [line.strip() for line in f if line.strip()]
            
            # Map to internal item IDs
            sensitive_item_ids = set()
            for item_token in sensitive_items_raw:
                try:
                    item_id = self.dataset.token2id(self.iid_field, item_token)
                    sensitive_item_ids.add(item_id)
                except (ValueError, KeyError):
                    pass
            
            logger.info(
                f"Loaded {len(sensitive_item_ids)} sensitive items "
                f"(out of {len(sensitive_items_raw)} raw items)"
            )
            
            # Find all sensitive interactions in full dataset
            user_ids = self.dataset.inter_feat[self.uid_field].to_numpy()
            item_ids = self.dataset.inter_feat[self.iid_field].to_numpy()
            
            sensitive_interactions = []
            
            if self.config["task_type"] == "NBR":
                # For NBR: find baskets containing sensitive items
                # Each row is a basket (user + history + target basket)
                target_items_field = "target_item_list"
                target_length_field = "target_item_length"
                
                if target_items_field in self.dataset.inter_feat.columns:
                    for idx in range(len(self.dataset.inter_feat)):
                        user_id = int(user_ids[idx])
                        target_items = self.dataset.inter_feat[target_items_field].iloc[idx]
                        target_length = int(self.dataset.inter_feat[target_length_field].iloc[idx])
                        
                        # Check if target basket contains any sensitive items
                        if isinstance(target_items, torch.Tensor):
                            target_items_list = target_items[:target_length].cpu().numpy().tolist()
                        elif isinstance(target_items, (list, tuple)):
                            target_items_list = list(target_items[:target_length])
                        else:
                            continue
                        
                        # Filter out padding (-1) and check for sensitive items
                        target_items_list = [int(i) for i in target_items_list if int(i) > 0]
                        if any(item_id in sensitive_item_ids for item_id in target_items_list):
                            interaction = {
                                "user_id": user_id,
                                "basket_items": target_items_list,
                                "index": int(idx),
                            }
                            # Store full interaction data for NBR
                            if "history_item_matrix" in self.dataset.inter_feat.columns:
                                interaction["history_item_matrix"] = self.dataset.inter_feat["history_item_matrix"].iloc[idx]
                            if "history_basket_length" in self.dataset.inter_feat.columns:
                                interaction["history_basket_length"] = int(self.dataset.inter_feat["history_basket_length"].iloc[idx])
                            if "history_item_length_per_basket" in self.dataset.inter_feat.columns:
                                interaction["history_item_length_per_basket"] = self.dataset.inter_feat["history_item_length_per_basket"].iloc[idx]
                            interaction["target_length"] = target_length
                            
                            sensitive_interactions.append(interaction)
                else:
                    logger.warning("NBR task but target_item_list field not found. Falling back to item-level.")
                    # Fallback to item-level (not ideal for NBR)
                    sensitive_mask = np.isin(item_ids, list(sensitive_item_ids))
                    for idx in np.where(sensitive_mask)[0]:
                        interaction = {
                            "user_id": int(user_ids[idx]),
                            "item_id": int(item_ids[idx]),
                            "index": int(idx),
                        }
                        sensitive_interactions.append(interaction)
            else:
                # For CF/SBR: find individual interactions with sensitive items
                sensitive_mask = np.isin(item_ids, list(sensitive_item_ids))
                
                for idx in np.where(sensitive_mask)[0]:
                    interaction = {
                        "user_id": int(user_ids[idx]),  # For SBR, this is actually session_id (uid_field)
                        "item_id": int(item_ids[idx]),
                        "index": int(idx),
                    }
                    # Add other fields if present
                    if "rating" in self.dataset.inter_feat.columns:
                        interaction["rating"] = float(self.dataset.inter_feat["rating"].iloc[idx])
                    if "timestamp" in self.dataset.inter_feat.columns:
                        interaction["timestamp"] = float(self.dataset.inter_feat["timestamp"].iloc[idx])
                    
                    # For SBR: also store sequence information if available
                    if self.config["task_type"] == "SBR":
                        # Store sequence fields for SBR models that need them
                        if "item_id_list" in self.dataset.inter_feat.columns:
                            interaction["item_id_list"] = self.dataset.inter_feat["item_id_list"].iloc[idx]
                        if "item_length" in self.dataset.inter_feat.columns:
                            interaction["item_length"] = int(self.dataset.inter_feat["item_length"].iloc[idx])
                        # Also store actual user_id if available (separate from session_id)
                        if "user_id" in self.dataset.inter_feat.columns and self.uid_field != "user_id":
                            interaction["actual_user_id"] = int(self.dataset.inter_feat["user_id"].iloc[idx])
                    
                    sensitive_interactions.append(interaction)
            
            logger.info(f"Found {len(sensitive_interactions)} sensitive interactions")
            
            # Create forget set lookup for overlap tracking
            forget_set_lookup = set()
            if forget_set is not None:
                if self.config["task_type"] == "NBR":
                    # For NBR: track baskets (user + basket items set)
                    for _, row in forget_set.iterrows():
                        user_token = str(row[self.uid_field])
                        try:
                            user_id = self.dataset.token2id(self.uid_field, user_token)
                            # For NBR, we need to check if the basket matches
                            # Store as (user_id, frozenset of basket items) for lookup
                            if "target_item_list" in row:
                                basket_items = row["target_item_list"]
                                if isinstance(basket_items, (list, tuple)):
                                    basket_set = frozenset(int(i) for i in basket_items if int(i) > 0)
                                    forget_set_lookup.add((user_id, basket_set))
                        except (ValueError, KeyError):
                            pass
                else:
                    # For CF/SBR: track (uid_field, item_id) pairs
                    # Note: For SBR, uid_field is session_id, not user_id
                    for _, row in forget_set.iterrows():
                        uid_token = str(row[self.uid_field])  # session_id for SBR, user_id for CF
                        item_token = str(row[self.iid_field])
                        try:
                            uid_value = self.dataset.token2id(self.uid_field, uid_token)
                            item_id = self.dataset.token2id(self.iid_field, item_token)
                            forget_set_lookup.add((uid_value, item_id))
                        except (ValueError, KeyError):
                            pass
            
            # Split sensitive interactions by train/test split
            train_interactions = []
            test_interactions = []
            
            # Determine which interactions are in train vs test
            # Note: For SBR, uid_field is session_id, so pairs are (session_id, item_id)
            # For CF, uid_field is user_id, so pairs are (user_id, item_id)
            train_uid_item_pairs = set()
            test_uid_item_pairs = set()
            
            # Get train pairs
            for batch in self.train_data:
                if isinstance(batch, dict):
                    batch_uid_values = batch[self.uid_field].cpu().numpy()  # session_id for SBR, user_id for CF
                    batch_item_ids = batch[self.iid_field].cpu().numpy()
                    for uid_val, i in zip(batch_uid_values, batch_item_ids):
                        train_uid_item_pairs.add((int(uid_val), int(i)))
            
            # Get test pairs
            for batch in self.test_data:
                if isinstance(batch, dict):
                    batch_uid_values = batch[self.uid_field].cpu().numpy()  # session_id for SBR, user_id for CF
                    batch_item_ids = batch[self.iid_field].cpu().numpy()
                    for uid_val, i in zip(batch_uid_values, batch_item_ids):
                        test_uid_item_pairs.add((int(uid_val), int(i)))
            
            # Categorize sensitive interactions
            for interaction in sensitive_interactions:
                if self.config["task_type"] == "NBR":
                    # For NBR: check if the basket index is in train/test
                    idx = interaction["index"]
                    # Check if this index appears in train or test data
                    # We'll use a simpler approach: check if user appears in train/test
                    user_id = interaction["user_id"]
                    user_in_train = any(u == user_id for u, _ in train_uid_item_pairs)
                    user_in_test = any(u == user_id for u, _ in test_uid_item_pairs)
                    
                    # For NBR, we categorize based on phase if available, otherwise by user presence
                    if "nbr_phase" in self.dataset.inter_feat.columns:
                        phase = self.dataset.inter_feat["nbr_phase"].iloc[idx]
                        if phase == "train":
                            train_interactions.append(interaction)
                        elif phase == "test":
                            test_interactions.append(interaction)
                        # If not categorized, use user presence as fallback
                        elif user_in_train and not user_in_test:
                            train_interactions.append(interaction)
                        elif user_in_test:
                            test_interactions.append(interaction)
                    else:
                        # Fallback: use user presence
                        if user_in_train:
                            train_interactions.append(interaction)
                        elif user_in_test:
                            test_interactions.append(interaction)
                else:
                    # For CF/SBR: check (uid_field, item_id) pair
                    # Note: For SBR, "user_id" in interaction dict is actually session_id (uid_field)
                    # For CF, "user_id" in interaction dict is user_id (uid_field)
                    pair = (interaction["user_id"], interaction["item_id"])
                    if pair in train_uid_item_pairs:
                        train_interactions.append(interaction)
                    elif pair in test_uid_item_pairs:
                        test_interactions.append(interaction)
                    # If not in either, we'll include it but note it's not in standard splits
            
            logger.info(
                f"Split sensitive interactions: {len(train_interactions)} train, "
                f"{len(test_interactions)} test"
            )
            
            # Sample from both train and test to construct D_target
            # Ensure balanced representation
            min_samples = min(len(train_interactions), len(test_interactions))
            if min_samples > 0:
                # Sample equal amounts from both
                n_samples_per_split = min(min_samples, self.n_population_samples // 2)
                
                train_samples = random.sample(
                    train_interactions,
                    min(n_samples_per_split, len(train_interactions))
                )
                test_samples = random.sample(
                    test_interactions,
                    min(n_samples_per_split, len(test_interactions))
                )
                
                d_target = train_samples + test_samples
            else:
                # If one split is empty, use all from the other
                d_target = train_interactions if len(train_interactions) > 0 else test_interactions
                d_target = random.sample(
                    d_target,
                    min(len(d_target), self.n_population_samples)
                )
            
            # Mark overlaps with forget set
            overlap_flags = []
            for interaction in d_target:
                if self.config["task_type"] == "NBR":
                    # For NBR: check if basket matches forget set
                    user_id = interaction["user_id"]
                    basket_items = frozenset(interaction.get("basket_items", []))
                    is_in_forget = (user_id, basket_items) in forget_set_lookup
                    overlap_flags.append(is_in_forget)
                else:
                    # For CF/SBR: check (uid_field, item_id) pair
                    # Note: For SBR, "user_id" is actually session_id (uid_field)
                    # For CF, "user_id" is user_id (uid_field)
                    pair = (interaction["user_id"], interaction["item_id"])
                    is_in_forget = pair in forget_set_lookup
                    overlap_flags.append(is_in_forget)
            
            n_overlaps = sum(overlap_flags)
            logger.info(
                f"Constructed D_target with {len(d_target)} samples "
                f"({len(d_target) - n_overlaps} not in forget set, {n_overlaps} in forget set)"
            )
            
            return d_target, overlap_flags
            
        finally:
            # Restore random state
            if seed is not None:
                if old_np_state is not None:
                    np.random.set_state(old_np_state)
                if old_random_state is not None:
                    random.setstate(old_random_state)
    
    def _get_observation(self, model: torch.nn.Module, sample: Dict) -> float:
        """
        Get observation (log probability) from a model for a sample.
        
        Args:
            model: Model to query
            sample: Sample to get observation for
        
        Returns:
            Log probability observation
        """
        model.eval()
        with torch.no_grad():
            interaction = self._create_interaction(sample)
            scores = model.full_sort_predict(interaction)
            log_probs = F.log_softmax(scores, dim=-1)
            
            if self.config["task_type"] == "NBR":
                basket_items = sample.get("basket_items", [])
                if basket_items:
                    basket_log_probs = [log_probs[0, item_id].item() for item_id in basket_items if item_id > 0]
                    if basket_log_probs:
                        return np.logaddexp.reduce(basket_log_probs) - np.log(len(basket_log_probs))
                    else:
                        return np.log(1e-10)
                else:
                    return np.log(1e-10)
            else:
                item_id = sample["item_id"]
                return log_probs[0, item_id].item()
    
    def _collect_observations_for_sample(
        self,
        target_sample: Dict,
        unlearned_shadow_models: Optional[List[torch.nn.Module]] = None,
    ):
        """
        Collect observations from shadow models for a target sample.
        
        According to RULI Algorithm 1, we need:
        - Qout: Observations from models trained WITHOUT target sample (Out)
        - Qh: Observations from models trained WITHOUT target sample, then unlearned (Held-out)
        - Qin: Observations from models trained WITH target sample (In) - not available with current setup
        - Qu: Observations from models trained WITH target sample, then unlearned (Unlearned) - not available with current setup
        
        Args:
            target_sample: Target sample to collect observations for
            unlearned_shadow_models: Optional list of unlearned shadow models (for Qh)
        """
        sample_id = self._get_sample_id(target_sample)
        
        # Collect Qout observations (shadow models trained without target sample)
        if len(self.observation_cache[sample_id]["qout"]) == 0:
            for shadow_model in self.shadow_models:
                obs = self._get_observation(shadow_model, target_sample)
                self.observation_cache[sample_id]["qout"].append(obs)
        
        # Collect Qh observations (held-out: shadow models without target, then unlearned)
        # Use instance unlearned_shadow_models if available, otherwise use provided parameter
        models_to_use = unlearned_shadow_models if unlearned_shadow_models is not None else self.unlearned_shadow_models
        
        if len(models_to_use) > 0 and len(self.observation_cache[sample_id]["qh"]) == 0:
            # Smart selection: only use unlearned shadow models from partitions that exclude
            # the target sample's user (these are proper Qh observations)
            target_user_id = target_sample.get("user_id")
            if target_user_id is not None and target_user_id in self.user_to_excluded_partitions:
                # Get partition indices that exclude this user
                valid_partition_indices = self.user_to_excluded_partitions[target_user_id]
                
                # Use only unlearned models from valid partitions
                # Note: We assume unlearned_shadow_models[i] corresponds to partition i
                if len(valid_partition_indices) > 0 and len(models_to_use) == len(self.shadow_models):
                    # Models are in partition order
                    for partition_idx in valid_partition_indices:
                        if partition_idx < len(models_to_use):
                            obs = self._get_observation(models_to_use[partition_idx], target_sample)
                            self.observation_cache[sample_id]["qh"].append(obs)
                    
                    if len(self.observation_cache[sample_id]["qh"]) == 0:
                        logger.warning(
                            f"No valid Qh observations found for sample {sample_id} "
                            f"(user {target_user_id} excluded from {len(valid_partition_indices)} partitions, "
                            f"but no matching unlearned models available)"
                        )
                else:
                    # Fallback: use all models if we can't match partitions
                    logger.debug(
                        f"Cannot match partitions for sample {sample_id}, using all unlearned models"
                    )
                    for unlearned_model in models_to_use:
                        obs = self._get_observation(unlearned_model, target_sample)
                        self.observation_cache[sample_id]["qh"].append(obs)
            else:
                # Fallback: use all models if user mapping not available
                logger.debug(
                    f"User {target_user_id} not in partition map for sample {sample_id}, using all unlearned models"
                )
                for unlearned_model in models_to_use:
                    obs = self._get_observation(unlearned_model, target_sample)
                    self.observation_cache[sample_id]["qh"].append(obs)
        elif len(self.observation_cache[sample_id]["qh"]) == 0:
            # If no unlearned shadow models available, use Qout as approximation
            # This is a limitation - ideally we'd have pre-computed unlearned shadow models
            self.observation_cache[sample_id]["qh"] = self.observation_cache[sample_id]["qout"].copy()
            logger.debug(
                f"No unlearned shadow models available for sample {sample_id}. "
                f"Using Qout as approximation for Qh."
            )
    
    def _get_sample_id(self, sample: Dict) -> str:
        """Generate a unique ID for a sample."""
        if self.config["task_type"] == "NBR":
            user_id = sample.get("user_id")
            basket_items = tuple(sorted(sample.get("basket_items", [])))
            return f"nbr_{user_id}_{basket_items}"
        else:
            return f"{sample.get('user_id')}_{sample.get('item_id')}"
    
    def compute_mia_score(
        self,
        target_model: torch.nn.Module,
        target_sample: Dict,
        population_samples: List[Dict],
        unlearned_shadow_models: Optional[List[torch.nn.Module]] = None,
        use_kde: bool = True,
    ) -> float:
        """
        Compute MIA score for a sample using Game 2 methodology (RULI) with KDE.
        
        According to the usenixsecurity25-naderloui paper, Game 2 uses a likelihood
        ratio test: Λ(z) = p(θU | Qu(z)) / p(θU | Qh(z))
        
        Where:
        - θU = the unlearned model being tested
        - Qu(z) = distribution of observations from shadow models trained WITH z, then unlearned
        - Qh(z) = distribution of observations from shadow models trained WITHOUT z, then unlearned
        
        Args:
            target_model: Model to test (unlearned or retrained)
            target_sample: Sample to test membership for
            population_samples: Population dataset for calibration
            unlearned_shadow_models: Optional list of unlearned shadow models (for Qh distribution)
            use_kde: Whether to use KDE for distribution estimation (default: True)
        
        Returns:
            MIA score (higher = more likely to be a member, range [0, 1])
        """
        # Collect observations if not already cached
        self._collect_observations_for_sample(target_sample, unlearned_shadow_models)
        
        sample_id = self._get_sample_id(target_sample)
        qu_obs = self.observation_cache[sample_id]["qu"]
        qh_obs = self.observation_cache[sample_id]["qh"]
        qout_obs = self.observation_cache[sample_id]["qout"]
        
        # Get observation from target model (θU)
        target_obs = self._get_observation(target_model, target_sample)
        
        if use_kde and KDE_AVAILABLE and len(qh_obs) > 0:
            # Use KDE-based likelihood ratio (proper RULI implementation)
            # Fit KDE on Qh observations (held-out distribution)
            qh_obs_array = np.array(qh_obs).reshape(-1, 1)
            kde_qh = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian')
            kde_qh.fit(qh_obs_array)
            
            # Compute log likelihood of target observation under Qh
            log_prob_qh = kde_qh.score_samples([[target_obs]])[0]
            
            # For Qu (unlearned distribution), we need models trained WITH target then unlearned
            # Since we don't have these, we use Qout as approximation or fall back to simplified approach
            if len(qu_obs) > 0:
                qu_obs_array = np.array(qu_obs).reshape(-1, 1)
                kde_qu = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian')
                kde_qu.fit(qu_obs_array)
                log_prob_qu = kde_qu.score_samples([[target_obs]])[0]
            else:
                # Fallback: use Qout as approximation for Qu
                # This is not ideal but necessary without proper Qu observations
                logger.debug(
                    f"No Qu observations for sample {sample_id}. "
                    f"Using Qout as approximation."
                )
                if len(qout_obs) > 0:
                    qout_obs_array = np.array(qout_obs).reshape(-1, 1)
                    kde_qout = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian')
                    kde_qout.fit(qout_obs_array)
                    log_prob_qu = kde_qout.score_samples([[target_obs]])[0]
                else:
                    # Ultimate fallback: use target observation itself
                    log_prob_qu = 0.0
            
            # Likelihood ratio in log space: log(Λ) = log p(θU | Qu) - log p(θU | Qh)
            log_likelihood_ratio = log_prob_qu - log_prob_qh
            
            # Compare against population to get MIA score
            population_log_ratios = []
            for pop_sample in population_samples[:self.n_population_samples]:
                pop_obs = self._get_observation(target_model, pop_sample)
                pop_log_prob_qh = kde_qh.score_samples([[pop_obs]])[0]
                if len(qu_obs) > 0:
                    pop_log_prob_qu = kde_qu.score_samples([[pop_obs]])[0]
                elif len(qout_obs) > 0:
                    pop_log_prob_qu = kde_qout.score_samples([[pop_obs]])[0]
                else:
                    pop_log_prob_qu = 0.0
                pop_log_ratio = pop_log_prob_qu - pop_log_prob_qh
                population_log_ratios.append(pop_log_ratio)
            
            # MIA score: fraction of population with lower likelihood ratio
            mia_score = np.mean([log_likelihood_ratio > r for r in population_log_ratios])
            
        else:
            # Fallback to simplified approach without KDE
            if not KDE_AVAILABLE:
                logger.warning("KDE not available. Falling back to simplified approach.")
            
            # Use mean-based approach
            avg_qh = np.mean(qh_obs) if len(qh_obs) > 0 else target_obs
            avg_qu = np.mean(qu_obs) if len(qu_obs) > 0 else np.mean(qout_obs) if len(qout_obs) > 0 else target_obs
            
            # Simplified likelihood ratio
            log_likelihood_ratio = (target_obs - avg_qh) - (target_obs - avg_qu)
            
            # Compare against population
            population_log_ratios = []
            for pop_sample in population_samples[:self.n_population_samples]:
                pop_obs = self._get_observation(target_model, pop_sample)
                pop_avg_qh = np.mean(qh_obs) if len(qh_obs) > 0 else pop_obs
                pop_avg_qu = np.mean(qu_obs) if len(qu_obs) > 0 else np.mean(qout_obs) if len(qout_obs) > 0 else pop_obs
                pop_log_ratio = (pop_obs - pop_avg_qh) - (pop_obs - pop_avg_qu)
                population_log_ratios.append(pop_log_ratio)
            
            mia_score = np.mean([log_likelihood_ratio > r for r in population_log_ratios])
        
        return mia_score
    
    def _create_interaction(self, sample: Dict) -> Dict:
        """Create interaction tensor from sample dictionary."""
        interaction = {}
        user_id = sample["user_id"]
        
        if self.config["task_type"] == "NBR":
            # For NBR: need history and target basket
            idx = sample.get("index")
            if idx is not None and idx < len(self.dataset.inter_feat):
                interaction[self.uid_field] = torch.tensor([user_id], device=self.config["device"])
                
                # Copy history fields from dataset
                history_items_field = "history_item_matrix"
                history_length_field = "history_basket_length"
                history_item_len_field = "history_item_length_per_basket"
                target_items_field = "target_item_list"
                target_length_field = "target_item_length"
                
                if history_items_field in self.dataset.inter_feat.columns:
                    history_data = self.dataset.inter_feat[history_items_field].iloc[idx]
                    if isinstance(history_data, torch.Tensor):
                        interaction[history_items_field] = history_data.unsqueeze(0).to(self.config["device"])
                    else:
                        interaction[history_items_field] = torch.tensor([history_data], device=self.config["device"])
                
                if history_length_field in self.dataset.inter_feat.columns:
                    hist_len = self.dataset.inter_feat[history_length_field].iloc[idx]
                    interaction[history_length_field] = torch.tensor([int(hist_len)], device=self.config["device"])
                
                if history_item_len_field in self.dataset.inter_feat.columns:
                    hist_item_len = self.dataset.inter_feat[history_item_len_field].iloc[idx]
                    if isinstance(hist_item_len, torch.Tensor):
                        interaction[history_item_len_field] = hist_item_len.unsqueeze(0).to(self.config["device"])
                    else:
                        interaction[history_item_len_field] = torch.tensor([hist_item_len], device=self.config["device"])
                
                if target_items_field in self.dataset.inter_feat.columns:
                    target_data = self.dataset.inter_feat[target_items_field].iloc[idx]
                    if isinstance(target_data, torch.Tensor):
                        interaction[target_items_field] = target_data.unsqueeze(0).to(self.config["device"])
                    else:
                        interaction[target_items_field] = torch.tensor([target_data], device=self.config["device"])
                
                if target_length_field in self.dataset.inter_feat.columns:
                    target_len = self.dataset.inter_feat[target_length_field].iloc[idx]
                    interaction[target_length_field] = torch.tensor([int(target_len)], device=self.config["device"])
            else:
                # Fallback: just user_id (not ideal for NBR)
                interaction[self.uid_field] = torch.tensor([user_id], device=self.config["device"])
        elif self.config["task_type"] == "SBR":
            # For SBR, need item sequence
            # Note: user_id in sample is actually session_id (uid_field) for SBR
            session_id = user_id  # In SBR, uid_field is session_id
            session_mask = self.dataset.inter_feat[self.uid_field] == session_id
            session_indices = np.where(session_mask)[0]
            
            if len(session_indices) > 0:
                last_idx = session_indices[-1]
                # Check if dataset has item sequence fields
                if "item_id_list" in self.dataset.inter_feat.columns:
                    item_seq_field = "item_id_list"
                    item_seq_len_field = "item_length"
                    interaction[self.uid_field] = torch.tensor([session_id], device=self.config["device"])
                    item_seq = self.dataset.inter_feat[item_seq_field].iloc[last_idx]
                    if isinstance(item_seq, torch.Tensor):
                        interaction[item_seq_field] = item_seq.unsqueeze(0).to(self.config["device"])
                    else:
                        interaction[item_seq_field] = torch.tensor([item_seq], device=self.config["device"])
                    if item_seq_len_field in self.dataset.inter_feat.columns:
                        seq_len = self.dataset.inter_feat[item_seq_len_field].iloc[last_idx]
                        interaction[item_seq_len_field] = torch.tensor([seq_len], device=self.config["device"])
                else:
                    # Fallback: try ITEM_SEQ field (standard RecBole field name)
                    from recbole.model.abstract_recommender import SequentialRecommender
                    item_seq_field = "item_id_list"  # Try common field names
                    if item_seq_field not in self.dataset.inter_feat.columns:
                        item_seq_field = "item_seq"  # Alternative field name
                    if item_seq_field in self.dataset.inter_feat.columns:
                        interaction[self.uid_field] = torch.tensor([session_id], device=self.config["device"])
                        item_seq = self.dataset.inter_feat[item_seq_field].iloc[last_idx]
                        if isinstance(item_seq, torch.Tensor):
                            interaction[item_seq_field] = item_seq.unsqueeze(0).to(self.config["device"])
                        else:
                            interaction[item_seq_field] = torch.tensor([item_seq], device=self.config["device"])
                    else:
                        # Final fallback: just session_id (uid_field)
                        interaction[self.uid_field] = torch.tensor([session_id], device=self.config["device"])
            else:
                # Session not found, just use session_id
                interaction[self.uid_field] = torch.tensor([session_id], device=self.config["device"])
        else:
            # For CF, just user_id
            interaction[self.uid_field] = torch.tensor(
                [user_id], device=self.config["device"]
            )
        
        return interaction
    
    def create_unlearned_shadow_models(
        self,
        forget_set: Optional[pd.DataFrame] = None,
        unlearning_algorithm: str = "scif",
        **unlearning_kwargs
    ) -> List[torch.nn.Module]:
        """
        Create unlearned versions of shadow models (Qh distribution).
        
        This applies unlearning to each shadow model to create the "held-out" distribution
        (models trained without target sample, then unlearned).
        
        Args:
            forget_set: DataFrame containing forget set interactions
            unlearning_algorithm: Unlearning algorithm to use
            **unlearning_kwargs: Additional arguments for unlearning algorithm
        
        Returns:
            List of unlearned shadow models
        """
        unlearned_shadow_models = []
        
        # This would require implementing unlearning on shadow models
        # For now, return empty list - this is a placeholder for future implementation
        logger.warning(
            "create_unlearned_shadow_models is not yet fully implemented. "
            "Unlearned shadow models should be pre-computed during shadow model creation."
        )
        
        return unlearned_shadow_models
    
    def evaluate_unlearning(
        self,
        unlearned_model: torch.nn.Module,
        retrained_model: torch.nn.Module,
        d_target: List[Dict],
        overlap_flags: List[bool],
        population_samples: Optional[List[Dict]] = None,
        unlearned_shadow_models: Optional[List[torch.nn.Module]] = None,
        use_kde: bool = True,
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate unlearning quality using RULI Privacy (Game 2) with KDE.
        
        Args:
            unlearned_model: Model after unlearning procedure
            retrained_model: Model retrained from scratch without unlearned data
            d_target: Target dataset for MIA evaluation
            overlap_flags: Flags indicating which samples in D_target overlap with D_forget
            population_samples: Population dataset for calibration (if None, uses test_data)
            unlearned_shadow_models: Optional list of unlearned shadow models (for Qh distribution)
            use_kde: Whether to use KDE for distribution estimation (default: True)
        
        Returns:
            Tuple of (detailed results, summary metrics)
        """
        if population_samples is None:
            # Use test data as population
            population_samples = []
            if self.config["task_type"] == "NBR":
                # For NBR: sample baskets from test data
                target_items_field = "target_item_list"
                target_length_field = "target_item_length"
                
                for batch in self.test_data:
                    if isinstance(batch, dict):
                        batch_user_ids = batch[self.uid_field].cpu().numpy()
                        if target_items_field in batch:
                            batch_target_items = batch[target_items_field].cpu().numpy()
                            batch_target_lengths = batch[target_length_field].cpu().numpy() if target_length_field in batch else None
                            
                            for idx, u in enumerate(batch_user_ids):
                                if batch_target_lengths is not None:
                                    target_len = int(batch_target_lengths[idx])
                                    target_items = batch_target_items[idx, :target_len].tolist()
                                else:
                                    # Fallback: use all non-zero items
                                    target_items = [int(i) for i in batch_target_items[idx] if int(i) > 0]
                                
                                if target_items:
                                    sample = {
                                        "user_id": int(u),
                                        "basket_items": target_items,
                                    }
                                    # Try to get index from dataset if possible
                                    # For now, we'll use the basket items as identifier
                                    population_samples.append(sample)
                        else:
                            # Fallback: treat as CF
                            batch_item_ids = batch[self.iid_field].cpu().numpy() if self.iid_field in batch else None
                            if batch_item_ids is not None:
                                for u, i in zip(batch_user_ids, batch_item_ids):
                                    population_samples.append({
                                        "user_id": int(u),
                                        "item_id": int(i),
                                    })
            else:
                # For CF/SBR: sample individual interactions
                for batch in self.test_data:
                    if isinstance(batch, dict):
                        batch_user_ids = batch[self.uid_field].cpu().numpy()
                        batch_item_ids = batch[self.iid_field].cpu().numpy()
                        for u, i in zip(batch_user_ids, batch_item_ids):
                            population_samples.append({
                                "user_id": int(u),
                                "item_id": int(i),
                            })
        
        results = {
            "unlearned_model": {
                "d_target_scores": [],
                "d_target_detected": [],
                "d_target_not_forget_scores": [],
                "d_target_not_forget_detected": [],
            },
            "retrained_model": {
                "d_target_scores": [],
                "d_target_detected": [],
                "d_target_not_forget_scores": [],
                "d_target_not_forget_detected": [],
            },
        }
        
        # Evaluate each sample in D_target
        for sample, is_in_forget in zip(d_target, overlap_flags):
            # Skip samples that are in forget set (to avoid double-processing)
            if is_in_forget:
                continue
            
            # Test on unlearned model
            # Use provided unlearned_shadow_models or fall back to instance's
            models_to_use = unlearned_shadow_models if unlearned_shadow_models is not None else self.unlearned_shadow_models
            unlearned_score = self.compute_mia_score(
                unlearned_model, sample, population_samples,
                unlearned_shadow_models=models_to_use,
                use_kde=use_kde
            )
            results["unlearned_model"]["d_target_scores"].append(unlearned_score)
            results["unlearned_model"]["d_target_detected"].append(
                unlearned_score >= self.beta_threshold
            )
            results["unlearned_model"]["d_target_not_forget_scores"].append(unlearned_score)
            results["unlearned_model"]["d_target_not_forget_detected"].append(
                unlearned_score >= self.beta_threshold
            )
            
            # Test on retrained model (baseline)
            retrained_score = self.compute_mia_score(
                retrained_model, sample, population_samples,
                unlearned_shadow_models=models_to_use,
                use_kde=use_kde
            )
            results["retrained_model"]["d_target_scores"].append(retrained_score)
            results["retrained_model"]["d_target_detected"].append(
                retrained_score >= self.beta_threshold
            )
            results["retrained_model"]["d_target_not_forget_scores"].append(retrained_score)
            results["retrained_model"]["d_target_not_forget_detected"].append(
                retrained_score >= self.beta_threshold
            )
        
        # Compute summary metrics
        summary = {
            "unlearned_model": {
                "mean_mia_score": np.mean(results["unlearned_model"]["d_target_not_forget_scores"]),
                "detection_rate": np.mean(results["unlearned_model"]["d_target_not_forget_detected"]),
            },
            "retrained_model": {
                "mean_mia_score": np.mean(results["retrained_model"]["d_target_not_forget_scores"]),
                "detection_rate": np.mean(results["retrained_model"]["d_target_not_forget_detected"]),
            },
            "comparison": {
                "score_difference": (
                    np.mean(results["unlearned_model"]["d_target_not_forget_scores"]) -
                    np.mean(results["retrained_model"]["d_target_not_forget_scores"])
                ),
                "detection_rate_difference": (
                    np.mean(results["unlearned_model"]["d_target_not_forget_detected"]) -
                    np.mean(results["retrained_model"]["d_target_not_forget_detected"])
                ),
            },
            "n_samples_evaluated": len(results["unlearned_model"]["d_target_not_forget_scores"]),
            "n_samples_in_forget_set": sum(overlap_flags),
        }
        
        return results, summary

