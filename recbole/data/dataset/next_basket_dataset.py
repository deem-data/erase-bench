import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils import FeatureSource, FeatureType, set_color


class NextBasketDataset(Dataset):
    """Dataset that converts merged next-basket JSON dumps into RecBole interactions.

    The source JSON is expected to contain a mapping ``user_id -> List[List[item_id]]`` where each
    inner list corresponds to one basket (already ordered by time). For every user kept in the
    dataset we will build **exactly one** record for each of train/valid/test following the temporal
    leave-one-out rule used in ``A-Next-Basket-Recommendation-Reality-Check``:

        - Train  : history = baskets ``[: n-3]``, target = ``baskets[n-3]``
        - Valid  : history = baskets ``[: n-2]``, target = ``baskets[n-2]``
        - Test   : history = baskets ``[: n-1]``, target = ``baskets[n-1]``

    Only users with ``len(baskets) >= MIN_NBR_BASKET_COUNT`` are kept. History/target baskets are
    padded / truncated to ``MAX_NBR_HISTORY_BASKETS`` and ``MAX_NBR_BASKET_SIZE`` respectively.

    Required config keys (can be set via yaml / command line):

        NEXT_BASKET_JSON         : relative/absolute path to ``*_merged.json``.
        MIN_NBR_BASKET_COUNT     : (default 4) minimal basket count per user.
        MAX_NBR_HISTORY_BASKETS  : (default 32) max number of historical baskets retained.
        MAX_NBR_BASKET_SIZE      : (default 64) max size of every basket.

    Notes
    -----
    - The JSON files themselves are **not** modified.
    - Dataset splitting ignores ``eval_args`` and always returns ``[train, valid, test]`` derived
      from the pre-defined temporal phases.
    """

    TRAIN_PHASE = "train"
    VALID_PHASE = "valid"
    TEST_PHASE = "test"

    def __init__(self, config, unlearning: bool = False, spam: bool = False):
        # Get dataset name from config
        dataset_name = config.get("dataset", None)
        
        if not dataset_name:
            raise ValueError(
                "dataset name must be provided for NextBasketDataset "
                "(e.g., via --dataset command line argument)."
            )
        
        # Automatically construct JSON filename from dataset name
        # Allow override via NEXT_BASKET_JSON if explicitly provided
        self.source_json = (
            config["NEXT_BASKET_JSON"] if "NEXT_BASKET_JSON" in config 
            else f"{dataset_name}_merged.json"
        )
        self.min_basket_count = int(
            config["MIN_NBR_BASKET_COUNT"]
            if "MIN_NBR_BASKET_COUNT" in config
            else 4
        )
        self.max_history_baskets = int(
            config["MAX_NBR_HISTORY_BASKETS"]
            if "MAX_NBR_HISTORY_BASKETS" in config
            else 50
        )
        self.max_basket_items = int(
            config["MAX_NBR_BASKET_SIZE"] if "MAX_NBR_BASKET_SIZE" in config else 100
        )
        
        # Support configurable split ratio (e.g., [0.72, 0.08, 0.20] for 72/8/20 split)
        # If None, uses temporal leave-one-out split (default behavior)
        self.split_ratio = config.get("NEXT_BASKET_SPLIT_RATIO", None)
        if self.split_ratio is not None:
            if not isinstance(self.split_ratio, (list, tuple)) or len(self.split_ratio) != 3:
                raise ValueError(
                    "NEXT_BASKET_SPLIT_RATIO must be a list/tuple of 3 floats "
                    "(e.g., [0.72, 0.08, 0.20]) or None for leave-one-out split"
                )
            # Normalize ratios to sum to 1.0
            total = sum(self.split_ratio)
            self.split_ratio = [r / total for r in self.split_ratio]

        self.history_items_field = "history_item_matrix"
        self.history_length_field = "history_basket_length"
        self.history_item_len_field = "history_item_length_per_basket"
        self.target_items_field = "target_item_list"
        self.target_length_field = "target_item_length"
        self.phase_field = "nbr_phase"

        super().__init__(config, unlearning=unlearning, spam=spam)
        
        # Log split strategy after logger is initialized
        # Debug: log what was read from config
        raw_config_value = config.get("NEXT_BASKET_SPLIT_RATIO", "NOT_FOUND")
        self.logger.info(
            f"[NBR Split Debug] Raw config value for NEXT_BASKET_SPLIT_RATIO: {raw_config_value} "
            f"(type: {type(raw_config_value)})"
        )
        if self.split_ratio is not None:
            self.logger.info(
                f"Using ratio-based temporal split: {self.split_ratio} "
                f"(train={self.split_ratio[0]:.2%}, valid={self.split_ratio[1]:.2%}, test={self.split_ratio[2]:.2%})"
            )
        else:
            self.logger.info("Using temporal leave-one-out split (default)")

    def _resolve_source_path(self, dataset_path: str) -> str:
        if os.path.isabs(self.source_json):
            candidate = self.source_json
        else:
            candidate = os.path.join(dataset_path, self.source_json)
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"Merged next-basket file [{candidate}] does not exist."
            )
        return candidate

    def _load_data(self, token, dataset_path):
        """Override to read merged JSON directly instead of .inter files."""
        json_path = self._resolve_source_path(dataset_path)
        self.logger.info(
            set_color("Loading next-basket interactions from", "green")
            + f" [{json_path}]"
        )

        # Load original data
        with open(json_path, "r") as fp:
            raw_data: Dict[str, List[List[int]]] = json.load(fp)
        
        # Inject fraud baskets if spam=True and not unlearning
        if self.spam and not self.unlearning:
            raw_data = self._inject_fraud_baskets(raw_data, dataset_path)
        
        # Build interactions from (possibly merged) data
        inter_df = self._build_interactions_from_json_data(raw_data)
        self.inter_feat = inter_df
        self.user_feat = None
        self.item_feat = None
        # Allow users to still attach additional feature files if desired.
        self._load_additional_feat(token, dataset_path)
    
    def _inject_fraud_sessions(self):
        """Override base class method to prevent loading .inter files for NBR.
        
        For NBR datasets, fraud baskets are already injected in _load_data() via
        _inject_fraud_baskets(). This method is called by the base Dataset.__init__
        but we don't need it since we handle fraud injection differently (JSON format).
        """
        # Fraud baskets are already injected in _load_data() for NBR datasets
        # No need to load .inter files like SBR datasets do
        pass
    
    def _inject_fraud_baskets(self, raw_data: Dict[str, List[List[int]]], dataset_path: str) -> Dict[str, List[List[int]]]:
        """Inject fraud baskets into the dataset before ID remapping.
        
        This method loads fraud baskets from a separate JSON file and merges them
        with the original merged JSON data. This must happen BEFORE ID remapping
        so that both clean and fraud baskets are remapped together consistently.
        
        Args:
            raw_data: Original data dict loaded from JSON
            dataset_path: Path to dataset directory
            
        Returns:
            Combined data dict with fraud baskets merged in
        """
        import json
        import os
        
        # Construct fraud baskets file path
        unlearning_fraction = self.config['unlearning_fraction']
        fraud_baskets_path = os.path.join(
            dataset_path,
            f"{self.config['dataset']}_fraud_baskets_bandwagon_unpopular_ratio_{unlearning_fraction}_seed_{self.config['unlearn_sample_selection_seed']}.json"
        )
        
        if not os.path.exists(fraud_baskets_path):
            self.logger.warning(f"Fraud baskets file not found: {fraud_baskets_path}")
            self.logger.warning("Training on clean data only!")
            return raw_data
        
        self.logger.info(f"Injecting fraud baskets from: {fraud_baskets_path}")
        
        # Load fraud baskets JSON
        with open(fraud_baskets_path, 'r') as f:
            fraud_data = json.load(f)
        
        # Convert keys to strings for consistency
        fraud_data = {str(k): v for k, v in fraud_data.items()}
        raw_data = {str(k): v for k, v in raw_data.items()}
        
        self.logger.info(f"Loaded {len(fraud_data)} fraud users")
        
        # Merge fraud baskets with original data
        combined_data = {**raw_data, **fraud_data}
        
        self.logger.info(f"Merged data: {len(raw_data)} original users + {len(fraud_data)} fraud users = {len(combined_data)} total users")
        
        return combined_data
    
    def _build_interactions_from_json_data(self, raw_data: Dict[str, List[List[int]]]) -> pd.DataFrame:
        """Build interactions DataFrame from raw data dict (used for merged data with fraud).
        
        This method replicates the logic of _build_interactions_from_json but works with
        an in-memory dict instead of loading from a file.
        """
        rows: List[Dict[str, object]] = []
        dropped_users = 0
        
        # First pass: compute global max basket size if max_basket_items is <= 0 (unlimited)
        if self.max_basket_items <= 0:
            global_max_basket_size = 0
            for user_raw, basket_seq in raw_data.items():
                baskets = self._sanitize_baskets(basket_seq)
                if len(baskets) < self.min_basket_count:
                    continue
                for basket in baskets:
                    global_max_basket_size = max(global_max_basket_size, len(basket))
            # Update max_basket_items to use computed global max
            self.max_basket_items = global_max_basket_size
            self.logger.info(
                f"Computed global max basket size from data: {self.max_basket_items}"
            )
        
        # Second pass: build interactions using the same logic as _build_interactions_from_json
        for user_raw, basket_seq in raw_data.items():
            baskets = self._sanitize_baskets(basket_seq)
            # Filter: need at least MIN_NBR_BASKET_COUNT baskets
            if len(baskets) < self.min_basket_count:
                dropped_users += 1
                continue
            
            user_id = int(user_raw)
            rows.extend(self._build_rows_for_user(user_id, baskets))
        
        if not rows:
            raise ValueError(
                "No user satisfied MIN_NBR_BASKET_COUNT when building NextBasketDataset."
            )
        
        self.logger.info(
            f"Constructed {len(rows)} interaction rows from "
            f"{len(raw_data) - dropped_users} users "
            f"(dropped {dropped_users} users with < {self.min_basket_count} baskets)."
        )
        
        df = pd.DataFrame(rows)
        self._register_field_properties()
        return df

    def _build_interactions_from_json(self, json_path: str) -> pd.DataFrame:
        with open(json_path, "r") as fp:
            raw_data: Dict[str, List[List[int]]] = json.load(fp)

        rows: List[Dict[str, object]] = []
        dropped_users = 0
        
        # First pass: compute global max basket size if max_basket_items is <= 0 (unlimited)
        if self.max_basket_items <= 0:
            global_max_basket_size = 0
            for user_raw, basket_seq in raw_data.items():
                baskets = self._sanitize_baskets(basket_seq)
                if len(baskets) < self.min_basket_count:
                    continue
                for basket in baskets:
                    global_max_basket_size = max(global_max_basket_size, len(basket))
            # Update max_basket_items to use computed global max
            self.max_basket_items = global_max_basket_size
            self.logger.info(
                f"Computed global max basket size from data: {self.max_basket_items}"
            )

        for user_raw, basket_seq in raw_data.items():
            # Filter users matching sets2sets_new.py: need at least MIN_NBR_BASKET_COUNT baskets
            # In sets2sets_new.py: len(history_data[x]) - 2 + len(future_data[x]) - 2 >= 4
            # They count baskets excluding empty ones, so we sanitize first, then count
            baskets = self._sanitize_baskets(basket_seq)
            # Filter: need at least MIN_NBR_BASKET_COUNT baskets (matching sets2sets_new.py requirement)
            if len(baskets) < self.min_basket_count:
                dropped_users += 1
                continue

            user_id = int(user_raw)
            rows.extend(self._build_rows_for_user(user_id, baskets))

        if not rows:
            raise ValueError(
                "No user satisfied MIN_NBR_BASKET_COUNT when building NextBasketDataset."
            )

        self.logger.info(
            f"Constructed {len(rows)} interaction rows from "
            f"{len(raw_data) - dropped_users} users "
            f"(dropped {dropped_users} users with < {self.min_basket_count} baskets)."
        )

        df = pd.DataFrame(rows)
        self._register_field_properties()
        return df

    def _sanitize_baskets(self, baskets: List[List[int]]) -> List[List[int]]:
        sanitized = []
        for basket in baskets:
            if not basket:
                continue
            sanitized.append([int(item) for item in basket])
        return sanitized

    def _build_rows_for_user(
        self, user_id: int, baskets: List[List[int]]
    ) -> List[Dict[str, object]]:
        total = len(baskets)
        
        if self.split_ratio is None:
            # Temporal leave-one-out split (default):
            # - Train: history = baskets[:-3], target = baskets[-3]
            # - Valid: history = baskets[:-2], target = baskets[-2]
            # - Test: history = baskets[:-1] (all baskets except last), target = baskets[-1]
            # Note: Original test uses ALL baskets from history_data as input, which includes
            # baskets[0] to baskets[n-2] if history_data contains n-1 baskets (excluding the test target).
            # Since merged JSON contains baskets[0] to baskets[n-1], test should use baskets[:n-1] as input.
            phases = [
                (self.TRAIN_PHASE, baskets[: total - 3], baskets[total - 3]),
                (self.VALID_PHASE, baskets[: total - 2], baskets[total - 2]),
                (self.TEST_PHASE, baskets[: total - 1], baskets[total - 1]),
            ]
        else:
            # Ratio-based temporal split (e.g., 72/8/20):
            # Split baskets temporally: first 72% train, next 8% valid, last 20% test
            # For next-basket prediction:
            # - Train: history = baskets[:train_end], target = baskets[train_end] (first basket after train period, i.e., first basket of valid period)
            # - Valid: history = baskets[:train_end], target = baskets[train_end] (same as train, for early stopping consistency)
            # - Test: history = baskets[:valid_end], target = baskets[test_start] (first basket of test period)
            train_end = max(1, int(total * self.split_ratio[0]))
            valid_end = min(total - 1, train_end + max(1, int(total * self.split_ratio[1])))
            test_start = valid_end
            
            phases = []
            
            # Train phase: use baskets up to train_end as history, predict first basket after train period
            # This is the first basket of the valid period (basket at index train_end)
            if train_end < total:
                phases.append((
                    self.TRAIN_PHASE,
                    baskets[:train_end],
                    baskets[train_end]
                ))
            elif total > 1:
                # Fallback: if train_end == total, use last basket as target
                phases.append((
                    self.TRAIN_PHASE,
                    baskets[:total - 1],
                    baskets[total - 1]
                ))
            
            # Valid phase: use same history as train, predict same target (for early stopping)
            # Both train and valid predict the first basket after the train period
            if train_end < total:
                phases.append((
                    self.VALID_PHASE,
                    baskets[:train_end],
                    baskets[train_end]
                ))
            
            # Test phase: use baskets up to valid_end as history, predict first basket of test period
            if test_start < total:
                phases.append((
                    self.TEST_PHASE,
                    baskets[:valid_end],
                    baskets[test_start]
                ))
            elif total > 0:
                # Fallback: if test_start == total, use last basket as test target
                phases.append((
                    self.TEST_PHASE,
                    baskets[:total - 1],
                    baskets[total - 1]
                ))

        rows: List[Dict[str, object]] = []
        for phase, history, target in phases:
            history_flat, per_basket_len, history_len = self._encode_history(history)
            target_vec, target_len = self._encode_target(target)

            rows.append(
                {
                    self.uid_field: user_id,
                    self.iid_field: target_vec[0] if target_vec else 0,
                    self.label_field: 1.0,
                    self.phase_field: phase,
                    self.history_items_field: tuple(history_flat),
                    self.history_length_field: history_len,
                    self.history_item_len_field: per_basket_len,
                    self.target_items_field: tuple(target_vec),
                    self.target_length_field: target_len,
                }
            )
        return rows

    def _encode_history(
        self, history: List[List[int]]
    ) -> Tuple[List[int], List[int], int]:
        # Truncate to max_history_baskets (excluding padding baskets)
        # Original structure: [[-1], basket1, ..., basketN, [-1]]
        # We need space for: padding[-1] + baskets + padding[-1]
        # So we can fit at most (max_history_baskets - 2) actual baskets
        max_actual_baskets = self.max_history_baskets - 2
        history = history[-max_actual_baskets :]
        num_hist = len(history)

        matrix = np.zeros(
            (self.max_history_baskets, self.max_basket_items), dtype=np.int64
        )
        per_basket_len = np.zeros(self.max_history_baskets, dtype=np.int64)

        # Match original structure: [[-1], basket1, ..., basketN, [-1]]
        # Position 0: padding basket [-1] (per_basket_len[0] = 0, items are 0 which is padding)
        # Positions 1 to num_hist: actual baskets
        # Position num_hist + 1: padding basket [-1] (per_basket_len[num_hist+1] = 0)
        
        # Position 0 is already padding (zeros), per_basket_len[0] = 0
        
        # Store actual baskets starting at position 1
        for idx, basket in enumerate(history):
            basket_pos = idx + 1  # Position 1, 2, ..., num_hist
            # Truncate basket if it exceeds max_basket_items
            truncated_basket = basket[: self.max_basket_items] if len(basket) > self.max_basket_items else basket
            per_basket_len[basket_pos] = len(truncated_basket)
            if truncated_basket:
                matrix[basket_pos, : len(truncated_basket)] = truncated_basket
        
        # Last position (num_hist + 1) is padding (already zeros), per_basket_len[num_hist+1] = 0
        
        # Return total length including padding baskets
        total_length = num_hist + 2  # padding + baskets + padding

        return (
            matrix.reshape(-1).tolist(),
            per_basket_len.tolist(),
            total_length,  # Return total length including padding (matching original input_length)
        )

    def _encode_target(self, basket: List[int]) -> Tuple[List[int], int]:
        # Encode as flat list: [item1, item2, ..., itemN, -1, -1, ...]
        # Functionally equivalent to original [[-1], basket, [-1]] structure
        # Original accesses target_variable[1] to skip front padding
        # We achieve same by filtering >= 0 (which skips all -1 padding)
        vec = np.full(self.max_basket_items, -1, dtype=np.int64)  # Use -1 for padding
        if basket:
            # Truncate basket if it exceeds max_basket_items
            truncated_basket = basket[: self.max_basket_items] if len(basket) > self.max_basket_items else basket
            vec[: len(truncated_basket)] = truncated_basket
            return vec.tolist(), len(truncated_basket)
        return vec.tolist(), 0

    def _register_field_properties(self):
        """Register feature metadata for subsequent RecBole processing."""
        self.set_field_property(
            self.uid_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )
        self.set_field_property(
            self.iid_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )
        self.set_field_property(
            self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1
        )
        self.set_field_property(
            self.phase_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )
        self.set_field_property(
            self.history_items_field,
            FeatureType.TOKEN_SEQ,
            FeatureSource.INTERACTION,
            self.max_history_baskets * self.max_basket_items,
        )
        self.set_field_property(
            self.history_length_field,
            FeatureType.TOKEN,
            FeatureSource.INTERACTION,
            1,
        )
        self.set_field_property(
            self.history_item_len_field,
            FeatureType.TOKEN_SEQ,
            FeatureSource.INTERACTION,
            self.max_history_baskets,
        )
        self.set_field_property(
            self.target_items_field,
            FeatureType.TOKEN_SEQ,
            FeatureSource.INTERACTION,
            self.max_basket_items,
        )
        self.set_field_property(
            self.target_length_field,
            FeatureType.TOKEN,
            FeatureSource.INTERACTION,
            1,
        )

    def _remap_ID_all(self):
        seq_fields = [self.history_items_field, self.target_items_field]
        seq_meta = {}
        seq_storage = {}
        for field in seq_fields:
            seq_meta[field] = (
                self.field2type.pop(field, None),
                self.field2source.pop(field, None),
                self.field2seqlen.pop(field, None),
            )
            if field in self.inter_feat:
                seq_storage[field] = self.inter_feat[field].copy()
                self.inter_feat.drop(columns=[field], inplace=True)

        # Remove seq_fields from _rest_fields to avoid KeyError in parent's _remap_ID_all
        self._rest_fields = np.setdiff1d(
            self._rest_fields, seq_fields, assume_unique=True
        )

        super()._remap_ID_all()

        token_map = self.field2token_id[self.iid_field]
        for field in seq_fields:
            ftype, fsource, fseqlen = seq_meta[field]
            if ftype is None:
                continue
            self.field2type[field] = ftype
            self.field2source[field] = fsource
            self.field2seqlen[field] = fseqlen
            if field not in seq_storage:
                continue
            
            # Debug: Check token_map and sample values before remapping
            if field == self.target_items_field and len(seq_storage[field]) > 0:
                sample_seq = seq_storage[field].iloc[0]
                sample_items = [item for item in sample_seq if item >= 0][:5]  # First 5 non-padding items (>= 0)
                if sample_items:
                    # Test remapping with string conversion
                    sample_mapped = [token_map.get(str(item), 0) for item in sample_items]
                    # Always log debug info for target_item_list remapping
                    self.logger.info(
                        f"[NBR Debug] Remapping {field}. "
                        f"Sample original items: {sample_items}, "
                        f"Sample mapped items (with str conversion): {sample_mapped}, "
                        f"Token map type: {type(list(token_map.keys())[0]) if token_map and len(token_map) > 0 else 'empty'}, "
                        f"Sample item type: {type(sample_items[0]) if sample_items else 'N/A'}"
                    )
                    if all(m == 0 for m in sample_mapped):
                        # All items mapped to 0 - this indicates a problem
                        self.logger.warning(
                            f"[NBR Debug] WARNING: All items mapped to 0! This will cause metrics to be 0."
                        )
                    else:
                        self.logger.info(
                            f"[NBR Debug] Remapping successful! Items are being mapped correctly."
                        )
                else:
                    self.logger.warning(
                        f"[NBR Debug] WARNING: No non-zero items found in sample_seq: {sample_seq[:10]}"
                    )
            
            # Convert items to strings for token_map lookup (token_map keys are strings)
            # Preserve -1 as padding (don't remap it)
            mapped = seq_storage[field].apply(
                lambda seq: tuple(item if item == -1 else token_map.get(str(item), 0) for item in seq)
            )
            self.inter_feat[field] = mapped

    def _change_feat_format(self):
        """Convert DataFrame columns to Interaction and reshape history tensors."""
        super()._change_feat_format()

        history_tensor = self.inter_feat[self.history_items_field]
        new_shape = (-1, self.max_history_baskets, self.max_basket_items)
        self.inter_feat[self.history_items_field] = history_tensor.view(new_shape)
        # Note: We do NOT update field2seqlen here. It must remain as the flattened
        # length (max_history_baskets * max_basket_items) for compatibility with
        # _dataframe_to_interaction() which uses it for slicing when creating
        # forget datasets during unlearning.

    def build(self):
        """Return (train, valid, test) datasets using fixed temporal phases."""
        self._change_feat_format()
        phase_token_map = self.field2token_id[self.phase_field]
        inter_phase = self.inter_feat[self.phase_field].view(-1)

        def select_by_phase(phase_name: str) -> Interaction:
            phase_id = phase_token_map[phase_name]
            indices = torch.nonzero(inter_phase == phase_id, as_tuple=False).view(-1)
            return self.inter_feat[indices]

        train_data = select_by_phase(self.TRAIN_PHASE)
        valid_data = select_by_phase(self.VALID_PHASE)
        test_data = select_by_phase(self.TEST_PHASE)

        return [
            self.copy(train_data),
            self.copy(valid_data),
            self.copy(test_data),
        ]


class Sets2SetsDataset(NextBasketDataset):
    """Alias dataset so that `config['model']='Sets2Sets'` picks the right loader."""

    pass

