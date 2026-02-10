"""
Create fraudulent/bot baskets for Next Basket Recommendation (NBR) unlearning experiments.

This script generates synthetic attack baskets that can be injected into clean NBR datasets
for testing unlearning algorithms. It supports bandwagon attacks by mixing popular items
with target items in fake user baskets.

Attack Strategy:
- Bandwagon Attack: Mix popular items with target items (most effective for NBR)

Usage:
    python create_fraud_baskets_nbr.py --dataset tafeng --attack bandwagon --poisoning_ratio 0.01 --seed 42
"""

import json
import numpy as np
import argparse
from datetime import datetime
import os
import random
import math
from collections import Counter
from typing import Dict, List, Set


class FraudBasketGenerator:
    def __init__(self, merged_data, attack_type="bandwagon", target_item_strategy="unpopular",
                 n_target_items=10, seed=42):
        """
        Initialize the fraud basket generator for next-basket recommendation.

        Args:
            merged_data: Dict mapping user_id -> list of baskets (each basket is a list of item_ids)
            attack_type: Type of attack ("bandwagon" for now, can be extended)
            target_item_strategy: How to select target items ("unpopular", "random", "popular")
            n_target_items: Number of items to promote
            seed: Random seed for reproducibility
        """
        self.merged_data = merged_data
        self.attack_type = attack_type
        self.target_item_strategy = target_item_strategy
        self.n_target_items = n_target_items
        self.seed = seed

        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Analyze normal basket patterns
        self.normal_stats = self._analyze_normal_baskets()

        # Select target items based on strategy
        self.target_items = self._select_target_items()

    def _analyze_normal_baskets(self):
        """Analyze normal basket patterns to mimic them."""
        # Collect all items and their frequencies
        all_items = []
        basket_sizes = []
        
        for user_id, baskets in self.merged_data.items():
            for basket in baskets:
                all_items.extend(basket)
                basket_sizes.append(len(basket))

        # Item popularity
        item_counter = Counter(all_items)
        item_popularity = dict(item_counter.most_common())

        # Get popular items (top 20% most frequent)
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        n_popular = max(1, int(len(sorted_items) * 0.2))
        popular_items = [item_id for item_id, _ in sorted_items[:n_popular]]

        # Get average popularity items (middle 40%)
        n_skip = max(0, int(len(sorted_items) * 0.3))
        n_avg = max(1, int(len(sorted_items) * 0.4))
        average_items = [item_id for item_id, _ in sorted_items[n_skip:n_skip+n_avg]]

        stats = {
            'avg_basket_size': np.mean(basket_sizes) if basket_sizes else 10,
            'std_basket_size': np.std(basket_sizes) if basket_sizes else 5,
            'min_basket_size': min(basket_sizes) if basket_sizes else 1,
            'max_basket_size': max(basket_sizes) if basket_sizes else 50,
            'popular_items': popular_items,
            'average_items': average_items,
            'item_popularity': item_popularity,
            'all_items': list(item_popularity.keys())
        }

        print(f"Normal basket statistics:")
        print(f"  Average basket size: {stats['avg_basket_size']:.2f}")
        print(f"  Basket size std: {stats['std_basket_size']:.2f}")
        print(f"  Min basket size: {stats['min_basket_size']}")
        print(f"  Max basket size: {stats['max_basket_size']}")
        print(f"  Popular items count: {len(popular_items)}")
        print(f"  Average items count: {len(average_items)}")
        print(f"  Total unique items: {len(item_popularity)}")

        return stats

    def _select_target_items(self):
        """Select items to promote based on strategy."""
        item_popularity = self.normal_stats['item_popularity']
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)

        if self.target_item_strategy == "unpopular":
            # Bottom 20% least popular items
            bottom_20_percent_count = max(1, int(len(sorted_items) * 0.2))
            bottom_items = [item_id for item_id, _ in sorted_items[-bottom_20_percent_count:]]
            target_items = np.random.choice(
                bottom_items,
                size=min(self.n_target_items, len(bottom_items)),
                replace=False
            ).tolist()
        elif self.target_item_strategy == "popular":
            # Top 5% popular items (nuke attack - damage popular items)
            top_5_percent_count = max(1, int(len(sorted_items) * 0.05))
            top_items = [item_id for item_id, _ in sorted_items[:top_5_percent_count]]
            target_items = np.random.choice(
                top_items,
                size=min(self.n_target_items, len(top_items)),
                replace=False
            ).tolist()
        else:  # random
            all_items = [item_id for item_id, _ in sorted_items]
            target_items = np.random.choice(
                all_items,
                size=min(self.n_target_items, len(all_items)),
                replace=False
            ).tolist()

        print(f"\nTarget items selected ({self.target_item_strategy} strategy):")
        print(f"  Target items: {target_items[:5]}..." if len(target_items) > 5 else f"  Target items: {target_items}")
        if len(target_items) > 0:
            target_ranks = [sorted(item_popularity.items(), key=lambda x: x[1], reverse=True).index((item, item_popularity[item])) 
                           for item in target_items[:3] if item in item_popularity]
            if target_ranks:
                print(f"  Avg popularity rank: {np.mean(target_ranks):.0f}")

        return target_items

    def _get_filler_items(self, n_items):
        """Get filler items based on attack type."""
        if self.attack_type == "bandwagon":
            # Use popular items (most effective)
            if len(self.normal_stats['popular_items']) == 0:
                # Fallback to all items if no popular items
                return np.random.choice(self.normal_stats['all_items'], size=n_items, replace=True)
            return np.random.choice(self.normal_stats['popular_items'], size=n_items, replace=True)
        elif self.attack_type == "average":
            # Use average popularity items (harder to detect)
            if len(self.normal_stats['average_items']) == 0:
                return np.random.choice(self.normal_stats['all_items'], size=n_items, replace=True)
            return np.random.choice(self.normal_stats['average_items'], size=n_items, replace=True)
        else:  # random
            # Use random items
            return np.random.choice(self.normal_stats['all_items'], size=n_items, replace=True)

    def generate_fraud_baskets(self, users_to_add):
        """
        Generate fraudulent user baskets based on attack type.

        Args:
            users_to_add: Number of fraud users to generate

        Returns:
            Dict mapping user_id -> list of baskets (fraud data)
        """
        print(f"\nGenerating {users_to_add} {self.attack_type} attack users...")

        # Get max user_id from original data to create new user IDs
        max_user_id = max(int(uid) for uid in self.merged_data.keys() if uid.isdigit()) if any(uid.isdigit() for uid in self.merged_data.keys()) else 0
        if max_user_id == 0:
            # Try to find max user_id as string
            try:
                max_user_id = max(int(uid) for uid in self.merged_data.keys())
            except:
                max_user_id = len(self.merged_data)

        fraud_data = {}

        # Average baskets per user in original data
        avg_baskets_per_user = np.mean([len(baskets) for baskets in self.merged_data.values()])
        avg_baskets_per_user = max(4, int(avg_baskets_per_user))  # At least 4 baskets for NBR

        for i in range(users_to_add):
            user_id = str(max_user_id + i + 1)

            # Generate number of baskets for this fraud user (similar to normal users)
            # Fraud users might have slightly fewer baskets (more efficient bots)
            n_baskets = max(4, int(np.random.normal(avg_baskets_per_user * 0.8, avg_baskets_per_user * 0.2)))
            n_baskets = np.clip(n_baskets, 4, int(avg_baskets_per_user * 1.5))

            user_baskets = []

            for basket_idx in range(n_baskets):
                # Sample basket size (slightly smaller for fraud baskets)
                mean_size = self.normal_stats['avg_basket_size'] * 0.9
                std_size = self.normal_stats['std_basket_size']
                basket_size = max(2, int(np.random.normal(mean_size, std_size)))
                basket_size = np.clip(basket_size, 2, int(self.normal_stats['max_basket_size'] * 0.8))

                # Determine how many target items to include (1-2 per basket)
                n_targets_in_basket = min(2 if basket_size >= 5 else 1, len(self.target_items))
                n_filler = basket_size - n_targets_in_basket

                # Get filler items
                filler_items = self._get_filler_items(n_filler).tolist()

                # Select target items for this basket
                target_items_in_basket = np.random.choice(
                    self.target_items,
                    size=n_targets_in_basket,
                    replace=False
                ).tolist()

                # Combine filler and target items, shuffle to make it look natural
                basket_items = filler_items + target_items_in_basket
                random.shuffle(basket_items)

                user_baskets.append(basket_items)

            fraud_data[user_id] = user_baskets

        # Calculate statistics
        total_baskets = sum(len(baskets) for baskets in fraud_data.values())
        total_items = sum(sum(len(basket) for basket in baskets) for baskets in fraud_data.values())
        target_appearances = sum(
            sum(1 for basket in baskets for item in basket if item in self.target_items)
            for baskets in fraud_data.values()
        )

        print(f"Generated fraud baskets statistics:")
        print(f"  Total fraud users: {len(fraud_data)}")
        print(f"  Total baskets: {total_baskets}")
        print(f"  Total items: {total_items}")
        print(f"  Avg baskets per user: {total_baskets / len(fraud_data):.2f}")
        print(f"  Target item appearances: {target_appearances}")

        return fraud_data


def main(dataset, attack_type="bandwagon", target_strategy="unpopular",
         poisoning_ratio=0.01, n_target_items=10, seed=42):
    """
    Main function to generate fraud baskets for next-basket datasets.

    Args:
        dataset: Dataset name (e.g., "tafeng", "instacart", "dunnhumby")
        attack_type: Attack strategy ("bandwagon" for now)
        target_strategy: Target item selection ("unpopular", "popular", "random")
        poisoning_ratio: Fraction of fraudulent users in final dataset
        n_target_items: Number of items to promote
        seed: Random seed
    """
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset - handle both relative and absolute paths
    dataset_paths = [
        f"./{dataset}/{dataset}_merged.json",
        f"./dataset/{dataset}/{dataset}_merged.json",
        f"{dataset}_merged.json",
        os.path.join(dataset, f"{dataset}_merged.json")
    ]

    merged_file = None
    output_dir = None

    for path in dataset_paths:
        if os.path.exists(path):
            merged_file = path
            output_dir = os.path.dirname(path) or "."
            break

    if merged_file is None:
        print(f"Error: Cannot find merged JSON file for '{dataset}'")
        print(f"Tried: {', '.join(dataset_paths)}")
        return

    if not os.path.exists(merged_file):
        print(f"Error: File {merged_file} not found!")
        return

    print(f"Loading dataset from {merged_file}...")
    print(f"Using random seed: {seed}")
    print(f"Attack type: {attack_type}")
    print(f"Target strategy: {target_strategy}")
    print(f"Poisoning ratio: {poisoning_ratio}")

    # Load merged JSON
    with open(merged_file, 'r') as f:
        merged_data = json.load(f)

    # Convert keys to strings if needed (JSON keys are always strings, but be safe)
    merged_data = {str(k): v for k, v in merged_data.items()}

    print(f"Dataset loaded: {len(merged_data)} users")
    total_baskets = sum(len(baskets) for baskets in merged_data.values())
    print(f"Total baskets: {total_baskets}")

    # Calculate users to add
    # users_to_add / (original_users + users_to_add) = poisoning_ratio
    # users_to_add = poisoning_ratio * original_users / (1 - poisoning_ratio)
    original_users = len(merged_data)
    users_to_add = math.ceil(poisoning_ratio * original_users / (1 - poisoning_ratio))

    print(f"\nWill add {users_to_add} fraud users to achieve {poisoning_ratio:.4f} poisoning ratio")

    # Generate fraud baskets
    generator = FraudBasketGenerator(
        merged_data,
        attack_type=attack_type,
        target_item_strategy=target_strategy,
        n_target_items=n_target_items,
        seed=seed
    )

    fraud_data = generator.generate_fraud_baskets(users_to_add)

    # Calculate stats for verification
    total_users = original_users + len(fraud_data)
    fraud_ratio = len(fraud_data) / total_users

    print(f"\nFinal dataset statistics (if combined):")
    print(f"  Total users: {total_users}")
    print(f"  Fraud ratio: {fraud_ratio:.4f}")

    # Save fraud baskets only (will be merged during training or manually)
    dataset_name = os.path.basename(dataset) if '/' not in dataset else dataset
    fraud_output_path = os.path.join(
        output_dir,
        f"{dataset_name}_fraud_baskets_{attack_type}_{target_strategy}_ratio_{poisoning_ratio}_seed_{seed}.json"
    )

    with open(fraud_output_path, 'w') as f:
        json.dump(fraud_data, f, indent=2)

    print(f"\nFraud baskets saved to: {fraud_output_path}")
    
    # Also create a merged version (original + fraud) for convenience
    merged_output_path = os.path.join(
        output_dir,
        f"{dataset_name}_merged_with_fraud_{attack_type}_{target_strategy}_ratio_{poisoning_ratio}_seed_{seed}.json"
    )
    
    # Merge original and fraud data
    combined_data = {**merged_data, **fraud_data}
    
    with open(merged_output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Merged dataset (original + fraud) saved to: {merged_output_path}")
    print(f"Note: Use the merged file for training with --spam flag, or merge manually")

    # Save metadata
    metadata = {
        'dataset': dataset,
        'attack_type': attack_type,
        'target_item_strategy': target_strategy,
        'poisoning_ratio': poisoning_ratio,
        'original_users': int(original_users),
        'fraud_users': int(len(fraud_data)),
        'target_items': generator.target_items,
        'n_target_items': n_target_items,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }

    metadata_path = os.path.join(
        output_dir,
        f"{dataset_name}_fraud_metadata_{attack_type}_{target_strategy}_ratio_{poisoning_ratio}_seed_{seed}.json"
    )
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fraudulent baskets for NBR unlearning")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["tafeng", "instacart", "dunnhumby"],
        help="Dataset name (tafeng, instacart, dunnhumby)"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="bandwagon",
        choices=["bandwagon"],
        help="Attack type (bandwagon=most effective)"
    )
    parser.add_argument(
        "--target_strategy",
        type=str,
        default="unpopular",
        choices=["unpopular", "popular", "random"],
        help="How to select target items"
    )
    parser.add_argument(
        "--poisoning_ratio",
        type=float,
        default=0.01,
        help="Fraction of fraudulent users (0.01 = 1%)"
    )
    parser.add_argument(
        "--n_target_items",
        type=int,
        default=10,
        help="Number of items to promote"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        attack_type=args.attack,
        target_strategy=args.target_strategy,
        poisoning_ratio=args.poisoning_ratio,
        n_target_items=args.n_target_items,
        seed=args.seed
    )
