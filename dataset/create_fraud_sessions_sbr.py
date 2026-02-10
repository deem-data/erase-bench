"""
Create fraudulent/bot sessions for Session-Based Recommendation (SBR) unlearning experiments.

This script generates synthetic attack sessions that can be injected into clean SBR datasets
for testing unlearning algorithms. It supports multiple attack strategies commonly used in
recommender system adversarial research.

Attack Strategies:
1. Bandwagon Attack: Mix popular items with target items (most effective)
2. Random Attack: Random filler items with target items (easiest to implement)
3. Average Attack: Items rated near average popularity with target items (harder to detect)
4. Push Attack: Boost unpopular items by creating fake popularity

Usage:
    python create_fraud_sessions_sbr.py --dataset rsc15 --attack bandwagon --poisoning_ratio 0.01 --seed 42
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import os
import random
import math
import json


class FraudSessionGenerator:
    def __init__(self, df, attack_type="bandwagon", target_item_strategy="unpopular",
                 n_target_items=10, seed=42):
        """
        Initialize the fraud session generator for session-based recommendation.

        Args:
            df: Original dataframe with columns [session_id, item_id, timestamp]
            attack_type: Type of attack ("bandwagon", "random", "average", "push")
            target_item_strategy: How to select target items ("unpopular", "random", "popular")
            n_target_items: Number of items to promote
            seed: Random seed for reproducibility
        """
        self.df = df
        self.attack_type = attack_type
        self.target_item_strategy = target_item_strategy
        self.n_target_items = n_target_items
        self.seed = seed

        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Analyze normal session patterns
        self.normal_stats = self._analyze_normal_sessions()

        # Select target items based on strategy
        self.target_items = self._select_target_items()

    def _analyze_normal_sessions(self):
        """Analyze normal session patterns to mimic them."""
        # Session length distribution
        session_lengths = self.df.groupby('session_id').size()

        # Time between clicks
        df_sorted = self.df.sort_values(['session_id', 'timestamp'])
        df_sorted['time_diff'] = df_sorted.groupby('session_id')['timestamp'].diff()

        # Item popularity
        item_popularity = self.df['item_id'].value_counts()

        # Get popular items (top 20% most clicked)
        n_popular = int(len(item_popularity) * 0.2)
        popular_items = item_popularity.head(n_popular).index.tolist()

        # Get average popularity items (middle 40%)
        n_skip = int(len(item_popularity) * 0.3)
        n_avg = int(len(item_popularity) * 0.4)
        average_items = item_popularity.iloc[n_skip:n_skip+n_avg].index.tolist()

        stats = {
            'avg_session_length': session_lengths.mean(),
            'std_session_length': session_lengths.std(),
            'min_session_length': session_lengths.min(),
            'max_session_length': session_lengths.max(),
            'avg_time_between_clicks': df_sorted['time_diff'].mean(),
            'std_time_between_clicks': df_sorted['time_diff'].std(),
            'popular_items': popular_items,
            'average_items': average_items,
            'item_popularity': item_popularity
        }

        print(f"Normal session statistics:")
        print(f"  Average session length: {stats['avg_session_length']:.2f}")
        print(f"  Session length std: {stats['std_session_length']:.2f}")
        print(f"  Popular items count: {len(popular_items)}")
        print(f"  Average items count: {len(average_items)}")
        print(f"  Avg time between clicks: {stats['avg_time_between_clicks']:.2f}")

        return stats

    def _select_target_items(self):
        """Select items to promote based on strategy."""
        item_popularity = self.normal_stats['item_popularity']

        if self.target_item_strategy == "unpopular":
            # Bottom 20% least popular items (good for push attacks)
            bottom_20_percent_count = int(len(item_popularity) * 0.2)
            bottom_20_items = item_popularity.tail(bottom_20_percent_count).index.tolist()
            target_items = np.random.choice(
                bottom_20_items,
                size=min(self.n_target_items, len(bottom_20_items)),
                replace=False
            ).tolist()
        elif self.target_item_strategy == "popular":
            # Top 5% popular items (nuke attack - damage popular items)
            top_5_percent_count = int(len(item_popularity) * 0.05)
            top_5_items = item_popularity.head(top_5_percent_count).index.tolist()
            target_items = np.random.choice(
                top_5_items,
                size=min(self.n_target_items, len(top_5_items)),
                replace=False
            ).tolist()
        else:  # random
            all_items = item_popularity.index.tolist()
            target_items = np.random.choice(
                all_items,
                size=min(self.n_target_items, len(all_items)),
                replace=False
            ).tolist()

        print(f"\nTarget items selected ({self.target_item_strategy} strategy):")
        print(f"  Target items: {target_items[:5]}..." if len(target_items) > 5 else f"  Target items: {target_items}")
        print(f"  Avg popularity rank: {[item_popularity.index.get_loc(item) for item in target_items[:3]]}")

        return target_items

    def _get_filler_items(self, n_items):
        """Get filler items based on attack type."""
        if self.attack_type == "bandwagon":
            # Use popular items (most effective)
            return np.random.choice(self.normal_stats['popular_items'], size=n_items, replace=True)
        elif self.attack_type == "average":
            # Use average popularity items (harder to detect)
            return np.random.choice(self.normal_stats['average_items'], size=n_items, replace=True)
        else:  # random or push
            # Use random items
            all_items = self.normal_stats['item_popularity'].index.tolist()
            return np.random.choice(all_items, size=n_items, replace=True)

    def generate_fraud_sessions(self, sessions_to_add):
        """
        Generate fraudulent sessions based on attack type.

        Args:
            sessions_to_add: Number of fraud sessions to generate

        Returns:
            DataFrame with fraud sessions
        """
        print(f"\nGenerating {sessions_to_add} {self.attack_type} attack sessions...")

        max_session_id = self.df['session_id'].max()
        min_timestamp = self.df['timestamp'].min()
        max_timestamp = self.df['timestamp'].max()

        fraud_sessions = []

        for i in range(sessions_to_add):
            session_id = max_session_id + i + 1

            # Sample session length (slightly shorter for bots)
            mean = self.normal_stats['avg_session_length'] * 0.8  # Bots are more efficient
            std = self.normal_stats['std_session_length']

            sigma_squared = np.log(1 + (std**2 / mean**2))
            mu = np.log(mean) - sigma_squared / 2
            lambda_param = np.random.lognormal(mu, np.sqrt(sigma_squared))

            session_length = max(4, np.random.poisson(lambda_param))
            session_length = np.clip(session_length, 4, self.normal_stats['max_session_length'])

            # Start timestamp randomly
            current_timestamp = np.random.randint(min_timestamp, max_timestamp)

            # Determine where to place target item(s)
            # Place 1-2 target items (sessions always >= 4, so safe)
            n_targets_in_session = min(2 if session_length >= 6 else 1, len(self.target_items))
            target_positions = []
            for _ in range(n_targets_in_session):
                pos = np.random.randint(
                    max(1, int(session_length * 0.2)),
                    max(2, int(session_length * 0.9))
                )
                target_positions.append(pos)

            # Get filler items (session_length >= 4, target >= 1, so n_filler >= 3)
            n_filler = session_length - len(target_positions)
            filler_items = self._get_filler_items(n_filler)
            filler_idx = 0

            # Build session
            for j in range(session_length):
                if j in target_positions:
                    item_id = np.random.choice(self.target_items)
                else:
                    if filler_idx < len(filler_items):
                        item_id = filler_items[filler_idx]
                        filler_idx += 1
                    else:
                        # Safety fallback (shouldn't happen with fixed session length >= 4)
                        item_id = np.random.choice(self.normal_stats['popular_items'])

                fraud_sessions.append({
                    'session_id': session_id,
                    'item_id': item_id,
                    'timestamp': current_timestamp
                })

                # Add realistic time gap (bots are slightly faster)
                if self.normal_stats['avg_time_between_clicks'] > 0:
                    mean_time = self.normal_stats['avg_time_between_clicks'] * 0.7
                    std_time = self.normal_stats['std_time_between_clicks']

                    cv = std_time / mean_time if mean_time > 0 else 1
                    k = 1.2 / cv
                    lambda_scale = mean_time / math.gamma(1 + 1 / k)

                    time_gap = max(1, round(np.random.weibull(k) * lambda_scale))
                    current_timestamp += abs(time_gap)
                else:
                    current_timestamp += 1

        fraud_df = pd.DataFrame(fraud_sessions)

        print(f"Generated fraud sessions statistics:")
        print(f"  Total clicks: {len(fraud_df)}")
        print(f"  Unique sessions: {fraud_df['session_id'].nunique()}")
        print(f"  Avg clicks per session: {len(fraud_df) / fraud_df['session_id'].nunique():.2f}")
        print(f"  Target item appearances: {fraud_df[fraud_df['item_id'].isin(self.target_items)].shape[0]}")

        return fraud_df

    def inject_fraud_sessions(self, fraud_df):
        """
        Combine original data with fraud sessions.

        Returns:
            Combined DataFrame
        """
        combined_df = pd.concat([self.df, fraud_df], ignore_index=True)
        combined_df = combined_df.sort_values(['session_id', 'timestamp'])

        print(f"\nCombined dataset statistics:")
        print(f"  Total sessions: {combined_df['session_id'].nunique()}")
        print(f"  Total clicks: {len(combined_df)}")
        print(f"  Fraud ratio: {fraud_df['session_id'].nunique() / combined_df['session_id'].nunique():.4f}")

        return combined_df


def main(dataset, attack_type="bandwagon", target_strategy="unpopular",
         poisoning_ratio=0.01, n_target_items=10, seed=42):
    """
    Main function to generate fraud sessions for session-based datasets.

    Args:
        dataset: Dataset name (e.g., "rsc15", "30music", "diginetica")
        attack_type: Attack strategy ("bandwagon", "random", "average", "push")
        target_strategy: Target item selection ("unpopular", "popular", "random")
        poisoning_ratio: Fraction of fraudulent sessions in final dataset
        n_target_items: Number of items to promote
        seed: Random seed
    """
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset - handle both relative and absolute paths
    if os.path.exists(f"./{dataset}/{dataset}.inter"):
        filepath = f"./{dataset}/{dataset}.inter"
        output_dir = f"./{dataset}"
    elif os.path.exists(f"./dataset/{dataset}/{dataset}.inter"):
        filepath = f"./dataset/{dataset}/{dataset}.inter"
        output_dir = f"./dataset/{dataset}"
    elif os.path.exists(f"{dataset}.inter"):
        filepath = f"{dataset}.inter"
        output_dir = os.path.dirname(filepath) or "."
    else:
        print(f"Error: Cannot find dataset file for '{dataset}'")
        print(f"Tried: ./{dataset}/{dataset}.inter and ./dataset/{dataset}/{dataset}.inter")
        return

    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found!")
        return

    print(f"Loading dataset from {filepath}...")
    print(f"Using random seed: {seed}")
    print(f"Attack type: {attack_type}")
    print(f"Target strategy: {target_strategy}")
    print(f"Poisoning ratio: {poisoning_ratio}")

    # Read with proper column names
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=0
    )

    # Rename columns to standard format (remove :token/:float suffixes)
    original_columns = df.columns.tolist()
    df.columns = [col.split(':')[0] for col in df.columns]
    
    # Check if dataset has user_id column (e.g., nowp)
    has_user_id = 'user_id' in df.columns
    if has_user_id:
        print(f"Dataset has user_id column - will preserve format in output")

    print(f"Dataset loaded: {len(df)} interactions, {df['session_id'].nunique()} sessions")

    # Calculate sessions to add
    # sessions_to_add / (original_sessions + sessions_to_add) = poisoning_ratio
    # sessions_to_add = poisoning_ratio * original_sessions / (1 - poisoning_ratio)
    original_sessions = df['session_id'].nunique()
    sessions_to_add = math.ceil(poisoning_ratio * original_sessions / (1 - poisoning_ratio))

    print(f"\nWill add {sessions_to_add} fraud sessions to achieve {poisoning_ratio:.4f} poisoning ratio")

    # Generate fraud sessions
    generator = FraudSessionGenerator(
        df,
        attack_type=attack_type,
        target_item_strategy=target_strategy,
        n_target_items=n_target_items,
        seed=seed
    )

    fraud_df = generator.generate_fraud_sessions(sessions_to_add)

    # Calculate stats for verification (no need to save combined dataset)
    total_sessions = df['session_id'].nunique() + fraud_df['session_id'].nunique()
    total_clicks = len(df) + len(fraud_df)
    fraud_ratio = fraud_df['session_id'].nunique() / total_sessions

    print(f"\nFinal dataset statistics (if combined):")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Total clicks: {total_clicks}")
    print(f"  Fraud ratio: {fraud_ratio:.4f}")

    # Save fraud sessions only (will be injected during training)
    dataset_name = os.path.basename(dataset) if '/' not in dataset else dataset
    fraud_output_path = f"{output_dir}/{dataset_name}_fraud_sessions_{attack_type}_{target_strategy}_ratio_{poisoning_ratio}_seed_{seed}.inter"
    fraud_df_renamed = fraud_df.copy()
    
    # Add user_id column if original dataset has it (e.g., nowp)
    if has_user_id:
        # For fraud sessions, use session_id as user_id (new sessions don't have real users)
        fraud_df_renamed['user_id'] = fraud_df_renamed['session_id']
        # Rename columns to match original format
        fraud_df_renamed = fraud_df_renamed.rename(columns={
            "user_id": "user_id:token",
            "session_id": "session_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float"
        })
        # Reorder columns to match nowp format: user_id, session_id, item_id, timestamp
        fraud_df_renamed = fraud_df_renamed[['user_id:token', 'session_id:token', 'item_id:token', 'timestamp:float']]
    else:
        # For datasets without user_id (e.g., rsc15), use session_id as user_id in output
        fraud_df_renamed = fraud_df_renamed.rename(columns={
            "session_id": "session_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float"
        })
    
    fraud_df_renamed.to_csv(fraud_output_path, sep="\t", index=False)
    print(f"\nFraud sessions saved to: {fraud_output_path}")
    print(f"Note: These will be automatically injected when training with --spam flag")

    # Save metadata
    metadata = {
        'dataset': dataset,
        'attack_type': attack_type,
        'target_item_strategy': target_strategy,
        'poisoning_ratio': poisoning_ratio,
        'original_sessions': int(original_sessions),
        'fraud_sessions': int(fraud_df['session_id'].nunique()),
        'target_items': generator.target_items,
        'n_target_items': n_target_items,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }

    metadata_path = f"{output_dir}/{dataset_name}_fraud_metadata_{attack_type}_{target_strategy}_ratio_{poisoning_ratio}_seed_{seed}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fraudulent sessions for SBR unlearning")
    parser.add_argument(
        "--dataset",
        type=str,
        default="rsc15",
        help="Dataset name (rsc15, 30music, diginetica, nowp)"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="bandwagon",
        choices=["bandwagon", "random", "average", "push"],
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
        help="Fraction of fraudulent sessions (0.01 = 1%%)"
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
