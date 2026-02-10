import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import os
import random
import math


class SpamSessionGenerator:
    def __init__(self, df, target_items, sessions_to_add=None, seed=42):
        """
        Initialize the spam session generator.
        
        Args:
            df: Original dataframe with columns [session_id, item_id, timestamp]
            target_items: List of item IDs to promote
            sessions_to_add: Amount of malicious sessions to add
            seed: Random seed for reproducibility
        """
        self.df = df
        self.target_items = target_items
        self.sessions_to_add = sessions_to_add
        self.seed = seed
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Analyze normal session patterns
        self.normal_stats = self._analyze_normal_sessions()
        
    def _analyze_normal_sessions(self):
        """Analyze normal session patterns to mimic them."""
        # Session length distribution
        session_lengths = self.df.groupby('session_id').size()
        
        # Time between clicks (assuming timestamp is unix timestamp)
        df_sorted = self.df.sort_values(['session_id', 'timestamp'])
        df_sorted['time_diff'] = df_sorted.groupby('session_id')['timestamp'].diff()
        
        # Item popularity
        item_popularity = self.df['item_id'].value_counts()
        
        # Get popular items (top 20% most clicked)
        n_popular = int(len(item_popularity) * 0.2)
        popular_items = item_popularity.head(n_popular).index.tolist()
        
        stats = {
            'avg_session_length': session_lengths.mean(),
            'std_session_length': session_lengths.std(),
            'min_session_length': session_lengths.min(),
            'max_session_length': session_lengths.max(),
            'avg_time_between_clicks': df_sorted['time_diff'].mean(),
            'std_time_between_clicks': df_sorted['time_diff'].std(),
            'popular_items': popular_items,
            'item_popularity': item_popularity
        }
        
        print(f"Normal session statistics:")
        print(f"  Average session length: {stats['avg_session_length']:.2f}")
        print(f"  Session length std: {stats['std_session_length']:.2f}")
        print(f"  Popular items count: {len(popular_items)}")
        print(f"  Avg time between clicks: {stats['avg_time_between_clicks']:.2f}")
        
        return stats
    
    def generate_spam_sessions(self, num_sessions=None):
        """
        Generate spam sessions that end with target items.
        
        Args:
            num_sessions: Number of spam sessions to generate. 
                         If None, uses sessions_to_add
        
        Returns:
            DataFrame with spam sessions
        """
        if num_sessions is None:
            original_sessions = self.df['session_id'].nunique()
            num_sessions = self.sessions_to_add
        
        print(f"\nGenerating {num_sessions} spam sessions...")
        
        max_session_id = self.df['session_id'].max()
        
        min_timestamp = self.df['timestamp'].min()
        max_timestamp = self.df['timestamp'].max()
        
        spam_sessions = []
        
        for i in range(num_sessions):
            session_id = max_session_id + i + 1
            
            # sample from Poisson log-normal distribution:
            mean = self.normal_stats['avg_session_length']
            std = self.normal_stats['std_session_length']

            # use mean and std of session length to infer log-normal parameters
            sigma_squared = np.log(1 + (std**2 / mean**2))
            mu = np.log(mean) - sigma_squared / 2

            # Sample lambda parameter from log-normal distribution
            lambda_param = np.random.lognormal(mu, np.sqrt(sigma_squared))

            # Sample session length from a poisson distribution with length >= 4 s.t. we have at least 2 train, 1 val, 1 test item
            session_length = max(4, np.random.poisson(lambda_param))
            
            # Clip to reasonable bounds
            session_length = np.clip(
                session_length, 
                4, 
                self.normal_stats['max_session_length']
            )
            
            # Start timestamp randomly within the dataset's time range
            current_timestamp = np.random.randint(min_timestamp, max_timestamp)
            
            # Generate clicks on popular items (excluding last click)
            target_position = np.random.randint(
                max(1, int(session_length * 0.2)), 
                max(2, int(session_length * 0.8))
            )
            
            for j in range(session_length):
                # Randomly select from popular items
                if j == target_position:
                    item_id = np.random.choice(self.target_items)
                else:
                    item_id = np.random.choice(self.normal_stats['popular_items'])
                
                spam_sessions.append({
                    'session_id': session_id,
                    'item_id': item_id,
                    'timestamp': current_timestamp
                })

                # Add realistic time gap
                if self.normal_stats['avg_time_between_clicks'] > 0:
                    # Sample from Weibull distribution built from mean and std of time between clicks
                    mean_time = self.normal_stats['avg_time_between_clicks']
                    std_time = self.normal_stats['std_time_between_clicks']
                    
                    # Estimate Weibull parameters from mean and std
                    # Using method of moments (approximate)
                    cv = std_time / mean_time
                    
                    # For Weibull: shape parameter k and scale parameter lambda
                    # These approximations work well for cv < 1
                    k = 1.2 / cv  # shape parameter (approximate)
                    # scale param
                    lambda_scale = mean_time / math.gamma(1 + 1 / k)
                    
                    time_gap = max(1, round(np.random.weibull(k) * lambda_scale))
                    current_timestamp += abs(time_gap)
                else:
                    current_timestamp += 1
            
        
        spam_df = pd.DataFrame(spam_sessions)
        
        # Print some statistics about generated sessions
        print(f"Generated spam sessions statistics:")
        print(f"  Total clicks: {len(spam_df)}")
        print(f"  Unique sessions: {spam_df['session_id'].nunique()}")
        print(f"  Avg clicks per session: {len(spam_df) / spam_df['session_id'].nunique():.2f}")
        
        return spam_df
    
    def inject_spam_sessions(self, spam_df):
        """
        Combine original data with spam sessions.
        
        Args:
            spam_df: DataFrame containing spam sessions
            
        Returns:
            Combined DataFrame
        """
        combined_df = pd.concat([self.df, spam_df], ignore_index=True)
        combined_df = combined_df.sort_values(['session_id', 'timestamp'])
        
        print(f"\nCombined dataset statistics:")
        print(f"  Total sessions: {combined_df['session_id'].nunique()}")
        print(f"  Total clicks: {len(combined_df)}")
        print(f"  Spam ratio: {spam_df['session_id'].nunique() / combined_df['session_id'].nunique()}")
        
        return combined_df


def main(dataset, seed=2, unlearning_fraction=0.001, n_target_items=10):
    # make reproducible
    np.random.seed(seed)
    random.seed(seed)
    
    if dataset == "rsc15":
        filepath = "./rsc15/rsc15.inter"
        output_dir = "./rsc15"
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found!")
            return
        
        print(f"Loading dataset from {filepath}...")
        print(f"Using random seed: {seed}")
        df = pd.read_csv(
            filepath,
            header=0,
            names=["session_id", "item_id", "timestamp"],
            dtype={
                "session_id": np.int64,
                "item_id": np.int64,
                "timestamp": np.int64,
            },
            sep="\t",
        )
        
        # For rsc15, we don't have user_id, so we'll use session_id as user_id in output
        has_user_id = False
        
    elif dataset == "nowp":
        filepath = "./nowp/nowp.inter"
        output_dir = "./nowp"
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found!")
            return
        
        print(f"Loading dataset from {filepath}...")
        print(f"Using random seed: {seed}")
        # Read nowp.inter which has user_id, session_id, item_id, timestamp
        df_full = pd.read_csv(
            filepath,
            header=0,
            sep="\t",
        )
        
        # Extract columns we need for spam generation (session_id, item_id, timestamp)
        # For nowp, item_id is string (musicbrainz_id), so we'll keep it as is
        df = df_full[['session_id:token', 'item_id:token', 'timestamp:float']].copy()
        df.columns = ['session_id', 'item_id', 'timestamp']
        
        # Convert session_id to int (it should be numeric in nowp)
        df['session_id'] = df['session_id'].astype(np.int64)
        
        # Convert timestamp to int64 (it's already numeric)
        df['timestamp'] = df['timestamp'].astype(np.int64)
        
        # item_id stays as string (object) - this is fine for the generator
        
        # Store original user_id mapping for later use
        if 'user_id:token' in df_full.columns:
            has_user_id = True
            # Create mapping from session_id to user_id (take first user_id for each session)
            user_id_mapping = df_full.groupby('session_id:token')['user_id:token'].first().to_dict()
        else:
            has_user_id = False
            user_id_mapping = None
    else:
        print(f"Error: Unsupported dataset '{dataset}'. Supported datasets: rsc15, nowp")
        return
    
    print(f"Dataset loaded: {len(df)} interactions, {df['session_id'].nunique()} sessions")
    
    # requirement such that we add the correct amount of interactions:
    # the amount of added interactions divided by the size of the resulting dataset has to be unlearning_fraction.
    # sessions_to_add / (|dataset| + sessions_to_add) = unlearning_fraction
    # iff sessions_to_add = unlearning_fraction * |dataset| / (1 - unlearning_fraction)
    sessions_to_add = math.ceil(unlearning_fraction * df['session_id'].nunique() / (1 - unlearning_fraction))
    print(f"unlearning_fraction: {unlearning_fraction}")
    print(f"We need to add {sessions_to_add} sessions to get an unlearning set of size unlearning_fraction * sessions_in_poisoned_dataset.")

    # Define target items to promote:
    # Randomly sample from the bottom 20% least popular items
    item_popularity = df['item_id'].value_counts()
    bottom_20_percent_count = int(len(item_popularity) * 0.2)
    bottom_20_items = item_popularity.tail(bottom_20_percent_count).index.tolist()
    target_items = np.random.choice(bottom_20_items, size=min(n_target_items, len(bottom_20_items)), replace=False).tolist()

    print(f"\nTarget items to promote: {target_items}")

    generator = SpamSessionGenerator(df, target_items, sessions_to_add, seed=seed)
    
    spam_df = generator.generate_spam_sessions()
    combined_df = generator.inject_spam_sessions(spam_df)
    
    # Prepare output format based on dataset
    if dataset == "nowp" and has_user_id:
        # For nowp, add user_id column to spam sessions (use session_id as user_id for spam)
        spam_df_output = spam_df.copy()
        spam_df_output['user_id'] = spam_df_output['session_id']
        
        # Rename columns to RecBole format
        spam_df_output = spam_df_output.rename(columns={
            "user_id": "user_id:token",
            "session_id": "session_id:token", 
            "item_id": "item_id:token", 
            "timestamp": "timestamp:float"
        })
        
        # Reorder columns to match nowp format
        spam_df_output = spam_df_output[['user_id:token', 'session_id:token', 'item_id:token', 'timestamp:float']]
        
        # For combined dataset, we need to merge with original user_id
        # Since we're adding new sessions, we'll use session_id as user_id for new sessions
        combined_df_output = combined_df.copy()
        if user_id_mapping is not None:
            # Map original sessions to their user_ids
            combined_df_output['user_id'] = combined_df_output['session_id'].map(user_id_mapping)
            # For new spam sessions (not in mapping), use session_id as user_id
            combined_df_output['user_id'] = combined_df_output['user_id'].fillna(combined_df_output['session_id'])
        else:
            combined_df_output['user_id'] = combined_df_output['session_id']
        
        combined_df_output = combined_df_output.rename(columns={
            "user_id": "user_id:token",
            "session_id": "session_id:token",
            "item_id": "item_id:token", 
            "timestamp": "timestamp:float"
        })
        combined_df_output = combined_df_output[['user_id:token', 'session_id:token', 'item_id:token', 'timestamp:float']]
    else:
        # For rsc15, use session_id as user_id in output
        spam_df_output = spam_df.rename(columns={
            "session_id": "user_id:token", 
            "item_id": "item_id:token", 
            "timestamp": "timestamp:float"
        })
        combined_df_output = combined_df.rename(columns={
            "session_id": "user_id:token", 
            "item_id": "item_id:token", 
            "timestamp": "timestamp:float"
        })
    
    # Save spam sessions separately
    spam_output_path = f"{output_dir}/{dataset}_spam_sessions_dataset_{dataset}_unlearning_fraction_{unlearning_fraction}_n_target_items_{n_target_items}_seed_{seed}.inter"
    spam_df_output.to_csv(spam_output_path, sep="\t", index=False)
    print(f"\nSpam sessions saved to: {spam_output_path}")
    
    
    # Save combined dataset
    combined_output_path = f"{output_dir}/{dataset}_with_spam_dataset_{dataset}_unlearning_fraction_{unlearning_fraction}_n_target_items_{n_target_items}_seed_{seed}.inter"
    combined_df_output.to_csv(combined_output_path, sep="\t", index=False)
    print(f"Combined dataset saved to: {combined_output_path}")
    
    # Save metadata about the attack using renamed cols
    metadata = {
        'original_sessions': df['session_id'].nunique(),
        'spam_sessions': spam_df['session_id'].nunique(),
        'target_items': target_items,
        'sessions_to_add': sessions_to_add,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    metadata_path = f"{output_dir}/spam_metadata_dataset_{dataset}_unlearning_fraction_{unlearning_fraction}_n_target_items_{n_target_items}_seed_{seed}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Attack metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name for which to create spam sessions",
        default="rsc15"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducibility",
        default=42
    )
    parser.add_argument(
        "--unlearning_fraction",
        type=float,
        help="amount of interactions we want to be unlearned/be malicious in the poisoned dataset",
        default=0.0001,
    )
    parser.add_argument(
        "--n_target_items",
        type=int,
        help="count of items the attacker wants to boost",
        default=10,
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        seed=args.seed,
        unlearning_fraction=args.unlearning_fraction,
        n_target_items=args.n_target_items,
    )