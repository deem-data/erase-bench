#!/usr/bin/env python3
"""
Create small test subsets of datasets for fast testing.

This script uses USER-CENTRIC sampling: it randomly samples a small percentage
of users and includes ALL their interactions (or baskets for basket-based datasets).
This ensures that user behavior patterns are preserved in the test subset.

IMPORTANT: Ensures at least 1% of sampled users have sensitive interactions.

Usage:
    python create_test_subset.py --dataset movielens --sample-ratio 0.01
    python create_test_subset.py --dataset amazon_reviews_books --sample-ratio 0.01
    python create_test_subset.py --dataset instacart --sample-ratio 0.01
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import glob
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Instacart sensitive product detection may not work.")


def load_sensitive_items_movielens(main_dir, category="health"):
    """Load sensitive movie IDs for MovieLens."""
    sensitive_file = os.path.join(main_dir, f"sensitive_asins_{category}.txt")
    
    # If file doesn't exist, try to generate it
    if not os.path.exists(sensitive_file):
        print(f"  Sensitive items file not found: {sensitive_file}")
        print(f"  Attempting to generate it...")
        
        identify_script = os.path.join(main_dir, "identify_sensitive_movies.py")
        if os.path.exists(identify_script):
            try:
                subprocess.run(
                    [sys.executable, identify_script, "--category", category, 
                     "--output", sensitive_file],
                    cwd=main_dir,
                    check=True,
                    capture_output=True
                )
                print(f"  Generated {sensitive_file}")
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Could not generate sensitive items: {e}")
                return set()
        else:
            print(f"  Warning: identify_sensitive_movies.py not found")
            return set()
    
    sensitive_items = set()
    if os.path.exists(sensitive_file):
        with open(sensitive_file, 'r', encoding='utf-8') as f:
            for line in f:
                item_id = line.strip()
                if item_id:
                    sensitive_items.add(item_id)
        print(f"  Loaded {len(sensitive_items)} sensitive items from {sensitive_file}")
    
    return sensitive_items


def load_sensitive_items_amazon_books(main_dir, category="health"):
    """Load sensitive book IDs for Amazon Reviews Books."""
    sensitive_file = os.path.join(main_dir, f"sensitive_asins_{category}.txt")
    
    # If file doesn't exist, try to generate it
    if not os.path.exists(sensitive_file):
        print(f"  Sensitive items file not found: {sensitive_file}")
        print(f"  Attempting to generate it...")
        
        identify_script = os.path.join(main_dir, "identify_sensitive_items.py")
        metadata_file = os.path.join(main_dir, "meta_Books.jsonl.gz")
        
        if os.path.exists(identify_script) and os.path.exists(metadata_file):
            try:
                subprocess.run(
                    [sys.executable, identify_script, "--category", category,
                     "--output", sensitive_file, "--files", metadata_file],
                    cwd=main_dir,
                    check=True,
                    capture_output=True
                )
                print(f"  Generated {sensitive_file}")
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Could not generate sensitive items: {e}")
                return set()
        else:
            print(f"  Warning: identify_sensitive_items.py or meta_Books.jsonl.gz not found")
            return set()
    
    sensitive_items = set()
    if os.path.exists(sensitive_file):
        with open(sensitive_file, 'r', encoding='utf-8') as f:
            for line in f:
                item_id = line.strip()
                if item_id:
                    sensitive_items.add(item_id)
        print(f"  Loaded {len(sensitive_items)} sensitive items from {sensitive_file}")
    
    return sensitive_items


def load_sensitive_items_instacart(main_dir, category="alcohol"):
    """Load sensitive product IDs for Instacart."""
    sensitive_file = os.path.join(main_dir, f"sensitive_products_{category}.txt")
    
    if not os.path.exists(sensitive_file):
        print(f"  Warning: Sensitive products file not found: {sensitive_file}")
        return set()
    
    sensitive_items = set()
    with open(sensitive_file, 'r', encoding='utf-8') as f:
        for line in f:
            product_id = line.strip()
            if product_id:
                # Store as integer for comparison
                try:
                    sensitive_items.add(int(product_id))
                except ValueError:
                    pass
    print(f"  Loaded {len(sensitive_items)} sensitive products from {sensitive_file}")
    
    return sensitive_items


def load_users_from_forget_sets(main_dir, category="alcohol"):
    """
    Load user IDs from existing forget set files in the main dataset directory.
    These users are guaranteed to have sensitive items.
    
    Returns a set of user IDs (as strings) from forget sets.
    """
    # Look for forget set JSON files
    import glob
    
    pattern = os.path.join(main_dir, f"instacart_unlearn_pairs_sensitive_category_{category}_seed_*_unlearning_fraction_*.json")
    forget_files = glob.glob(pattern)
    
    if not forget_files:
        print(f"  Warning: No forget set files found matching pattern: {pattern}")
        return set()
    
    # Load user IDs from the first forget set file (they should all have the same users for same category)
    # Actually, let's combine all forget sets for this category to get all users with sensitive items
    all_users = set()
    for forget_file in forget_files:
        try:
            with open(forget_file, 'r', encoding='utf-8') as f:
                user_ids = json.load(f)
                if isinstance(user_ids, list):
                    all_users.update(str(uid) for uid in user_ids)
                elif isinstance(user_ids, dict):
                    # If it's a dict, take keys or values depending on structure
                    all_users.update(str(uid) for uid in user_ids.keys())
        except Exception as e:
            print(f"  Warning: Could not load {forget_file}: {e}")
            continue
    
    print(f"  Loaded {len(all_users)} unique users from {len(forget_files)} forget set files")
    return all_users


def create_movielens_test_subset(input_file, output_file, sample_ratio, seed, main_dir):
    """
    Create test subset for MovieLens dataset using USER-CENTRIC sampling.
    Ensures at least 1% of sampled users have sensitive interactions.
    Only includes users with >= 4 interactions remaining after removing sensitive items.
    
    Format: user_id:token	item_id:token	rating:float	timestamp:float
    """
    print(f"Creating MovieLens test subset...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Sample ratio: {sample_ratio} ({sample_ratio*100}%)")
    print(f"  Seed: {seed}")
    
    random.seed(seed)
    
    # Step 0: Load sensitive items
    print("  Step 0: Loading sensitive items...")
    sensitive_items = load_sensitive_items_movielens(main_dir, category="health")
    
    # Step 1: Collect all users and their interactions, identify users with sensitive items
    # Also filter to only users with >= 4 interactions remaining after removing sensitive items
    user_interactions = defaultdict(list)
    users_with_sensitive = set()
    user_sensitive_counts = defaultdict(int)
    header = None
    
    print("  Step 1: Loading all interactions and identifying users with sensitive items...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        header = '\t'.join(reader.fieldnames)
        
        for row in reader:
            user_id = row['user_id:token']
            item_id = row['item_id:token']
            user_interactions[user_id].append(row)
            
            # Check if this user has sensitive items
            if item_id in sensitive_items:
                users_with_sensitive.add(user_id)
                user_sensitive_counts[user_id] += 1
    
    # Filter to only users with >= 4 interactions remaining after removing sensitive items
    eligible_users = {}
    for user_id, interactions in user_interactions.items():
        total_interactions = len(interactions)
        sensitive_count = user_sensitive_counts[user_id]
        remaining_interactions = total_interactions - sensitive_count
        if remaining_interactions >= 4:
            eligible_users[user_id] = interactions
    
    # Update user_interactions to only include eligible users
    user_interactions = eligible_users
    users_with_sensitive = {u for u in users_with_sensitive if u in eligible_users}
    
    total_users = len(user_interactions)
    total_interactions = sum(len(interactions) for interactions in user_interactions.values())
    n_users_with_sensitive = len(users_with_sensitive)
    
    print(f"    Total users (after filtering >= 4 interactions remaining): {total_users:,}")
    print(f"    Total interactions: {total_interactions:,}")
    print(f"    Users with sensitive items: {n_users_with_sensitive:,} ({n_users_with_sensitive/total_users*100:.2f}%)")
    
    # Step 2: Sample users ensuring at least 1% have sensitive items
    print("  Step 2: Sampling users (ensuring >=1% have sensitive items)...")
    all_users = list(user_interactions.keys())
    users_without_sensitive = [u for u in all_users if u not in users_with_sensitive]
    
    random.shuffle(all_users)
    random.shuffle(users_without_sensitive)
    
    n_sample = max(1, int(total_users * sample_ratio))
    min_sensitive_users = max(1, int(n_sample * 0.01))  # At least 1% of sampled users
    
    print(f"    Target sample size: {n_sample:,} users")
    print(f"    Minimum users with sensitive items: {min_sensitive_users:,}")
    
    # First, sample users with sensitive items
    sampled_sensitive_users = list(users_with_sensitive)[:min_sensitive_users]
    random.shuffle(sampled_sensitive_users)
    
    # Then, sample remaining users (can be with or without sensitive)
    remaining_needed = n_sample - len(sampled_sensitive_users)
    if remaining_needed > 0:
        # Remove already sampled sensitive users from all_users
        remaining_users = [u for u in all_users if u not in sampled_sensitive_users]
        random.shuffle(remaining_users)
        sampled_other_users = remaining_users[:remaining_needed]
    else:
        sampled_other_users = []
    
    sampled_users = set(sampled_sensitive_users + sampled_other_users)
    actual_sensitive_count = len([u for u in sampled_users if u in users_with_sensitive])
    
    print(f"    Sampled {len(sampled_users):,} users")
    print(f"    Users with sensitive items in sample: {actual_sensitive_count:,} ({actual_sensitive_count/len(sampled_users)*100:.2f}%)")
    
    # Step 3: Write output
    print("  Step 3: Writing output...")
    sampled_interactions = []
    for user_id in sampled_users:
        sampled_interactions.extend(user_interactions[user_id])
    
    sampled_interactions.sort(key=lambda x: (x['user_id:token'], float(x['timestamp:float'])))
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write(header + '\n')
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames, delimiter='\t')
        writer.writerows(sampled_interactions)
    
    print(f"    Written {len(sampled_interactions):,} interactions")
    print(f"    Output file: {output_file}")
    
    return len(sampled_users), len(sampled_interactions), actual_sensitive_count


def create_amazon_reviews_books_test_subset(input_file, output_file, sample_ratio, seed, main_dir):
    """
    Create test subset for Amazon Reviews Books dataset using USER-CENTRIC sampling.
    Ensures at least 1% of sampled users have sensitive interactions.
    Only includes users with >= 4 interactions remaining after removing sensitive items.
    
    Format: user_id:token	session_id:token	item_id:token	rating:float	timestamp:float
    """
    print(f"Creating Amazon Reviews Books test subset...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Sample ratio: {sample_ratio} ({sample_ratio*100}%)")
    print(f"  Seed: {seed}")
    
    random.seed(seed)
    
    # Step 0: Load sensitive items
    print("  Step 0: Loading sensitive items...")
    sensitive_items = load_sensitive_items_amazon_books(main_dir, category="health")
    
    # Step 1: Collect all users and their interactions, identify users with sensitive items
    # Also filter to only users with >= 4 interactions remaining after removing sensitive items
    user_interactions = defaultdict(list)
    users_with_sensitive = set()
    user_sensitive_counts = defaultdict(int)
    header = None
    
    print("  Step 1: Loading all interactions and identifying users with sensitive items...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        header = '\t'.join(reader.fieldnames)
        
        for row in reader:
            user_id = row['user_id:token']
            item_id = row['item_id:token']
            user_interactions[user_id].append(row)
            
            # Check if this user has sensitive items
            if item_id in sensitive_items:
                users_with_sensitive.add(user_id)
                user_sensitive_counts[user_id] += 1
    
    # Filter to only users with >= 4 interactions remaining after removing sensitive items
    eligible_users = {}
    for user_id, interactions in user_interactions.items():
        total_interactions = len(interactions)
        sensitive_count = user_sensitive_counts[user_id]
        remaining_interactions = total_interactions - sensitive_count
        if remaining_interactions >= 4:
            eligible_users[user_id] = interactions
    
    # Update user_interactions to only include eligible users
    user_interactions = eligible_users
    users_with_sensitive = {u for u in users_with_sensitive if u in eligible_users}
    
    total_users = len(user_interactions)
    total_interactions = sum(len(interactions) for interactions in user_interactions.values())
    n_users_with_sensitive = len(users_with_sensitive)
    
    print(f"    Total users (after filtering >= 4 interactions remaining): {total_users:,}")
    print(f"    Total interactions: {total_interactions:,}")
    print(f"    Users with sensitive items: {n_users_with_sensitive:,} ({n_users_with_sensitive/total_users*100:.2f}%)")
    
    # Step 2: Sample users ensuring at least 1% have sensitive items
    print("  Step 2: Sampling users (ensuring >=1% have sensitive items)...")
    all_users = list(user_interactions.keys())
    
    random.shuffle(all_users)
    
    n_sample = max(1, int(total_users * sample_ratio))
    min_sensitive_users = max(1, int(n_sample * 0.01))  # At least 1% of sampled users
    
    print(f"    Target sample size: {n_sample:,} users")
    print(f"    Minimum users with sensitive items: {min_sensitive_users:,}")
    
    # First, sample users with sensitive items
    sensitive_users_list = list(users_with_sensitive)
    random.shuffle(sensitive_users_list)
    sampled_sensitive_users = sensitive_users_list[:min_sensitive_users]
    
    # Then, sample remaining users
    remaining_needed = n_sample - len(sampled_sensitive_users)
    if remaining_needed > 0:
        remaining_users = [u for u in all_users if u not in sampled_sensitive_users]
        random.shuffle(remaining_users)
        sampled_other_users = remaining_users[:remaining_needed]
    else:
        sampled_other_users = []
    
    sampled_users = set(sampled_sensitive_users + sampled_other_users)
    actual_sensitive_count = len([u for u in sampled_users if u in users_with_sensitive])
    
    print(f"    Sampled {len(sampled_users):,} users")
    print(f"    Users with sensitive items in sample: {actual_sensitive_count:,} ({actual_sensitive_count/len(sampled_users)*100:.2f}%)")
    
    # Step 3: Write output
    print("  Step 3: Writing output...")
    sampled_interactions = []
    for user_id in sampled_users:
        sampled_interactions.extend(user_interactions[user_id])
    
    sampled_interactions.sort(key=lambda x: (x['user_id:token'], float(x['timestamp:float'])))
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write(header + '\n')
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames, delimiter='\t')
        writer.writerows(sampled_interactions)
    
    print(f"    Written {len(sampled_interactions):,} interactions")
    print(f"    Output file: {output_file}")
    
    return len(sampled_users), len(sampled_interactions), actual_sensitive_count


def create_instacart_test_subset(input_file, output_file, sample_ratio, seed, main_dir):
    """
    Create test subset for Instacart dataset using USER-CENTRIC sampling.
    Ensures at least 1% of sampled users have sensitive products in their baskets.
    Only includes users with >= 4 baskets remaining after removing sensitive items.
    
    Format: JSON with user_id -> list of baskets (each basket is a list of item_ids)
    """
    print(f"Creating Instacart test subset...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Sample ratio: {sample_ratio} ({sample_ratio*100}%)")
    print(f"  Seed: {seed}")
    
    random.seed(seed)
    
    # Step 0: Load sensitive products
    print("  Step 0: Loading sensitive products...")
    sensitive_products = load_sensitive_items_instacart(main_dir, category="alcohol")
    
    # Step 1: Load JSON data first (needed to identify users with sensitive in history)
    print("  Step 1: Loading JSON data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Step 2: Load user IDs from existing forget sets (these users have sensitive items)
    print("  Step 2: Loading users with sensitive items from existing forget sets...")
    users_with_sensitive = load_users_from_forget_sets(main_dir, category="alcohol")
    
    # Filter to only users that exist in the merged data
    users_with_sensitive = {uid for uid in users_with_sensitive if uid in data}
    
    # Step 3: Filter to only users with >= 4 baskets remaining after removing sensitive items
    # For Instacart, we assume users in forget sets will have their baskets removed
    # So we only keep users with >= 4 baskets total (they'll have >= 4 remaining)
    eligible_data = {}
    for user_id, baskets in data.items():
        if len(baskets) >= 4:
            eligible_data[user_id] = baskets
    
    data = eligible_data
    users_with_sensitive = {uid for uid in users_with_sensitive if uid in data}
    
    total_users = len(data)
    total_baskets = sum(len(baskets) for baskets in data.values())
    n_users_with_sensitive = len(users_with_sensitive)
    
    print(f"    Total users (after filtering >= 4 baskets): {total_users:,}")
    print(f"    Total baskets: {total_baskets:,}")
    print(f"    Users with sensitive items (from forget sets): {n_users_with_sensitive:,} ({n_users_with_sensitive/total_users*100:.2f}%)")
    
    # Step 3: Sample users ensuring at least 1% have sensitive products
    print("  Step 3: Sampling users (ensuring >=1% have sensitive products)...")
    all_users = list(data.keys())
    
    random.shuffle(all_users)
    
    n_sample = max(1, int(total_users * sample_ratio))
    min_sensitive_users = max(1, int(n_sample * 0.01))  # At least 1% of sampled users
    
    print(f"    Target sample size: {n_sample:,} users")
    print(f"    Minimum users with sensitive products: {min_sensitive_users:,}")
    
    # First, sample users with sensitive products (from forget sets)
    sensitive_users_list = list(users_with_sensitive)
    random.shuffle(sensitive_users_list)
    sampled_sensitive_users = sensitive_users_list[:min_sensitive_users]
    
    # Then, sample remaining users
    remaining_needed = n_sample - len(sampled_sensitive_users)
    if remaining_needed > 0:
        remaining_users = [u for u in all_users if u not in sampled_sensitive_users]
        random.shuffle(remaining_users)
        sampled_other_users = remaining_users[:remaining_needed]
    else:
        sampled_other_users = []
    
    sampled_users = sampled_sensitive_users + sampled_other_users
    actual_sensitive_count = len([u for u in sampled_users if u in users_with_sensitive])
    
    print(f"    Sampled {len(sampled_users):,} users")
    print(f"    Users with sensitive products: {actual_sensitive_count:,} ({actual_sensitive_count/len(sampled_users)*100:.2f}%)")
    
    # Step 4: Write output
    print("  Step 4: Including all baskets for sampled users...")
    output_data = {user_id: data[user_id] for user_id in sampled_users}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f)
    
    sampled_baskets = sum(len(baskets) for baskets in output_data.values())
    print(f"    Written {sampled_baskets:,} baskets from {len(sampled_users):,} users")
    print(f"    Output file: {output_file}")
    
    return len(sampled_users), sampled_baskets, actual_sensitive_count


def main():
    parser = argparse.ArgumentParser(
        description='Create small test subsets of datasets for fast testing'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['movielens', 'amazon_reviews_books', 'instacart'],
        help='Dataset name'
    )
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=0.01,
        help='Fraction of users to sample (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2,
        help='Random seed for reproducibility (default: 2)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory (default: dataset/{dataset}/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: dataset/{dataset}_test/)'
    )
    
    args = parser.parse_args()
    
    # Set default paths
    if args.input_dir is None:
        args.input_dir = f"dataset/{args.dataset}"
    if args.output_dir is None:
        args.output_dir = f"dataset/{args.dataset}_test"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine input and output files
    if args.dataset == 'movielens':
        input_file = os.path.join(args.input_dir, 'movielens.inter')
        output_file = os.path.join(args.output_dir, 'movielens_test.inter')
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return
        n_users, n_interactions, n_sensitive = create_movielens_test_subset(
            input_file, output_file, args.sample_ratio, args.seed, args.input_dir
        )
        
    elif args.dataset == 'amazon_reviews_books':
        input_file = os.path.join(args.input_dir, 'amazon_reviews_books.inter')
        output_file = os.path.join(args.output_dir, 'amazon_reviews_books_test.inter')
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return
        n_users, n_interactions, n_sensitive = create_amazon_reviews_books_test_subset(
            input_file, output_file, args.sample_ratio, args.seed, args.input_dir
        )
        
    elif args.dataset == 'instacart':
        input_file = os.path.join(args.input_dir, 'instacart_merged.json')
        output_file = os.path.join(args.output_dir, 'instacart_test_merged.json')
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return
        n_users, n_baskets, n_sensitive = create_instacart_test_subset(
            input_file, output_file, args.sample_ratio, args.seed, args.input_dir
        )
        n_interactions = n_baskets  # For consistency in output
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Sample ratio: {args.sample_ratio} ({args.sample_ratio*100}%)")
    print(f"Seed: {args.seed}")
    print(f"Sampled users: {n_users:,}")
    if args.dataset == 'instacart':
        print(f"Sampled baskets: {n_interactions:,}")
    else:
        print(f"Sampled interactions: {n_interactions:,}")
    print(f"Users with sensitive items: {n_sensitive:,} ({n_sensitive/n_users*100:.2f}%)")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    print("\nNext steps:")
    print("1. Generate sensitive items for the test dataset (if needed)")
    print("2. Run generate_sensitive_forget_sets.sh in the test dataset directory")
    print("   (Make sure to update the dataset name in the script to use *_test.inter)")


if __name__ == "__main__":
    main()
