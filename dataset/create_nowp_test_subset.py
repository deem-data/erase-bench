#!/usr/bin/env python3
"""
Create small test subset of nowp dataset for fast testing.

This script uses USER-CENTRIC sampling: it randomly samples a small percentage
of users and includes ALL their interactions (sessions). This ensures that user
behavior patterns are preserved in the test subset.

Usage:
    python create_nowp_test_subset.py --sample-ratio 0.01 --seed 2
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def create_nowp_test_subset(input_file, output_file, sample_ratio, seed):
    """
    Create test subset for nowp dataset using USER-CENTRIC sampling.
    
    Format: user_id:token	session_id:token	item_id:token	timestamp:float
    """
    print(f"Creating nowp test subset...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Sample ratio: {sample_ratio} ({sample_ratio*100}%)")
    print(f"  Seed: {seed}")
    
    random.seed(seed)
    
    # Step 1: Collect all users and their interactions
    user_interactions = defaultdict(list)
    header = None
    
    print("  Step 1: Loading all interactions...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        header = '\t'.join(reader.fieldnames)
        
        for row in reader:
            user_id = row['user_id:token']
            user_interactions[user_id].append(row)
    
    total_users = len(user_interactions)
    total_interactions = sum(len(interactions) for interactions in user_interactions.values())
    
    print(f"    Total users: {total_users:,}")
    print(f"    Total interactions: {total_interactions:,}")
    
    # Step 2: Sample users
    print("  Step 2: Sampling users...")
    all_users = list(user_interactions.keys())
    random.shuffle(all_users)
    
    n_sample = max(1, int(total_users * sample_ratio))
    
    print(f"    Target sample size: {n_sample:,} users")
    
    sampled_users = set(all_users[:n_sample])
    
    print(f"    Sampled {len(sampled_users):,} users")
    
    # Step 3: Write output
    print("  Step 3: Writing output...")
    sampled_interactions = []
    for user_id in sampled_users:
        sampled_interactions.extend(user_interactions[user_id])
    
    # Sort by user_id, then session_id, then timestamp
    sampled_interactions.sort(key=lambda x: (
        x['user_id:token'], 
        int(x['session_id:token']) if x['session_id:token'].isdigit() else x['session_id:token'],
        float(x['timestamp:float'])
    ))
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write(header + '\n')
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames, delimiter='\t')
        writer.writerows(sampled_interactions)
    
    sampled_interactions_count = len(sampled_interactions)
    sampled_sessions = len(set(row['session_id:token'] for row in sampled_interactions))
    
    print(f"    Written {sampled_interactions_count:,} interactions")
    print(f"    Sampled sessions: {sampled_sessions:,}")
    print(f"    Output file: {output_file}")
    
    return len(sampled_users), sampled_interactions_count, sampled_sessions


def main():
    parser = argparse.ArgumentParser(
        description='Create small test subset of nowp dataset for fast testing'
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
        '--input-file',
        type=str,
        default='nowp/nowp.inter',
        help='Input file path (default: nowp/nowp.inter, relative to dataset directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dataset/nowp_test',
        help='Output directory (default: dataset/nowp_test)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output file
    output_file = os.path.join(args.output_dir, 'nowp_test.inter')
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    n_users, n_interactions, n_sessions = create_nowp_test_subset(
        args.input_file, output_file, args.sample_ratio, args.seed
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset: nowp")
    print(f"Sample ratio: {args.sample_ratio} ({args.sample_ratio*100}%)")
    print(f"Seed: {args.seed}")
    print(f"Sampled users: {n_users:,}")
    print(f"Sampled sessions: {n_sessions:,}")
    print(f"Sampled interactions: {n_interactions:,}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    print("\nNext steps:")
    print("1. Generate fraud sessions for the test dataset:")
    print("   python create_fraud_sessions_sbr.py --dataset nowp_test --attack bandwagon --target_strategy unpopular --poisoning_ratio 0.01 --seed 2 --n_target_items 10")


if __name__ == "__main__":
    main()
