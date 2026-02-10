#!/usr/bin/env python3
"""
Create forget sets for sensitive categories in 30music dataset

This script:
1. Identifies tracks with sensitive tags (health, explicit)
2. Finds all interactions with those tracks in the .inter file
3. Creates separate forget set files for each category
4. Only includes users with >= min_interactions sensitive interactions
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Define sensitive category mappings
SENSITIVE_CATEGORIES = {
    'health': {
        'tag_ids': {'154395', '15345', '233689', '214036', '59294'},
        'tags': ['mental', 'anxiety', 'suicide', 'self-harm', 'depression'],
        'description': 'Mental health related content that may be triggering'
    },
    'explicit': {
        'tag_ids': {'77138', '193026', '174963'},
        'tags': ['explicit', 'profanity', 'nsfw'],
        'description': 'Explicit content with profanity or mature themes'
    }
}

def identify_sensitive_tracks(tracks_file, categories):
    """
    Identify tracks that have sensitive tags

    Args:
        tracks_file: Path to tracks.idomaar file
        categories: Dictionary of sensitive categories with tag_ids

    Returns:
        Dictionary mapping category name to set of track IDs
    """
    print("Identifying sensitive tracks from tags")

    # Initialize sets for each category
    sensitive_tracks = {cat: set() for cat in categories}

    print(f"\nProcessing tracks from: {tracks_file}")

    total_tracks = 0
    with open(tracks_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if i % 500000 == 0:
                print(f"  Processed {i:,} tracks...")

            total_tracks += 1
            parts = line.strip().split('\t')

            if len(parts) < 5:
                continue

            track_id = parts[1]

            try:
                relations = json.loads(parts[4])

                if 'tags' in relations and relations['tags']:
                    for tag in relations['tags']:
                        tag_id = str(tag['id'])

                        # Check each category
                        for cat_name, cat_info in categories.items():
                            if tag_id in cat_info['tag_ids']:
                                sensitive_tracks[cat_name].add(track_id)
            except:
                pass

    print(f"\nProcessed {total_tracks:,} total tracks")
    print("\nSensitive tracks found:")
    for cat_name, track_set in sensitive_tracks.items():
        print(f"  {cat_name}: {len(track_set):,} tracks")

    return sensitive_tracks

def create_forget_sets(inter_file, sensitive_tracks, output_dir, seeds=[2, 3, 5, 7, 11],
                       fractions=[1e-6, 1e-5, 1e-4], min_interactions=3, min_remaining_interactions=3):
    """
    Create forget set files using user-centric sampling:
    - For each user with sensitive interactions, add ALL their sensitive interactions
    - Continue until the target sample size is met
    - Only users with >= min_interactions sensitive interactions are eligible
    - Only users who will have >= min_remaining_interactions after unlearning are eligible

    Args:
        inter_file: Path to .inter file
        sensitive_tracks: Dictionary mapping category to set of sensitive track IDs
        output_dir: Directory to save forget set files
        seeds: List of random seeds for sampling users
        fractions: List of unlearning fractions to sample
        min_interactions: Minimum number of sensitive interactions for a user to be eligible
        min_remaining_interactions: Minimum number of interactions that must remain after unlearning
    """
    print(f"\nCreating forget sets (USER-CENTRIC SAMPLING, min_interactions={min_interactions}, min_remaining={min_remaining_interactions})")

    # First, load all interactions and organize by user and category
    print(f"\nLoading interactions from: {inter_file}")

    # Store interactions by category and user: {category: {user_id: [interactions]}}
    user_sensitive_interactions = defaultdict(lambda: defaultdict(list))
    # Track total interactions per user
    user_total_interactions = defaultdict(int)
    total_interactions = 0

    with open(inter_file, 'r') as f:
        header = f.readline().strip()

        for i, line in enumerate(f, 1):
            if i % 1000000 == 0:
                print(f"  Processed {i:,} interactions...")

            total_interactions += 1
            parts = line.strip().split('\t')

            if len(parts) < 4:
                continue

            user_id, session_id, item_id, timestamp = parts

            # Track total interactions for this user
            user_total_interactions[user_id] += 1

            # Check if this interaction involves a sensitive track
            for cat_name, track_set in sensitive_tracks.items():
                if item_id in track_set:
                    user_sensitive_interactions[cat_name][user_id].append(line.strip())

    print(f"\nTotal interactions: {total_interactions:,}")
    print("\nUser-centric sensitive interaction statistics (BEFORE filtering):")
    for cat_name, user_dict in user_sensitive_interactions.items():
        total_sensitive = sum(len(interactions) for interactions in user_dict.values())
        print(f"  {cat_name}:")
        print(f"    Users with sensitive interactions: {len(user_dict):,}")
        print(f"    Total sensitive interactions: {total_sensitive:,}")
        print(f"    ({total_sensitive/total_interactions*100:.4f}% of total)")
        print(f"    Avg interactions per user: {total_sensitive/len(user_dict):.2f}")

    # Filter users based on:
    # 1. Minimum sensitive interactions threshold
    # 2. Minimum remaining interactions after unlearning
    print(f"\nFiltering users with >= {min_interactions} sensitive interactions AND >= {min_remaining_interactions} remaining:")
    eligible_user_interactions = {}

    for cat_name, user_dict in user_sensitive_interactions.items():
        eligible_users = {}
        filtered_too_few_sensitive = 0
        filtered_too_few_remaining = 0

        for user_id, interactions in user_dict.items():
            num_sensitive = len(interactions)
            num_total = user_total_interactions[user_id]
            num_remaining = num_total - num_sensitive

            # Check both conditions
            if num_sensitive < min_interactions:
                filtered_too_few_sensitive += 1
                continue
            if num_remaining < min_remaining_interactions:
                filtered_too_few_remaining += 1
                continue

            eligible_users[user_id] = interactions

        eligible_user_interactions[cat_name] = eligible_users

        total_eligible = sum(len(interactions) for interactions in eligible_users.values())
        original_total = sum(len(interactions) for interactions in user_dict.values())

        print(f"  {cat_name}:")
        print(f"    Eligible users: {len(eligible_users):,} (from {len(user_dict):,})")
        print(f"    Filtered out (< {min_interactions} sensitive): {filtered_too_few_sensitive:,}")
        print(f"    Filtered out (< {min_remaining_interactions} remaining): {filtered_too_few_remaining:,}")
        print(f"    Eligible interactions: {total_eligible:,} (from {original_total:,})")
        print(f"    ({total_eligible/total_interactions*100:.4f}% of total dataset)")
        if len(eligible_users) > 0:
            print(f"    Avg interactions per eligible user: {total_eligible/len(eligible_users):.2f}")

    # Create forget sets for each category, seed, and fraction
    print("\nGenerating forget set files (user-centric sampling)")

    import random

    for cat_name, user_dict in eligible_user_interactions.items():
        if len(user_dict) == 0:
            print(f"\nSkipping {cat_name}: no eligible users found")
            continue

        total_sensitive = sum(len(interactions) for interactions in user_dict.values())
        print(f"\n{cat_name.upper()}:")
        print(f"  Eligible users (>= {min_interactions} interactions): {len(user_dict):,}")
        print(f"  Total eligible interactions: {total_sensitive:,}")

        for seed in seeds:
            for fraction in fractions:
                # Calculate target sample size based on total dataset size
                target_size = int(total_interactions * fraction)

                if target_size == 0:
                    print(f"  Skipping seed={seed}, fraction={fraction}: target size = 0")
                    continue

                if target_size > total_sensitive:
                    print(f"  Skipping seed={seed}, fraction={fraction}: target size ({target_size}) exceeds available sensitive interactions ({total_sensitive})")
                    continue

                # USER-CENTRIC SAMPLING:
                # Randomly shuffle users and add all their sensitive interactions
                # until we meet or exceed the target size
                random.seed(seed)
                user_ids = list(user_dict.keys())
                random.shuffle(user_ids)

                sampled_interactions = []
                sampled_users = []
                current_size = 0

                for user_id in user_ids:
                    user_interactions = user_dict[user_id]

                    # Add all sensitive interactions from this user
                    sampled_interactions.extend(user_interactions)
                    sampled_users.append(user_id)
                    current_size += len(user_interactions)

                    # Stop once we meet or exceed the target
                    if current_size >= target_size:
                        break

                # Create output filename
                output_file = output_dir / f"30music_unlearn_pairs_sensitive_category_{cat_name}_seed_{seed}_unlearning_fraction_{fraction}.inter"

                # Write forget set
                with open(output_file, 'w') as f:
                    f.write(header + '\n')
                    for interaction in sampled_interactions:
                        f.write(interaction + '\n')

                print(f"  Created: seed={seed}, fraction={fraction}")
                print(f"    Target size: {target_size:,}, Actual size: {current_size:,}")
                print(f"    Users sampled: {len(sampled_users):,}")
                print(f"    File: {output_file.name}")

    print("\nForget set generation complete!")

def main():
    # Set paths
    base_dir = Path(__file__).parent
    tracks_file = base_dir / 'entities' / 'tracks.idomaar'
    inter_file = base_dir / '30music.inter'
    output_dir = base_dir

    # Check files exist
    if not tracks_file.exists():
        print(f"Error: Tracks file not found: {tracks_file}")
        sys.exit(1)

    if not inter_file.exists():
        print(f"Error: .inter file not found: {inter_file}")
        print("Please run convert_sessions_to_inter.py first")
        sys.exit(1)

    print("30MUSIC FORGET SET GENERATOR\n")
    print("Sensitive categories:")
    for cat_name, cat_info in SENSITIVE_CATEGORIES.items():
        print(f"\n{cat_name}:")
        print(f"  Tags: {', '.join(cat_info['tags'])}")
        print(f"  Description: {cat_info['description']}")

    # Step 1: Identify sensitive tracks
    sensitive_tracks = identify_sensitive_tracks(tracks_file, SENSITIVE_CATEGORIES)

    # Save sensitive track lists
    # Use 'sensitive_asins' naming for consistency with other datasets
    print("\nSaving sensitive track IDs...")
    for cat_name, track_set in sensitive_tracks.items():
        output_file = output_dir / f"sensitive_asins_{cat_name}.txt"
        with open(output_file, 'w') as f:
            for track_id in sorted(track_set, key=lambda x: int(x)):
                f.write(f"{track_id}\n")
        print(f"  {cat_name}: {output_file}")

    # Step 2: Create forget sets (only users with >= 3 sensitive interactions AND >= 3 remaining)
    create_forget_sets(inter_file, sensitive_tracks, output_dir, min_interactions=3, min_remaining_interactions=3)

if __name__ == "__main__":
    main()