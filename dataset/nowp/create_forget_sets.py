#!/usr/bin/env python3
"""
Create forget sets for sensitive categories in NowPlaying dataset

This script:
1. Identifies tracks with sensitive content using track titles and artist names
2. Uses keyword-based filtering for categories like health, explicit content, etc.
3. Finds all interactions with those tracks in the .inter file
4. Creates separate forget set files for each category using USER-CENTRIC sampling

Note: The NowPlaying dataset doesn't have tags, so we use keyword matching
on track titles and artist names to identify sensitive content.
"""

import csv
import sys
import re
from pathlib import Path
from collections import defaultdict

# Define sensitive category mappings with keywords
# Each category has two types of keywords:
# - 'exact': require full word match (to avoid false positives)
# - 'relaxed': match as substring (for words unlikely to cause false positives)
SENSITIVE_CATEGORIES = {
    'health': {
        'exact': ['die'],
        'relaxed': [
            'depress', 'anxiety', 'suicide', 'self harm', 'therapy',
            'ptsd', 'trauma', 'breakdown', 'bipolar', 'schizo',
            'medication', 'overdose', 'cutting', 'death wish',
            'end it all', 'kill myself'
        ],
        'description': 'Mental health related content that may be triggering'
    },
    'explicit': {
        'exact': ['ass', 'hell', 'damn', 'sex'],
        'relaxed': [
            'fuck', 'shit', 'bitch', 'drugs', 'explicit',
            'parental advisory', 'nsfw', 'uncensored', 'xxx',
            'cocaine', 'heroin', 'weed', 'marijuana', 'drunk', 'alcohol'
        ],
        'description': 'Explicit content with profanity or mature themes'
    },
    'violence': {
        'exact': ['die', 'pain', 'kill', 'hell'],
        'relaxed': [
            'murder', 'blood', 'violence', 'gun', 'shoot', 'weapon',
            'war', 'fight', 'attack', 'revenge', 'torture', 'suffer',
            'massacre', 'slaughter', 'death', 'corpse', 'brutal'
        ],
        'description': 'Violent content and themes'
    }
}

def identify_sensitive_tracks(csv_file, categories):
    """
    Identify tracks that have sensitive content based on keywords in titles/artists
    Memory efficient: streams through the CSV file

    Args:
        csv_file: Path to sessions_2018.csv
        categories: Dictionary of sensitive categories with keywords

    Returns:
        Dictionary mapping category name to set of musicbrainz IDs
    """
    print("Identifying sensitive tracks from CSV file")
    print("Using keyword matching on track titles and artist names")

    # Initialize sets for each category
    sensitive_tracks = {cat: set() for cat in categories}

    # Compile regex patterns for each category (case insensitive)
    # Two patterns per category: exact word match and relaxed substring match
    exact_patterns = {}
    relaxed_patterns = {}

    for cat_name, cat_info in categories.items():
        # Exact word match pattern (with word boundaries)
        if cat_info.get('exact'):
            exact_pattern = '|'.join(r'\b' + re.escape(kw) + r'\b' for kw in cat_info['exact'])
            exact_patterns[cat_name] = re.compile(exact_pattern, re.IGNORECASE)
        else:
            exact_patterns[cat_name] = None

        # Relaxed substring match pattern
        if cat_info.get('relaxed'):
            relaxed_pattern = '|'.join(re.escape(kw) for kw in cat_info['relaxed'])
            relaxed_patterns[cat_name] = re.compile(relaxed_pattern, re.IGNORECASE)
        else:
            relaxed_patterns[cat_name] = None

    print(f"\nProcessing tracks from: {csv_file}")

    total_tracks_checked = 0
    unique_tracks = set()

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, quotechar="'")

        for i, row in enumerate(reader, 1):
            if i % 1000000 == 0:
                print(f"  Processed {i:,} lines...")

            if len(row) < 7:
                continue

            track_title = row[3]
            artist_name = row[4]
            musicbrainz_id = row[5]
            session_id = row[6]

            # Skip header
            if session_id == 'session_id':
                continue

            if not musicbrainz_id:
                continue

            # Track unique items
            if musicbrainz_id not in unique_tracks:
                unique_tracks.add(musicbrainz_id)
                total_tracks_checked += 1

                # Combine title and artist for keyword search
                text = f"{track_title} {artist_name}"

                # Check each category with both exact and relaxed patterns
                for cat_name in categories.keys():
                    matched = False

                    # Check exact word match pattern
                    if exact_patterns[cat_name] and exact_patterns[cat_name].search(text):
                        matched = True

                    # Check relaxed substring match pattern
                    if not matched and relaxed_patterns[cat_name] and relaxed_patterns[cat_name].search(text):
                        matched = True

                    if matched:
                        sensitive_tracks[cat_name].add(musicbrainz_id)

    print(f"\nProcessed {total_tracks_checked:,} unique tracks")
    print("\nSensitive tracks found:")
    for cat_name, track_set in sensitive_tracks.items():
        print(f"  {cat_name}: {len(track_set):,} tracks")

    return sensitive_tracks

def create_forget_sets(inter_file, sensitive_tracks, output_dir, seeds=[2, 3, 5, 7, 11],
                       fractions=[1e-6, 1e-5, 1e-4]):
    """
    Create forget set files using user-centric sampling:
    - For each user with sensitive interactions, add ALL their sensitive interactions
    - Continue until the target sample size is met

    Args:
        inter_file: Path to .inter file
        sensitive_tracks: Dictionary mapping category to set of sensitive track IDs
        output_dir: Directory to save forget set files
        seeds: List of random seeds for sampling users
        fractions: List of unlearning fractions to sample
    """
    print("\nCreating forget sets (USER-CENTRIC SAMPLING)")

    # First, load all interactions and organize by user and category
    print(f"\nLoading interactions from: {inter_file}")

    # Store interactions by category and user: {category: {user_id: [interactions]}}
    user_sensitive_interactions = defaultdict(lambda: defaultdict(list))
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

            # Check if this interaction involves a sensitive track
            for cat_name, track_set in sensitive_tracks.items():
                if item_id in track_set:
                    user_sensitive_interactions[cat_name][user_id].append(line.strip())

    print(f"\nTotal interactions: {total_interactions:,}")
    print("\nUser-centric sensitive interaction statistics:")
    for cat_name, user_dict in user_sensitive_interactions.items():
        total_sensitive = sum(len(interactions) for interactions in user_dict.values())
        print(f"  {cat_name}:")
        print(f"    Users with sensitive interactions: {len(user_dict):,}")
        print(f"    Total sensitive interactions: {total_sensitive:,}")
        print(f"    ({total_sensitive/total_interactions*100:.4f}% of total)")
        if len(user_dict) > 0:
            print(f"    Avg interactions per user: {total_sensitive/len(user_dict):.2f}")

    # Create forget sets for each category, seed, and fraction
    print("\nGenerating forget set files (user-centric sampling)")

    import random

    for cat_name, user_dict in user_sensitive_interactions.items():
        if len(user_dict) == 0:
            print(f"\nSkipping {cat_name}: no users with sensitive interactions found")
            continue

        total_sensitive = sum(len(interactions) for interactions in user_dict.values())
        print(f"\n{cat_name.upper()}:")
        print(f"  Users with sensitive interactions: {len(user_dict):,}")
        print(f"  Total sensitive interactions: {total_sensitive:,}")

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
                output_file = output_dir / f"nowp_unlearn_pairs_sensitive_category_{cat_name}_seed_{seed}_unlearning_fraction_{fraction}.inter"

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
    csv_file = base_dir / 'sessions_2018.csv'
    inter_file = base_dir / 'nowp.inter'
    output_dir = base_dir

    # Check files exist
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)

    if not inter_file.exists():
        print(f"Error: .inter file not found: {inter_file}")
        print("Please run preprocess_nowp.py first")
        sys.exit(1)

    print("NOWPLAYING FORGET SET GENERATOR\n")
    print("Sensitive categories:")
    for cat_name, cat_info in SENSITIVE_CATEGORIES.items():
        print(f"\n{cat_name}:")
        exact_kw = cat_info.get('exact', [])
        relaxed_kw = cat_info.get('relaxed', [])
        all_keywords = exact_kw + relaxed_kw
        print(f"  Keywords (exact match): {', '.join(exact_kw[:5])}{'...' if len(exact_kw) > 5 else ''}")
        print(f"  Keywords (relaxed match): {', '.join(relaxed_kw[:5])}{'...' if len(relaxed_kw) > 5 else ''}")
        print(f"  Description: {cat_info['description']}")

    # Step 1: Identify sensitive tracks
    sensitive_tracks = identify_sensitive_tracks(csv_file, SENSITIVE_CATEGORIES)

    # Save sensitive track lists
    print("\nSaving sensitive track IDs...")
    for cat_name, track_set in sensitive_tracks.items():
        output_file = output_dir / f"sensitive_tracks_{cat_name}.txt"
        with open(output_file, 'w') as f:
            for track_id in sorted(track_set):
                f.write(f"{track_id}\n")
        print(f"  {cat_name}: {output_file}")

    # Step 2: Create forget sets
    create_forget_sets(inter_file, sensitive_tracks, output_dir)

if __name__ == "__main__":
    main()
