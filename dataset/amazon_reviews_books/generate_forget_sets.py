import csv
import random
import argparse
from collections import defaultdict
import os


def load_sensitive_items(sensitive_items_file):
    """Load sensitive item IDs from file."""
    sensitive_items = set()
    with open(sensitive_items_file, 'r', encoding='utf-8') as f:
        for line in f:
            item_id = line.strip()
            if item_id:
                sensitive_items.add(item_id)
    print(f"Loaded {len(sensitive_items)} sensitive items")
    return sensitive_items


def load_interactions_for_forget_set(dataset_file, sensitive_items, threshold=None):
    """
    Stream through interactions and organize by user, keeping only sensitive interactions.
    Memory efficient for user-centric sampling.

    Returns:
        - user_sensitive_interactions: dict mapping user_id to list of their sensitive interaction lines
        - header: the header line from the file
        - total_count: total number of interactions in dataset
    """
    user_sensitive_interactions = defaultdict(list)
    total_count = 0

    print(f"Streaming through {dataset_file} to find sensitive interactions...")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        # Read the first line to get the header
        header_line = f.readline().strip()

        # Parse the header and normalize field names
        # Remove type annotations like :token, :float
        fields = [field.split(':')[0] for field in header_line.split('\t')]

        # Create reader with normalized field names
        reader = csv.DictReader(f, fieldnames=fields, delimiter='\t')

        for row in reader:
            total_count += 1

            # Only keep interactions with sensitive items which are used in training
            if row['item_id'] in sensitive_items and (threshold is None or float(row['rating']) >= threshold):
                # Store the full line for this user
                line = '\t'.join([row['user_id'], row['session_id'], row['item_id'], row['rating'], row['timestamp']])
                user_sensitive_interactions[row['user_id']].append(line)

    total_sensitive = sum(len(interactions) for interactions in user_sensitive_interactions.values())

    print(f"Total interactions in dataset: {total_count}")
    print(f"Found {len(user_sensitive_interactions)} users with sensitive items")
    print(f"Total sensitive interactions: {total_sensitive}")

    return user_sensitive_interactions, header_line, total_count


def sample_forget_set_user_centric(user_sensitive_interactions, target_size, seed):
    """
    Sample a forget set using USER-CENTRIC approach:
    - Randomly shuffle users
    - For each user, add ALL their sensitive interactions
    - Continue until target size is met or exceeded

    Args:
        user_sensitive_interactions: dict mapping user_id to list of their sensitive interaction lines
        target_size: target number of interactions in forget set
        seed: random seed for reproducibility

    Returns:
        forget_set: list of interaction lines (strings)
    """
    random.seed(seed)

    # Get list of users and shuffle
    user_list = list(user_sensitive_interactions.keys())
    random.shuffle(user_list)

    forget_set = []
    sampled_users = []

    print(f"\nSampling forget set (USER-CENTRIC, target size: {target_size}, seed: {seed})...")

    for user_id in user_list:
        # Add all sensitive interactions for this user
        user_interactions = user_sensitive_interactions[user_id]
        forget_set.extend(user_interactions)
        sampled_users.append(user_id)

        # Stop once we meet or exceed target
        if len(forget_set) >= target_size:
            break

    print(f"Forget set size: {len(forget_set)} interactions from {len(sampled_users)} users")

    return forget_set


def write_dataset(interactions, output_file, header):
    """Write interactions to file in the same format as original dataset."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for line in interactions:
            f.write(line + '\n')

    print(f"Written {len(interactions)} interactions to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate forget sets for unlearning experiments with USER-CENTRIC sampling'
    )
    parser.add_argument(
        '--dataset',
        default='amazon_reviews_books.inter',
        help='Input dataset file (preprocessed interactions with sessions)'
    )
    parser.add_argument(
        '--sensitive-items',
        default='sensitive_asins_health.txt',
        help='File containing sensitive ASINs (one per line)'
    )
    parser.add_argument(
        '--forget-ratio',
        type=float,
        default=0.0001,
        help='Ratio of dataset to include in forget set (default: 0.0001)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2,
        help='Random seed for reproducibility (default: 2)'
    )
    parser.add_argument(
        '--rating-threshold',
        type=float,
        default=None,
        help='Only include interactions with rating >= threshold'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found!")
        return

    if not os.path.exists(args.sensitive_items):
        print(f"Error: Sensitive items file '{args.sensitive_items}' not found!")
        print("Please run identify_sensitive_items.py first.")
        return

    # Load data
    sensitive_items = load_sensitive_items(args.sensitive_items)

    # Stream through dataset and organize by user
    user_sensitive_interactions, header, total_count = load_interactions_for_forget_set(
        args.dataset, sensitive_items, threshold=args.rating_threshold
    )

    if not user_sensitive_interactions:
        print("Error: No users found with sensitive item interactions!")
        return

    # Calculate target forget set size
    target_size = int(total_count * args.forget_ratio)
    print(f"\nTarget forget set size: {target_size} ({args.forget_ratio * 100}% of dataset)")

    # Check if we have enough sensitive interactions
    total_sensitive = sum(len(interactions) for interactions in user_sensitive_interactions.values())
    if total_sensitive < target_size:
        print(f"Warning: Only {total_sensitive} sensitive interactions available, less than target {target_size}")
        print("Will use all available sensitive interactions.")
        target_size = total_sensitive

    # Sample forget set using user-centric approach
    forget_set = sample_forget_set_user_centric(user_sensitive_interactions, target_size, args.seed)

    # Write outputs
    # Extract category from filename pattern "sensitive_asins_{category}.txt"
    base_filename = args.sensitive_items[:-len('.txt')]  # Remove .txt
    if base_filename.startswith('sensitive_asins_'):
        category = base_filename[len('sensitive_asins_'):]
    else:
        # Fallback for backwards compatibility
        category = base_filename.split("_")[-1]
    forget_file = f"{args.dataset[:-len('.inter')]}_unlearn_pairs_sensitive_category_{category}_seed_{args.seed}_unlearning_fraction_{args.forget_ratio}.inter"

    write_dataset(forget_set, forget_file, header)

    user_count, item_count = len(set(line.split('\t')[0] for line in forget_set)), len(set(line.split('\t')[2] for line in forget_set))
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original dataset: {total_count} interactions")
    print(f"Forget set: {len(forget_set)} interactions ({len(forget_set)/total_count*100:.3f}%)")
    print(f"Forget set users: {user_count}")
    print(f"Forget set items: {item_count}")
    print(f"\nOutput file:")
    print(f"  - {forget_file}")
    print("="*60)


if __name__ == "__main__":
    main()
