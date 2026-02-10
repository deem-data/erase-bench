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
    Stream through interactions and only keep those with sensitive items.
    Memory efficient - doesn't load the entire dataset.
    
    Returns:
        - sensitive_interactions: dict mapping user_id to list of their sensitive interactions
        - total_count: total number of interactions in dataset
    """
    sensitive_interactions = defaultdict(list)
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
            
            # Only keep interactions with sensitive items which are used in training (aka have a rating above threshold)
            if row['item_id'] in sensitive_items and (threshold is None or float(row['rating']) >= threshold):
                sensitive_interactions[row['user_id']].append(row)
    
    total_sensitive = sum(len(interactions) for interactions in sensitive_interactions.values())
    
    print(f"Total interactions in dataset: {total_count}")
    print(f"Found {len(sensitive_interactions)} users with sensitive items")
    print(f"Total sensitive interactions: {total_sensitive}")
    
    return sensitive_interactions, total_count


def sample_forget_set(sensitive_interactions, target_size, seed):
    """
    Sample a forget set by selecting users and including all their sensitive interactions.
    
    Args:
        sensitive_interactions: dict mapping user_id to list of their sensitive interaction dicts
        target_size: target number of interactions in forget set
        seed: random seed for reproducibility
    
    Returns:
        forget_set: list of interaction dicts
    """
    random.seed(seed)
    
    # Get list of users and shuffle
    user_list = list(sensitive_interactions.keys())
    random.shuffle(user_list)
    
    forget_set = []
    
    print(f"\nSampling forget set (target size: {target_size}, seed: {seed})...")
    
    for user_id in user_list:
        # Add all sensitive interactions for this user
        user_interactions = sensitive_interactions[user_id]
        
        # Check if adding this user would exceed target
        if len(forget_set) + len(user_interactions) > target_size:
            # Check if we should add this user anyway (to get closer to target)
            current_size = len(forget_set)
            with_user = current_size + len(user_interactions)
            
            # Add if it gets us closer to target
            if abs(with_user - target_size) < abs(current_size - target_size):
                forget_set.extend(user_interactions)
            break
        
        forget_set.extend(user_interactions)
        
        if len(forget_set) >= target_size:
            break
    
    print(f"Forget set size: {len(forget_set)} interactions from {len(set(row['user_id'] for row in forget_set))} users")
    
    return forget_set


def write_dataset(interactions, output_file):
    """Write interactions to file in the same format as original dataset."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        writer = csv.DictWriter(
            f,
            fieldnames=['user_id', 'item_id', 'rating', 'timestamp'],
            delimiter='\t'
        )
        writer.writerows(interactions)
    
    print(f"Written {len(interactions)} interactions to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate forget and retain sets for unlearning experiments'
    )
    parser.add_argument(
        '--dataset',
        default='amazon_reviews.inter',
        help='Input dataset file (preprocessed interactions)'
    )
    parser.add_argument(
        '--sensitive-items',
        default='sensitive_asins.txt',
        help='File containing sensitive ASINs (one per line) - use output from extract_asins_from_reviews.py'
    )
    parser.add_argument(
        '--forget-ratio',
        type=float,
        default=0.001,
        help='Ratio of dataset to include in forget set (default: 0.001)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-prefix',
        default='forget_set',
        help='Prefix for output files (default: forget_set)'
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=None,
        help="For implicit collaborative filtering only binary signals are used (interaction between user and item: yes/no)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found!")
        return
    
    if not os.path.exists(args.sensitive_items):
        print(f"Error: Sensitive items file '{args.sensitive_items}' not found!")
        print("Please run the following workflow:")
        print("  1. identify_sensitive_items.py (to get parent ASINs from metadata)")
        print("  2. extract_asins_from_reviews.py (to get ASINs from your review data)")
        print("Then use the output file with this script.")
        return
    
    # Load data
    sensitive_items = load_sensitive_items(args.sensitive_items)
    
    # Stream through dataset and only keep sensitive interactions
    sensitive_interactions, total_count = load_interactions_for_forget_set(args.dataset, sensitive_items, threshold=args.rating_threshold)
    
    if not sensitive_interactions:
        print("Error: No users found with sensitive item interactions!")
        return
    
    # Calculate target forget set size
    target_size = int(total_count * args.forget_ratio)
    print(f"\nTarget forget set size: {target_size} ({args.forget_ratio * 100}% of dataset)")
    
    # Check if we have enough sensitive interactions
    total_sensitive = sum(len(interactions) for interactions in sensitive_interactions.values())
    if total_sensitive < target_size:
        print(f"Warning: Only {total_sensitive} sensitive interactions available, less than target {target_size}")
        print("Will use all available sensitive interactions.")
        target_size = total_sensitive
    
    # Sample forget set
    forget_set = sample_forget_set(sensitive_interactions, target_size, args.seed)
    
    # Write outputs
    # Extract category from filename pattern "sensitive_asins_{category}.txt"
    base_filename = args.sensitive_items[:-len('.txt')]  # Remove .txt
    if base_filename.startswith('sensitive_asins_'):
        category = base_filename[len('sensitive_asins_'):]
    else:
        # Fallback for backwards compatibility
        category = base_filename.split("_")[-1]
    forget_file = f"{args.dataset[:-len('.inter')]}_unlearn_pairs_sensitive_category_{category}_seed_{args.seed}_unlearning_fraction_{args.forget_ratio}.inter"
    
    write_dataset(forget_set, forget_file)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original dataset: {total_count} interactions")
    print(f"Forget set: {len(forget_set)} interactions ({len(forget_set)/total_count*100:.3f}%)")
    print(f"Forget set users: {len(set(row['user_id'] for row in forget_set))}")
    print(f"Forget set items: {len(set(row['item_id'] for row in forget_set))}")
    print(f"\nOutput files:")
    print(f"  - {forget_file}")
    print("="*60)


if __name__ == "__main__":
    main()
