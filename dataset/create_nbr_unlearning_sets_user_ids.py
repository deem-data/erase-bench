"""
Create NBR unlearning sets that save only user IDs (not user-item pairs).

This script generates files containing lists of user IDs for sensitive category-based
unlearning in NBR models. It's designed to match the naming convention used in quick_start.py.

Key differences from create_nbr_unlearning_sets_merged.py:
- Saves only user IDs, not user-to-items mappings
- Filters users to ensure they still have >= 4 baskets after sensitive item removal
- Uses naming convention: {dataset}_unlearn_pairs_sensitive_category_{category}_seed_{seed}_unlearning_fraction_{fraction}
- Supports both .pkl and .json output formats
"""

import json
import numpy as np
import argparse
import random
import pickle
import os
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create NBR unlearning sets (user IDs only) for sensitive categories"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["instacart", "tafeng", "dunnhumby"],
        help="Dataset name"
    )
    parser.add_argument(
        "--unlearning_fraction",
        type=float,
        default=0.001,
        help="Fraction of total items to unlearn (default: 0.001 = 0.1%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--merged_file",
        type=str,
        default=None,
        help="Path to merged JSON file (if not specified, will look in dataset/{dataset}/{dataset}_merged.json)"
    )
    parser.add_argument(
        "--nbr_repo_path",
        type=str,
        default="/home/pierre/workspace/forks/A-Next-Basket-Recommendation-Reality-Check",
        help="Path to the NBR repository (for product metadata files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for unlearning files (default: dataset root, from script location)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pkl", "json", "both"],
        default="both",
        help="Output file format (default: both)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to process (default: all available for dataset)"
    )
    return parser.parse_args()


def get_sensitive_categories(dataset):
    """
    Define sensitive categories and their identifiers for each dataset.

    For Instacart, categories are defined by aisle IDs.
    For Ta-Feng and Dunnhumby, this needs to be defined based on product categories.
    """
    if dataset == "instacart":
        return {
            "aisle_id_mapping": {
                "meat": [5, 95, 96, 15, 33, 34, 35, 49, 106, 122],
                "alcohol": [27, 28, 62, 124, 134],
                "baby": [82, 92, 102, 56],
            },
            "products_file": "instacart_products.csv"
        }
    elif dataset == "tafeng":
        # TODO: Define Ta-Feng sensitive categories
        return {
            "category_mapping": {
                "alcohol": [],
                "meat": [],
                "baby": [],
            },
            "products_file": None
        }
    elif dataset == "dunnhumby":
        # For Dunnhumby, we can load from the sensitive_products_alcohol.txt file
        return {
            "use_product_list": True,
            "sensitive_files": {
                "alcohol": "sensitive_products_alcohol.txt",
                # Add other categories as needed
            },
            "products_file": None
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_product_mappings(dataset, nbr_repo_path, dataset_dir):
    """
    Load product metadata and create mapping from sensitive categories to product IDs.
    """
    sensitive_config = get_sensitive_categories(dataset)

    if dataset == "instacart":
        products_path = os.path.join(nbr_repo_path, "dataset", sensitive_config["products_file"])

        if not os.path.exists(products_path):
            print(f"ERROR: Products file not found at {products_path}")
            return {}

        products = pd.read_csv(products_path)

        category_to_items = {
            cat: set(products[products["aisle_id"].isin(aisle_ids)]["product_id"])
            for cat, aisle_ids in sensitive_config["aisle_id_mapping"].items()
        }
        return category_to_items

    elif dataset == "dunnhumby":
        # Load from sensitive product files in the dataset directory
        if not sensitive_config.get("use_product_list"):
            print(f"WARNING: Product mappings for {dataset} not yet implemented.")
            return {}

        category_to_items = {}
        for cat, filename in sensitive_config["sensitive_files"].items():
            filepath = os.path.join(dataset_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    # Each line contains a product ID
                    items = set(int(line.strip()) for line in f if line.strip())
                category_to_items[cat] = items
                print(f"Loaded {len(items)} {cat} products from {filename}")
            else:
                print(f"WARNING: Sensitive products file not found: {filepath}")

        return category_to_items

    elif dataset == "tafeng":
        # TODO: Implement product mapping for Ta-Feng
        print(f"WARNING: Product mappings for {dataset} not yet implemented.")
        return {}

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_sensitive_unlearning_user_ids(merged_data, category_to_items,
                                         unlearning_fraction, seed):
    """
    Create unlearning sets containing only user IDs (not user-item pairs).

    Users are selected if:
    1. They have purchased items from the sensitive category
    2. After removing all sensitive items, they still have >= 4 baskets

    Args:
        merged_data: Dict mapping user_id -> list of baskets
        category_to_items: Dict mapping category name -> set of product IDs
        unlearning_fraction: Fraction of total items to unlearn
        seed: Random seed

    Returns:
        Dict mapping category -> list of user IDs to unlearn
    """
    random.seed(seed)
    np.random.seed(seed)

    # Filter users with at least 4 baskets
    user_list = [
        user for user, baskets in merged_data.items()
        if len(baskets) >= 4
    ]

    # Compute total item count across all baskets
    total_item_count = sum(
        len(basket)
        for user in user_list
        for basket in merged_data[user]
    )

    wanted_unlearning_count = int(unlearning_fraction * total_item_count)

    print(f"Total users: {len(user_list)}")
    print(f"Total item count: {total_item_count}")
    print(f"Unlearning target: {wanted_unlearning_count} items")

    category_user_ids = {}

    for category, sensitive_product_ids in category_to_items.items():
        print(f"\n[{category}] Processing category with {len(sensitive_product_ids)} products...")

        # Find eligible users: have sensitive items AND would still have >= 4 baskets after removal
        eligible_users = {}

        for user in user_list:
            user_sensitive_items = []
            clean_basket_count = 0

            # Count how many baskets would remain after removing sensitive items
            for basket in merged_data[user]:
                basket_sensitive_items = set(basket) & sensitive_product_ids
                user_sensitive_items.extend(basket_sensitive_items)

                # Check if basket would remain non-empty after removal
                clean_basket = [item for item in basket if item not in sensitive_product_ids]
                if len(clean_basket) > 0:
                    clean_basket_count += 1

            # User is eligible if they have sensitive items AND >= 4 clean baskets
            if len(user_sensitive_items) > 0 and clean_basket_count >= 4:
                eligible_users[user] = len(user_sensitive_items)

        print(f"[{category}] Eligible users (have sensitive items + >= 4 baskets after removal): {len(eligible_users)}")

        # Sample users until we reach the unlearning threshold
        selected_user_ids = []
        total_unlearned = 0
        user_pool = list(eligible_users.keys())
        random.shuffle(user_pool)

        for user in user_pool:
            item_count = eligible_users[user]
            selected_user_ids.append(user)
            total_unlearned += item_count
            if total_unlearned >= wanted_unlearning_count:
                break

        print(f"[{category}] Selected {len(selected_user_ids)} users, "
              f"total sensitive items: {total_unlearned}")

        category_user_ids[category] = selected_user_ids

    return category_user_ids


def save_unlearning_data(category_user_ids, dataset, seed, unlearning_fraction,
                         output_dir, output_format):
    """
    Save unlearning data in the specified format(s).

    Naming convention: {dataset}_unlearn_pairs_sensitive_category_{category}_seed_{seed}_unlearning_fraction_{fraction}.{ext}
    """
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    for category, user_ids in category_user_ids.items():
        base_name = (f"{dataset}_unlearn_pairs_sensitive_category_{category}"
                    f"_seed_{seed}_unlearning_fraction_{unlearning_fraction}")

        # Save as pickle
        if output_format in ["pkl", "both"]:
            pkl_path = os.path.join(output_dir, f"{base_name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(user_ids, f)
            print(f"Saved: {pkl_path}")
            saved_files.append(pkl_path)

        # Save as JSON
        if output_format in ["json", "both"]:
            json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(json_path, "w") as f:
                json.dump(user_ids, f, indent=2)
            print(f"Saved: {json_path}")
            saved_files.append(json_path)

    return saved_files


def _get_dataset_root():
    """Dataset root: script dir if script lives in dataset/, else script_dir/dataset."""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(_script_dir) == "dataset":
        return _script_dir
    return os.path.join(_script_dir, "dataset")


def main():
    args = parse_args()

    # Resolve paths from script location so script works from any CWD (e.g. in dataset/)
    dataset_root = _get_dataset_root()
    if args.output_dir is None:
        args.output_dir = dataset_root

    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Determine merged file path
    if args.merged_file:
        merged_file = args.merged_file
    else:
        merged_file = os.path.join(dataset_root, args.dataset, f"{args.dataset}_merged.json")

    if not os.path.exists(merged_file):
        print(f"ERROR: Merged file not found at {merged_file}")
        print(f"Please specify --merged_file or ensure the file exists at the default location")
        return

    print(f"Loading merged data from: {merged_file}")
    with open(merged_file, 'r') as f:
        merged_data = json.load(f)

    print(f"Loaded {len(merged_data)} users")

    # Show sample user data
    sample_user = list(merged_data.keys())[0]
    print(f"Sample user {sample_user} has {len(merged_data[sample_user])} baskets")
    print(f"  First basket: {merged_data[sample_user][0][:10]}...")

    # Load product mappings for sensitive categories
    dataset_dir = os.path.join(dataset_root, args.dataset)
    print(f"\nLoading product mappings for {args.dataset}...")
    category_to_items = load_product_mappings(args.dataset, args.nbr_repo_path, dataset_dir)

    if not category_to_items:
        print(f"\nERROR: No sensitive categories defined for {args.dataset}")
        return

    # Filter categories if specified
    if args.categories:
        category_to_items = {
            cat: items for cat, items in category_to_items.items()
            if cat in args.categories
        }
        if not category_to_items:
            print(f"ERROR: None of the specified categories {args.categories} found.")
            return

    print(f"\nSensitive categories to process: {list(category_to_items.keys())}")
    for cat, items in category_to_items.items():
        print(f"  {cat}: {len(items)} products")

    # Create unlearning sets (user IDs only)
    print(f"\nCreating unlearning sets (user IDs only)...")
    category_user_ids = create_sensitive_unlearning_user_ids(
        merged_data,
        category_to_items,
        args.unlearning_fraction,
        args.seed
    )

    # Save to files
    output_dir = os.path.join(args.output_dir, args.dataset)
    print(f"\nSaving files to: {output_dir}")

    saved_files = save_unlearning_data(
        category_user_ids,
        args.dataset,
        args.seed,
        args.unlearning_fraction,
        output_dir,
        args.output_format
    )

    print("\n" + "="*80)
    print("SUCCESS! Unlearning sets created successfully.")
    print("="*80)
    print(f"\nGenerated {len(saved_files)} file(s):")
    for path in saved_files:
        print(f"  - {path}")

    print("\nTo load and use the data:")
    print("```python")
    if args.output_format in ["pkl", "both"]:
        print("# Load from pickle:")
        print("import pickle")
        print(f"with open('{saved_files[0]}', 'rb') as f:")
        print(f"    user_ids = pickle.load(f)")
        print(f"    # user_ids is a list of user IDs: ['user1', 'user2', ...]")
    if args.output_format in ["json", "both"]:
        json_idx = 1 if args.output_format == "both" else 0
        print("\n# Load from JSON:")
        print("import json")
        print(f"with open('{saved_files[json_idx] if json_idx < len(saved_files) else saved_files[0]}', 'r') as f:")
        print(f"    user_ids = json.load(f)")
    print("```")

    # Print summary statistics
    print("\nSummary:")
    for category, user_ids in category_user_ids.items():
        print(f"  {category}: {len(user_ids)} users")


if __name__ == "__main__":
    main()
