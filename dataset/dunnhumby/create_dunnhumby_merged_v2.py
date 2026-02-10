#!/usr/bin/env python3
"""
Create dunnhumby_merged_v2.json from transaction_data.csv

This script processes the raw Dunnhumby transaction data and creates a JSON file
in the same format as dunnhumby_merged.json, with the following structure:
{
    "user_id": [[product_ids_in_basket_1], [product_ids_in_basket_2], ...],
    ...
}

Mapping:
- household_key -> user_id
- BASKET_ID -> session/basket identifier
- PRODUCT_ID -> item_id
- TRANS_TIME -> used for ordering baskets within a user
- QUANTITY is ignored (only presence matters)
"""

import pandas as pd
import json
from collections import defaultdict
from pathlib import Path

# Configuration
DATASET_DIR = Path(__file__).parent
TRANSACTION_FILE = DATASET_DIR / "transaction_data.csv"
OUTPUT_FILE = DATASET_DIR / "dunnhumby_merged_v2.json"
MIN_BASKETS_PER_USER = 4  # Minimum number of baskets a user must have to be included

# Read transaction data
print(f"Reading transaction data from {TRANSACTION_FILE}...")
df = pd.read_csv(TRANSACTION_FILE)

print(f"Loaded {len(df)} transactions")
print(f"Number of unique households: {df['household_key'].nunique()}")
print(f"Number of unique baskets: {df['BASKET_ID'].nunique()}")
print(f"Number of unique products: {df['PRODUCT_ID'].nunique()}")

# Group by household and basket to create sessions
print("\nGrouping transactions by household and basket...")

# Create a dictionary to store user -> list of baskets
user_baskets = defaultdict(list)

# Group by household_key first
for household_key, household_df in df.groupby('household_key'):
    # For each household, group by BASKET_ID and get the average TRANS_TIME for ordering
    basket_info = []

    for basket_id, basket_df in household_df.groupby('BASKET_ID'):
        # Get all products in this basket (deduplicated)
        products = basket_df['PRODUCT_ID'].unique().tolist()

        # Get the average TRANS_TIME for this basket (for ordering)
        avg_trans_time = basket_df['TRANS_TIME'].mean()

        basket_info.append({
            'basket_id': basket_id,
            'trans_time': avg_trans_time,
            'products': products
        })

    # Sort baskets by TRANS_TIME
    basket_info.sort(key=lambda x: x['trans_time'])

    # Extract just the product lists in order
    user_baskets[str(household_key)] = [b['products'] for b in basket_info]

print(f"\nCreated {len(user_baskets)} user sessions (before filtering)")

# Filter users with at least MIN_BASKETS_PER_USER baskets
print(f"Filtering users with at least {MIN_BASKETS_PER_USER} baskets...")
user_baskets = {
    user_id: baskets
    for user_id, baskets in user_baskets.items()
    if len(baskets) >= MIN_BASKETS_PER_USER
}

print(f"After filtering: {len(user_baskets)} users remaining")

# Calculate statistics
total_baskets = sum(len(baskets) for baskets in user_baskets.values())
avg_baskets_per_user = total_baskets / len(user_baskets)
total_interactions = sum(sum(len(basket) for basket in baskets) for baskets in user_baskets.values())
avg_products_per_basket = total_interactions / total_baskets

print(f"Total baskets: {total_baskets}")
print(f"Average baskets per user: {avg_baskets_per_user:.2f}")
print(f"Total interactions: {total_interactions}")
print(f"Average products per basket: {avg_products_per_basket:.2f}")

# Save to JSON
print(f"\nSaving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(user_baskets, f)

print("Done!")

# Print a sample entry
sample_user = list(user_baskets.keys())[0]
print(f"\nSample entry for user {sample_user}:")
print(f"  Number of baskets: {len(user_baskets[sample_user])}")
print(f"  First basket products: {user_baskets[sample_user][0][:10]}...")
print(f"  Number of products in first basket: {len(user_baskets[sample_user][0])}")
